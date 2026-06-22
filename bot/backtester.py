"""
Bar-by-bar backtest engine that replicates the Pine script's exit logic:

- ATR stop = entry +/- (ATR * atr_stop_mult)
- Target   = entry +/- (ATR * atr_stop_mult * reward_risk)
- Breakeven: once price has moved favourably by one full risk-unit,
  the stop is moved to entry price (matches the Pine `beActive` logic).
- Forced flat at `force_exit_hour:force_exit_minute` regardless of P&L.
- Max `max_trades_per_day` new entries per day.
- No pyramiding: at most one open position at a time (matches
  `strategy.position_size == 0` gating in the Pine script).

The exit decision itself (stop/target/breakeven/force-exit) lives in
`evaluate_exit()` below, which is also imported and reused by
bot/paper_trader.py. Keeping this logic in one place means a fix here
automatically applies to paper trading too, instead of the two
diverging silently.

This intentionally mirrors the original bar-close decision structure
rather than trying to be a more "realistic" tick-level simulator -- the
goal here is strategy-logic parity with the source Pine script, not a
generic backtesting framework.
"""

from dataclasses import dataclass, field

import pandas as pd

from .config import CapitalConfig, InstrumentConfig, StrategyConfig


@dataclass
class Trade:
    direction: str  # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    stop_price: float = 0.0
    target_price: float = 0.0
    original_stop_price: float = 0.0  # stop level before any breakeven adjustment
    breakeven_active: bool = False
    pnl_points: float = 0.0
    pnl_inr: float = 0.0


def evaluate_exit(
    direction: str,
    entry_price: float,
    stop_price: float,
    original_stop_price: float,
    target_price: float,
    breakeven_active: bool,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    is_forced_exit: bool,
) -> dict:
    """
    Single source of truth for the strategy's exit decision on one bar,
    shared by both the backtester and the paper trader so the two can
    never silently drift apart on this logic.

    Returns a dict with possibly-updated stop_price/breakeven_active,
    plus exit_reason/exit_price (both None if the trade stays open).

    Same-bar sequencing note: if a single candle's range touches BOTH
    the original stop level and the breakeven-trigger level (1R
    favourable), we cannot know from OHLC alone which happened first
    intrabar. We assume the worse-case ordering -- original stop hit
    first -- rather than optimistically granting breakeven, since
    assuming the favourable path happened first would understate real
    risk.
    """
    exit_reason = None
    exit_price = None

    if direction == "long":
        breakeven_trigger_price = entry_price + (entry_price - original_stop_price)
        stop_hit_this_bar = (not breakeven_active) and (bar_low <= original_stop_price)

        if not breakeven_active and not stop_hit_this_bar and bar_high >= breakeven_trigger_price:
            breakeven_active = True
            stop_price = entry_price

        if bar_low <= stop_price:
            exit_reason = "breakeven_stop" if breakeven_active else "stop_loss"
            exit_price = stop_price
        elif bar_high >= target_price:
            exit_reason = "target"
            exit_price = target_price
        elif is_forced_exit:
            exit_reason = "force_exit_1500"
            exit_price = bar_close

    else:  # short
        breakeven_trigger_price = entry_price - (original_stop_price - entry_price)
        stop_hit_this_bar = (not breakeven_active) and (bar_high >= original_stop_price)

        if not breakeven_active and not stop_hit_this_bar and bar_low <= breakeven_trigger_price:
            breakeven_active = True
            stop_price = entry_price

        if bar_high >= stop_price:
            exit_reason = "breakeven_stop" if breakeven_active else "stop_loss"
            exit_price = stop_price
        elif bar_low <= target_price:
            exit_reason = "target"
            exit_price = target_price
        elif is_forced_exit:
            exit_reason = "force_exit_1500"
            exit_price = bar_close

    return {
        "stop_price": stop_price,
        "breakeven_active": breakeven_active,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
    }


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> dict:
        closed = [t for t in self.trades if t.exit_price is not None]
        if not closed:
            return {"total_trades": 0}

        wins = [t for t in closed if t.pnl_inr > 0]
        losses = [t for t in closed if t.pnl_inr <= 0]
        gross_profit = sum(t.pnl_inr for t in wins)
        gross_loss = sum(t.pnl_inr for t in losses)
        total_pnl = gross_profit + gross_loss

        equity = self.equity_curve
        running_max = equity.cummax() if not equity.empty else pd.Series(dtype=float)
        drawdown = (equity - running_max) if not equity.empty else pd.Series(dtype=float)
        max_dd = drawdown.min() if not drawdown.empty else 0.0

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100 * len(wins) / len(closed), 2) if closed else 0,
            "gross_profit_inr": round(gross_profit, 2),
            "gross_loss_inr": round(gross_loss, 2),
            "net_pnl_inr": round(total_pnl, 2),
            "profit_factor": round(gross_profit / abs(gross_loss), 2) if gross_loss != 0 else None,
            "max_drawdown_inr": round(max_dd, 2),
            "avg_win_inr": round(gross_profit / len(wins), 2) if wins else 0,
            "avg_loss_inr": round(gross_loss / len(losses), 2) if losses else 0,
        }


def run_backtest(
    df: pd.DataFrame,
    strat_cfg: StrategyConfig,
    instr_cfg: InstrumentConfig,
    cap_cfg: CapitalConfig,
) -> BacktestResult:
    """
    df must already contain signal columns from signals.compute_signals().
    Expects a tz-aware DatetimeIndex (Asia/Kolkata).
    """
    trades: list[Trade] = []
    open_trade: Trade | None = None
    daily_trade_count: dict = {}

    equity = cap_cfg.initial_capital
    equity_curve = []

    point_value = instr_cfg.lot_size * cap_cfg.lots_per_trade
    round_trip_cost = instr_cfg.est_cost_per_lot_round_trip * cap_cfg.lots_per_trade

    for ts, row in df.iterrows():
        day_key = ts.date()
        daily_trade_count.setdefault(day_key, 0)

        # ----- manage open trade -----
        if open_trade is not None:
            is_forced_exit = (
                ts.hour == strat_cfg.force_exit_hour and ts.minute >= strat_cfg.force_exit_minute
            )

            result = evaluate_exit(
                direction=open_trade.direction,
                entry_price=open_trade.entry_price,
                stop_price=open_trade.stop_price,
                original_stop_price=open_trade.original_stop_price,
                target_price=open_trade.target_price,
                breakeven_active=open_trade.breakeven_active,
                bar_high=row["high"],
                bar_low=row["low"],
                bar_close=row["close"],
                is_forced_exit=is_forced_exit,
            )
            open_trade.stop_price = result["stop_price"]
            open_trade.breakeven_active = result["breakeven_active"]

            if result["exit_reason"] is not None:
                exit_price = result["exit_price"]
                open_trade.exit_time = ts
                open_trade.exit_price = exit_price
                open_trade.exit_reason = result["exit_reason"]

                if open_trade.direction == "long":
                    open_trade.pnl_points = exit_price - open_trade.entry_price
                else:
                    open_trade.pnl_points = open_trade.entry_price - exit_price

                open_trade.pnl_inr = (open_trade.pnl_points * point_value) - round_trip_cost
                equity += open_trade.pnl_inr
                trades.append(open_trade)
                open_trade = None

        # ----- new entries (only if flat, within daily cap) -----
        if (
            open_trade is None
            and daily_trade_count[day_key] < strat_cfg.max_trades_per_day
            and not pd.isna(row.get("risk", float("nan")))
        ):
            if row.get("long_entry", False):
                risk = row["risk"]
                entry_price = row["close"]
                open_trade = Trade(
                    direction="long",
                    entry_time=ts,
                    entry_price=entry_price,
                    stop_price=entry_price - risk,
                    original_stop_price=entry_price - risk,
                    target_price=entry_price + risk * strat_cfg.reward_risk,
                )
                daily_trade_count[day_key] += 1

            elif row.get("short_entry", False):
                risk = row["risk"]
                entry_price = row["close"]
                open_trade = Trade(
                    direction="short",
                    entry_time=ts,
                    entry_price=entry_price,
                    stop_price=entry_price + risk,
                    original_stop_price=entry_price + risk,
                    target_price=entry_price - risk * strat_cfg.reward_risk,
                )
                daily_trade_count[day_key] += 1

        equity_curve.append((ts, equity))

    # Force-close any trade still open at the end of the data window
    if open_trade is not None:
        last_ts = df.index[-1]
        last_close = df.iloc[-1]["close"]
        open_trade.exit_time = last_ts
        open_trade.exit_price = last_close
        open_trade.exit_reason = "end_of_data"
        if open_trade.direction == "long":
            open_trade.pnl_points = last_close - open_trade.entry_price
        else:
            open_trade.pnl_points = open_trade.entry_price - last_close
        open_trade.pnl_inr = (open_trade.pnl_points * point_value) - round_trip_cost
        trades.append(open_trade)

    equity_series = pd.Series(
        [e for _, e in equity_curve], index=[t for t, _ in equity_curve]
    )

    return BacktestResult(trades=trades, equity_curve=equity_series)
