"""
Paper trading runner.

Designed to be invoked on a schedule (every 5 minutes during market
hours, via GitHub Actions cron) as a stateless process that:

  1. Loads persisted state (open position, daily trade count, equity)
     from a JSON file committed back to the repo.
  2. Fetches the latest candles.
  3. Recomputes signals.
  4. Applies exactly one bar's worth of decision logic (the most recent
     closed candle) -- entry, stop/target/breakeven check, or forced
     exit -- exactly like one iteration of the backtester's loop. The
     exit decision itself is delegated to backtester.evaluate_exit so
     paper trading and backtesting can never silently diverge.
  5. Persists updated state and appends to a trade log CSV.

State is stored as plain JSON/CSV (not a database) deliberately, so it
diffs cleanly in git and needs no extra infrastructure -- consistent
with running this entirely inside GitHub Actions with no server.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtester import Trade, evaluate_exit
from .config import USE_KITE, CapitalConfig, InstrumentConfig, StrategyConfig

if USE_KITE:
    from .data_feed_kite import fetch_and_cache, filter_market_hours
else:
    from .data_feed import fetch_and_cache, filter_market_hours

from .signals import compute_signals

logger = logging.getLogger(__name__)


STATE_FILE = Path("data/paper_state.json")
TRADE_LOG_FILE = Path("reports/paper_trade_log.csv")
EQUITY_LOG_FILE = Path("reports/paper_equity_log.csv")


def _load_state(cap_cfg: CapitalConfig) -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        # Backward-compat: older state files (written before
        # original_stop_price was tracked) may have an open_trade dict
        # missing this key. Falling back to the current stop_price is
        # the best available approximation -- it only matters for one
        # bar's worth of same-bar stop/breakeven ambiguity resolution,
        # and only for a trade that was already open when the code
        # changed.
        if state.get("open_trade") is not None and "original_stop_price" not in state["open_trade"]:
            state["open_trade"]["original_stop_price"] = state["open_trade"]["stop_price"]
        return state
    return {
        "equity": cap_cfg.initial_capital,
        "daily_trade_count": {},
        "open_trade": None,
        "last_processed_bar": None,
    }


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _append_trade_log(trade: Trade) -> None:
    TRADE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(trade)])
    if TRADE_LOG_FILE.exists():
        row.to_csv(TRADE_LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(TRADE_LOG_FILE, mode="w", header=True, index=False)


def _append_equity_log(timestamp, equity: float) -> None:
    EQUITY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"timestamp": timestamp, "equity": equity}])
    if EQUITY_LOG_FILE.exists():
        row.to_csv(EQUITY_LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(EQUITY_LOG_FILE, mode="w", header=True, index=False)


def run_paper_trading_step(
    strat_cfg: StrategyConfig,
    instr_cfg: InstrumentConfig,
    cap_cfg: CapitalConfig,
    cache_dir: Path = Path("data/cache"),
) -> None:
    state = _load_state(cap_cfg)

    raw = fetch_and_cache(instr_cfg.symbol, strat_cfg.ltf_interval, period="60d", cache_dir=cache_dir)
    raw = filter_market_hours(
        raw,
        strat_cfg.market_open_hour,
        strat_cfg.market_open_minute,
        strat_cfg.market_close_hour,
        strat_cfg.market_close_minute,
    )

    if raw.empty or len(raw) < max(strat_cfg.ema_slow, strat_cfg.atr_len) + 5:
        logger.warning("Not enough data to evaluate signals yet (%d bars).", len(raw))
        return

    df = compute_signals(raw, strat_cfg)

    # Only act on the most recently *closed* bar to avoid acting on a
    # still-forming candle if this script happens to run mid-bar.
    latest_closed = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]
    ts = df.index[-2] if len(df) >= 2 else df.index[-1]

    if state["last_processed_bar"] == str(ts):
        logger.info("Bar %s already processed -- nothing new to do.", ts)
        return

    day_key = str(ts.date())
    state["daily_trade_count"].setdefault(day_key, 0)

    point_value = instr_cfg.lot_size * cap_cfg.lots_per_trade
    round_trip_cost = instr_cfg.est_cost_per_lot_round_trip * cap_cfg.lots_per_trade

    open_trade = state["open_trade"]

    # ---- manage existing open paper position ----
    if open_trade is not None:
        is_forced_exit = ts.hour == strat_cfg.force_exit_hour and ts.minute >= strat_cfg.force_exit_minute

        result = evaluate_exit(
            direction=open_trade["direction"],
            entry_price=open_trade["entry_price"],
            stop_price=open_trade["stop_price"],
            original_stop_price=open_trade["original_stop_price"],
            target_price=open_trade["target_price"],
            breakeven_active=open_trade["breakeven_active"],
            bar_high=latest_closed["high"],
            bar_low=latest_closed["low"],
            bar_close=latest_closed["close"],
            is_forced_exit=is_forced_exit,
        )
        open_trade["stop_price"] = result["stop_price"]
        open_trade["breakeven_active"] = result["breakeven_active"]

        if result["exit_reason"] is not None:
            exit_price = result["exit_price"]
            pnl_points = (
                exit_price - open_trade["entry_price"]
                if open_trade["direction"] == "long"
                else open_trade["entry_price"] - exit_price
            )
            pnl_inr = pnl_points * point_value - round_trip_cost
            state["equity"] += pnl_inr

            closed_trade = Trade(
                direction=open_trade["direction"],
                entry_time=open_trade["entry_time"],
                entry_price=open_trade["entry_price"],
                exit_time=str(ts),
                exit_price=exit_price,
                exit_reason=result["exit_reason"],
                stop_price=open_trade["stop_price"],
                target_price=open_trade["target_price"],
                original_stop_price=open_trade["original_stop_price"],
                breakeven_active=open_trade["breakeven_active"],
                pnl_points=pnl_points,
                pnl_inr=pnl_inr,
            )
            _append_trade_log(closed_trade)
            logger.info("Closed paper trade: %s", closed_trade)
            open_trade = None

    # ---- new entry, only if flat and under daily cap ----
    if (
        open_trade is None
        and state["daily_trade_count"][day_key] < strat_cfg.max_trades_per_day
        and not pd.isna(latest_closed.get("risk", float("nan")))
    ):
        if latest_closed.get("long_entry", False):
            risk = latest_closed["risk"]
            entry_price = latest_closed["close"]
            open_trade = {
                "direction": "long",
                "entry_time": str(ts),
                "entry_price": entry_price,
                "stop_price": entry_price - risk,
                "original_stop_price": entry_price - risk,
                "target_price": entry_price + risk * strat_cfg.reward_risk,
                "breakeven_active": False,
            }
            state["daily_trade_count"][day_key] += 1
            logger.info("Opened paper LONG at %s, price=%.2f", ts, entry_price)

        elif latest_closed.get("short_entry", False):
            risk = latest_closed["risk"]
            entry_price = latest_closed["close"]
            open_trade = {
                "direction": "short",
                "entry_time": str(ts),
                "entry_price": entry_price,
                "stop_price": entry_price + risk,
                "original_stop_price": entry_price + risk,
                "target_price": entry_price - risk * strat_cfg.reward_risk,
                "breakeven_active": False,
            }
            state["daily_trade_count"][day_key] += 1
            logger.info("Opened paper SHORT at %s, price=%.2f", ts, entry_price)

    state["open_trade"] = open_trade
    state["last_processed_bar"] = str(ts)

    # Trim daily_trade_count to last 10 days to keep state file small
    if len(state["daily_trade_count"]) > 10:
        keys_sorted = sorted(state["daily_trade_count"].keys())
        for k in keys_sorted[:-10]:
            del state["daily_trade_count"][k]

    _save_state(state)
    _append_equity_log(ts, state["equity"])
    logger.info("Paper trading step complete. Equity: %.2f", state["equity"])
