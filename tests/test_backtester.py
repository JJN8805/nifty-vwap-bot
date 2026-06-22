import pandas as pd

from bot.backtester import run_backtest
from bot.config import CapitalConfig, InstrumentConfig, StrategyConfig
from bot.signals import compute_signals
from tests.synthetic_data import make_synthetic_session


def test_backtest_runs_end_to_end():
    df = make_synthetic_session(n_days=10)
    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    signals = compute_signals(df, strat_cfg)
    result = run_backtest(signals, strat_cfg, instr_cfg, cap_cfg)

    summary = result.summary()
    assert "total_trades" in summary
    assert len(result.equity_curve) == len(df)


def test_max_trades_per_day_respected():
    df = make_synthetic_session(n_days=10)
    strat_cfg = StrategyConfig(max_trades_per_day=2)
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    signals = compute_signals(df, strat_cfg)
    result = run_backtest(signals, strat_cfg, instr_cfg, cap_cfg)

    entries_per_day = {}
    for t in result.trades:
        day = t.entry_time.date()
        entries_per_day[day] = entries_per_day.get(day, 0) + 1

    for day, count in entries_per_day.items():
        assert count <= strat_cfg.max_trades_per_day, f"{day} had {count} trades, exceeds cap"


def test_no_overlapping_positions():
    """At most one open position at a time -- matches no-pyramiding rule."""
    df = make_synthetic_session(n_days=10)
    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    signals = compute_signals(df, strat_cfg)
    result = run_backtest(signals, strat_cfg, instr_cfg, cap_cfg)

    sorted_trades = sorted(result.trades, key=lambda t: t.entry_time)
    for prev, curr in zip(sorted_trades, sorted_trades[1:]):
        assert prev.exit_time <= curr.entry_time, "Found overlapping trades"


def test_force_exit_before_1500_never_held_past_close():
    df = make_synthetic_session(n_days=5)
    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    signals = compute_signals(df, strat_cfg)
    result = run_backtest(signals, strat_cfg, instr_cfg, cap_cfg)

    for t in result.trades:
        if t.exit_reason not in ("end_of_data",):
            assert t.exit_time.hour < 15 or (t.exit_time.hour == 15 and t.exit_time.minute <= 5), (
                f"Trade exited late: {t.exit_time} reason={t.exit_reason}"
            )


def test_pnl_inr_consistent_with_points_and_lot_size():
    df = make_synthetic_session(n_days=10)
    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig(lots_per_trade=1)

    signals = compute_signals(df, strat_cfg)
    result = run_backtest(signals, strat_cfg, instr_cfg, cap_cfg)

    for t in result.trades:
        expected = t.pnl_points * instr_cfg.lot_size - instr_cfg.est_cost_per_lot_round_trip
        assert abs(t.pnl_inr - expected) < 1e-6


def test_ambiguous_bar_resolves_to_original_stop_not_breakeven():
    """
    Regression test: if a single candle's range touches BOTH the
    breakeven-trigger level (1R favourable) AND the original stop level,
    the backtester must assume the worse-case ordering (original stop
    hit first) rather than optimistically granting breakeven. Silently
    picking the favourable interpretation would understate real risk.
    """
    idx = pd.date_range("2026-01-05 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        index=idx,
        columns=["open", "high", "low", "close", "long_entry", "short_entry", "risk"],
    )
    # entry bar: close=100.5, risk=5 -> stop=95.5, target(RR=2.5)=113.0
    df.loc[idx[0]] = [100, 101, 99, 100.5, True, False, 5.0]
    # ambiguous bar: high=113 triggers breakeven (1R=105.5 reached, even target reached),
    # low=95.0 also breaches the original stop of 95.5
    df.loc[idx[1]] = [100.5, 113.0, 95.0, 105, False, False, 5.0]
    df.loc[idx[2]] = [105, 106, 104, 105, False, False, 5.0]

    for c in ["open", "high", "low", "close", "risk"]:
        df[c] = df[c].astype(float)
    df["long_entry"] = df["long_entry"].astype(bool)
    df["short_entry"] = df["short_entry"].astype(bool)

    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    result = run_backtest(df, strat_cfg, instr_cfg, cap_cfg)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.breakeven_active is False
    assert trade.exit_price == 95.5
    assert trade.pnl_points == -5.0
