"""
Tests for the paper trading step logic using synthetic data, run
against a temp directory so we don't touch real state/report files.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from bot import paper_trader
from bot.config import CapitalConfig, InstrumentConfig, StrategyConfig
from tests.synthetic_data import make_synthetic_session


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    state_file = tmp_path / "paper_state.json"
    trade_log = tmp_path / "paper_trade_log.csv"
    equity_log = tmp_path / "paper_equity_log.csv"

    monkeypatch.setattr(paper_trader, "STATE_FILE", state_file)
    monkeypatch.setattr(paper_trader, "TRADE_LOG_FILE", trade_log)
    monkeypatch.setattr(paper_trader, "EQUITY_LOG_FILE", equity_log)

    return state_file, trade_log, equity_log


def test_paper_trading_step_runs_and_persists_state(isolated_paths, monkeypatch):
    state_file, trade_log, equity_log = isolated_paths

    synthetic = make_synthetic_session(n_days=10)

    def fake_fetch_and_cache(symbol, interval, period, cache_dir):
        return synthetic

    monkeypatch.setattr(paper_trader, "fetch_and_cache", fake_fetch_and_cache)

    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    paper_trader.run_paper_trading_step(strat_cfg, instr_cfg, cap_cfg, cache_dir=Path("unused"))

    assert state_file.exists()
    with open(state_file) as f:
        state = json.load(f)
    assert "equity" in state
    assert "last_processed_bar" in state
    assert state["last_processed_bar"] is not None


def test_paper_trading_step_is_idempotent_for_same_bar(isolated_paths, monkeypatch):
    """Running twice against the same data shouldn't double-process the same bar."""
    state_file, trade_log, equity_log = isolated_paths
    synthetic = make_synthetic_session(n_days=10)

    def fake_fetch_and_cache(symbol, interval, period, cache_dir):
        return synthetic

    monkeypatch.setattr(paper_trader, "fetch_and_cache", fake_fetch_and_cache)

    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    paper_trader.run_paper_trading_step(strat_cfg, instr_cfg, cap_cfg, cache_dir=Path("unused"))
    with open(state_file) as f:
        state_after_first = json.load(f)

    paper_trader.run_paper_trading_step(strat_cfg, instr_cfg, cap_cfg, cache_dir=Path("unused"))
    with open(state_file) as f:
        state_after_second = json.load(f)

    assert state_after_first["last_processed_bar"] == state_after_second["last_processed_bar"]
    assert state_after_first["equity"] == state_after_second["equity"]


def test_open_trade_with_original_stop_price_resolves_ambiguous_bar_conservatively(
    isolated_paths, monkeypatch
):
    """
    Regression test for the shared-logic refactor: construct a known
    open paper position (with original_stop_price set, as new code
    always sets it) and a single ambiguous bar that touches both the
    breakeven trigger and the original stop. The paper trader must
    resolve this the same conservative way the backtester does --
    stop_loss, not breakeven_stop -- proving the two code paths are
    now backed by the same evaluate_exit logic instead of two
    independently-written (and previously divergent) copies.
    """
    state_file, trade_log, equity_log = isolated_paths

    # Build 1 day of synthetic bars, but override the last two rows to
    # create the deterministic ambiguous-bar setup, then seed state with
    # an already-open trade so run_paper_trading_step only has to manage
    # the exit, not also discover an entry.
    synthetic = make_synthetic_session(n_days=1)
    synthetic = synthetic.iloc[:60].copy()
    # Ambiguous bar: high reaches breakeven trigger (entry+5=105.5) AND
    # low breaches the original stop (95.5), for an open long at 100.5
    # with risk=5.
    synthetic.iloc[-1] = {"open": 100.5, "high": 113.0, "low": 95.0, "close": 105.0, "volume": 50000}
    # Bar before it (will be the "latest_closed" bar once one more bar
    # is appended) needs to exist too; append one more trailing bar so
    # the ambiguous bar becomes index -2 (the "most recently closed" bar
    # the paper trader acts on).
    trailing_ts = synthetic.index[-1] + pd.Timedelta(minutes=5)
    trailing_row = pd.DataFrame(
        {"open": [105.0], "high": [106.0], "low": [104.0], "close": [105.0], "volume": [50000]},
        index=[trailing_ts],
    )
    synthetic = pd.concat([synthetic, trailing_row])

    def fake_fetch_and_cache(symbol, interval, period, cache_dir):
        return synthetic

    monkeypatch.setattr(paper_trader, "fetch_and_cache", fake_fetch_and_cache)

    strat_cfg = StrategyConfig()
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    seeded_state = {
        "equity": cap_cfg.initial_capital,
        "daily_trade_count": {},
        "open_trade": {
            "direction": "long",
            "entry_time": str(synthetic.index[-3]),
            "entry_price": 100.5,
            "stop_price": 95.5,
            "original_stop_price": 95.5,
            "target_price": 113.0,
            "breakeven_active": False,
        },
        "last_processed_bar": str(synthetic.index[-3]),
    }
    with open(state_file, "w") as f:
        json.dump(seeded_state, f)

    paper_trader.run_paper_trading_step(strat_cfg, instr_cfg, cap_cfg, cache_dir=Path("unused"))

    assert trade_log.exists(), "Expected the ambiguous bar to trigger a stop-loss exit"
    logged = pd.read_csv(trade_log)
    assert len(logged) == 1
    assert logged.iloc[0]["exit_reason"] == "stop_loss"
    assert logged.iloc[0]["breakeven_active"] == False
    assert abs(logged.iloc[0]["exit_price"] - 95.5) < 1e-6
