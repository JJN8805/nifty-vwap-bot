import pandas as pd

from bot.config import StrategyConfig
from bot.signals import compute_signals
from tests.synthetic_data import make_synthetic_session


def test_compute_signals_runs_and_has_expected_columns():
    df = make_synthetic_session(n_days=5)
    cfg = StrategyConfig()
    out = compute_signals(df, cfg)

    expected_cols = {
        "ema_fast", "ema_slow", "bull_trend", "bear_trend", "trend_strong",
        "htf_ema_fast", "htf_ema_slow", "htf_bull", "htf_bear",
        "vwap", "vwap_bull", "vwap_bear",
        "rsi", "long_rsi", "short_rsi",
        "long_momentum", "short_momentum",
        "long_vwap_trigger", "short_vwap_trigger",
        "atr", "risk", "long_entry", "short_entry",
    }
    assert expected_cols.issubset(out.columns)
    assert len(out) == len(df)


def test_no_lookahead_in_htf_merge():
    """
    HTF EMA values attached to an LTF bar must come from an HTF candle
    that closed strictly before that LTF bar's own timestamp -- never
    from a still-forming or future HTF candle, and never from the very
    HTF candle that the bar itself would complete.
    """
    df = make_synthetic_session(n_days=3)
    cfg = StrategyConfig()
    out = compute_signals(df, cfg)

    # The very first bar of the whole series has no prior bar at all,
    # so it cannot have any HTF data yet.
    assert pd.isna(out["htf_ema_fast"].iloc[0])
    assert pd.isna(out["htf_ema_slow"].iloc[0])

    # Independently recompute the HTF EMA series the same way
    # signals.py does, then verify every LTF bar's attached HTF value
    # equals the HTF EMA as of the close of the *previous* LTF bar --
    # i.e. it lags by exactly one LTF bar relative to any HTF boundary,
    # which is the no-lookahead guarantee we need.
    from bot.signals import _resample_htf
    from bot.indicators import ema as ema_fn

    htf = _resample_htf(df, cfg.htf_interval)
    htf["htf_ema_fast"] = ema_fn(htf["close"], cfg.ema_fast)
    htf_closed_before = htf["htf_ema_fast"].dropna()

    for i in range(1, len(out)):
        prev_ts = out.index[i - 1]
        eligible = htf_closed_before[htf_closed_before.index <= prev_ts]
        expected = eligible.iloc[-1] if len(eligible) else float("nan")
        actual = out["htf_ema_fast"].iloc[i]
        if pd.isna(expected):
            assert pd.isna(actual)
        else:
            assert abs(actual - expected) < 1e-9


def test_entries_are_mutually_exclusive():
    df = make_synthetic_session(n_days=5)
    cfg = StrategyConfig()
    out = compute_signals(df, cfg)
    both = out["long_entry"] & out["short_entry"]
    assert not both.any()


def test_entries_require_all_conditions():
    """
    Sanity check the AND logic: wherever long_entry is True, every
    individual long sub-condition must also be True (and vice versa
    for short), matching the Pine script's combined boolean exactly.
    """
    df = make_synthetic_session(n_days=5)
    cfg = StrategyConfig()
    out = compute_signals(df, cfg)

    long_rows = out[out["long_entry"]]
    for col in ["htf_bull", "bull_trend", "trend_strong", "vwap_bull",
                "long_rsi", "long_momentum", "long_vwap_trigger"]:
        assert long_rows[col].all(), f"{col} should be True wherever long_entry is True"

    short_rows = out[out["short_entry"]]
    for col in ["htf_bear", "bear_trend", "trend_strong", "vwap_bear",
                "short_rsi", "short_momentum", "short_vwap_trigger"]:
        assert short_rows[col].all(), f"{col} should be True wherever short_entry is True"
