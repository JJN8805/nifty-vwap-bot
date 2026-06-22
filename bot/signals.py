"""
Signal generation logic — a direct, bar-by-bar port of the Pine Script
strategy "VWAP V10 Low DD".

Design note on lookahead bias:
The Pine script computes its HTF (15m) EMAs via request.security() on
the *same bar* as the LTF signal. On a live/closed-bar chart this is
safe (no future leakage) as long as the HTF candle has actually closed.
To replicate that safely here, we:
  1. Resample LTF data into HTF candles.
  2. Compute HTF EMAs on the HTF series.
  3. Forward-fill HTF EMA values onto LTF bars using only HTF candles
     that have *already closed* relative to each LTF timestamp
     (merge_asof with direction='backward').
This avoids accidentally peeking at a still-forming HTF candle.
"""

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .indicators import atr, ema, rsi, session_vwap


def _to_pandas_freq(interval: str) -> str:
    """
    Normalize a yfinance-style interval string (e.g. "5m", "15m", "1h")
    into a pandas-compatible offset alias. Pandas 2.2+ deprecated the
    bare "m" alias for minutes (it now means "month-end"); "min" is the
    unambiguous, version-safe alias. This keeps cfg.htf_interval /
    cfg.ltf_interval expressible in the same shorthand yfinance uses,
    while still working with whatever pandas version is installed.
    """
    if interval.endswith("m") and not interval.endswith("min"):
        return f"{interval[:-1]}min"
    return interval


def _resample_htf(ltf_df: pd.DataFrame, htf_interval: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    freq = _to_pandas_freq(htf_interval)
    htf = ltf_df.resample(freq, label="right", closed="right").agg(agg)
    htf = htf.dropna(subset=["close"])
    return htf


def compute_signals(ltf_df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    ltf_df: DataFrame indexed by tz-aware datetime, columns
            [open, high, low, close, volume], at cfg.ltf_interval granularity,
            already filtered to a single trading day or a contiguous range
            of trading days/sessions.

    Returns ltf_df with extra columns including `long_entry`, `short_entry`,
    and helper indicator columns, matching the Pine script's logic exactly.
    """
    df = ltf_df.copy()

    # ---------- LTF trend ----------
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)
    df["bull_trend"] = df["ema_fast"] > df["ema_slow"]
    df["bear_trend"] = df["ema_fast"] < df["ema_slow"]
    df["ema_sep"] = (df["ema_fast"] - df["ema_slow"]).abs() / df["ema_slow"]
    df["trend_strong"] = df["ema_sep"] > cfg.ema_sep_threshold

    # ---------- HTF trend (no-lookahead) ----------
    htf = _resample_htf(df, cfg.htf_interval)
    htf["htf_ema_fast"] = ema(htf["close"], cfg.ema_fast)
    htf["htf_ema_slow"] = ema(htf["close"], cfg.ema_slow)
    htf_lookup = htf[["htf_ema_fast", "htf_ema_slow"]].dropna()

    # A 15m candle that closes at, say, 09:30 is labelled 09:30 -- the
    # same timestamp as the LTF bar that completes it. Looking that HTF
    # candle up "as of" the LTF bar's own timestamp would let the bar see
    # data from itself (1-bar lookahead leak). Instead we look up HTF
    # state as of the END of the *previous* LTF bar, guaranteeing only
    # HTF candles that closed strictly before the current bar are used.
    htf_times = htf_lookup.index.values
    prev_ltf_times = df.index.values.astype(htf_times.dtype)
    # shift right by one bar: bar i looks up using bar (i-1)'s timestamp;
    # bar 0 has no previous bar at all, so it must get no HTF data.
    has_prev = np.ones(len(df), dtype=bool)
    has_prev[0] = False
    prev_ltf_times = np.r_[prev_ltf_times[:1], prev_ltf_times[:-1]]  # placeholder for bar 0, masked out below

    positions = np.searchsorted(htf_times, prev_ltf_times, side="right") - 1
    valid = has_prev & (positions >= 0)

    htf_ema_fast_aligned = np.full(len(df), np.nan)
    htf_ema_slow_aligned = np.full(len(df), np.nan)
    htf_ema_fast_aligned[valid] = htf_lookup["htf_ema_fast"].values[positions[valid]]
    htf_ema_slow_aligned[valid] = htf_lookup["htf_ema_slow"].values[positions[valid]]

    df["htf_ema_fast"] = htf_ema_fast_aligned
    df["htf_ema_slow"] = htf_ema_slow_aligned

    df["htf_sep"] = (df["htf_ema_fast"] - df["htf_ema_slow"]).abs() / df["htf_ema_slow"]
    df["htf_bull"] = (df["htf_ema_fast"] > df["htf_ema_slow"]) & (df["htf_sep"] > cfg.htf_sep_threshold)
    df["htf_bear"] = (df["htf_ema_fast"] < df["htf_ema_slow"]) & (df["htf_sep"] > cfg.htf_sep_threshold)

    # ---------- VWAP ----------
    df["vwap"] = session_vwap(df)
    df["vwap_bull"] = df["vwap"] > df["vwap"].shift(cfg.vwap_slope_lookback)
    df["vwap_bear"] = df["vwap"] < df["vwap"].shift(cfg.vwap_slope_lookback)

    # ---------- RSI ----------
    df["rsi"] = rsi(df["close"], cfg.rsi_len)
    df["long_rsi"] = df["rsi"] > 50
    df["short_rsi"] = df["rsi"] < 50

    # ---------- Momentum ----------
    df["long_momentum"] = df["close"] > df["close"].shift(1)
    df["short_momentum"] = df["close"] < df["close"].shift(1)

    # ---------- VWAP reclaim trigger ----------
    df["long_vwap_trigger"] = (
        (df["low"] <= df["vwap"]) & (df["close"] > df["vwap"]) & (df["close"] > df["open"])
    )
    df["short_vwap_trigger"] = (
        (df["high"] >= df["vwap"]) & (df["close"] < df["vwap"]) & (df["close"] < df["open"])
    )

    # ---------- ATR risk ----------
    df["atr"] = atr(df, cfg.atr_len)
    df["risk"] = df["atr"] * cfg.atr_stop_mult

    # ---------- Combined entries (all conditions, matching Pine) ----------
    df["long_entry"] = (
        df["htf_bull"]
        & df["bull_trend"]
        & df["trend_strong"]
        & df["vwap_bull"]
        & df["long_rsi"]
        & df["long_momentum"]
        & df["long_vwap_trigger"]
    )

    df["short_entry"] = (
        df["htf_bear"]
        & df["bear_trend"]
        & df["trend_strong"]
        & df["vwap_bear"]
        & df["short_rsi"]
        & df["short_momentum"]
        & df["short_vwap_trigger"]
    )

    return df
