"""
Indicator calculations matching Pine Script's built-in ta.* functions
closely enough for signal parity:

- ta.ema   -> pandas .ewm(span=length, adjust=False)
- ta.rsi   -> Wilder's smoothed RSI (matches Pine's ta.rsi)
- ta.atr   -> Wilder's smoothed ATR (matches Pine's ta.atr)
- ta.vwap  -> session-anchored VWAP, resets every trading day like Pine's
              intraday VWAP does on a fresh session
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (alpha = 1/length), matches Pine's ta.rsi
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.fillna(100)  # if avg_loss is 0, RSI -> 100
    return out


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing, matches Pine's ta.atr
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Anchored VWAP that resets at the start of each trading day, mirroring
    Pine's ta.vwap() behaviour on an intraday chart (new session = new
    anchor). `df.index` must be a tz-aware or naive DatetimeIndex.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"].replace(0, np.nan)

    day_key = df.index.date
    grouped_pv = pd.Series(pv.values, index=df.index).groupby(day_key).cumsum()
    grouped_vol = (
        df["volume"].replace(0, np.nan).groupby(day_key).cumsum()
    )

    vwap = grouped_pv / grouped_vol
    # Fallback to typical price if volume is unavailable (e.g. index data
    # from yfinance often has zero/NaN volume) so the strategy still runs,
    # at the cost of VWAP degrading to a simple price proxy on those bars.
    return vwap.fillna(typical_price)
