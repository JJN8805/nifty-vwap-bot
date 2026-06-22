"""
Data fetching for NIFTY 50 index price data via yfinance.

IMPORTANT LIMITATION (read this before trusting backtest numbers):
yfinance has no reliable feed for NSE NIFTY *futures* contracts (the
rolling-expiry NFO instruments). What it does have is the NIFTY 50
*index* (^NSEI). This bot uses index price as a proxy for futures
price action, which is reasonable for *signal generation* (futures
track the index very closely intraday) but introduces two
approximations:

  1. Basis risk: futures trade at a small premium/discount to the index
     (the "basis"), which this bot ignores. Near expiry or in high
     volatility this can matter.
  2. Volume: ^NSEI volume data from yfinance is frequently zero/NaN,
     since an index itself isn't "traded". This bot's VWAP calculation
     falls back to typical price ((H+L+C)/3) on bars with no volume,
     which is NOT the same as a true volume-weighted average price.
     Treat VWAP-based signals here as directional confirmation, not as
     a precise volume-based indicator like the original Pine script
     intended.

If you have a Zerodha Kite Connect subscription (per your other bot),
swapping this module for real NFO futures candles with real volume
would remove both approximations. That's a natural upgrade path; the
rest of the codebase (signals, backtester, paper trader) doesn't care
where the candles come from as long as they arrive in the same shape.
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # With auto_adjust=False, yfinance returns BOTH "adj close" and
    # "close" as separate columns. We want raw "close" (already
    # present), not the dividend/split-adjusted variant -- and since
    # ^NSEI is an index with no dividends/splits to adjust for, the two
    # are identical anyway. Drop "adj close" outright rather than
    # renaming it: a rename onto an already-existing "close" column
    # silently produces two columns both named "close", which pandas
    # tolerates but pyarrow's to_parquet() correctly rejects with
    # "Duplicate column names found" -- which is the bug that broke the
    # very first live GitHub Actions run of this bot.
    df = df.drop(columns=["adj close"], errors="ignore")
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(IST)
    else:
        df.index = df.index.tz_convert(IST)

    df.index.name = "datetime"
    return df.sort_index()


def fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """
    Fetch intraday candles for `symbol` at `interval` granularity.

    yfinance limits intraday history depth by interval:
      - 1m  data: last ~7 days only
      - 5m/15m data: last ~60 days
    `period` should respect these limits (e.g. "60d" for 5m/15m).
    """
    raw = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    df = _normalize(raw)
    if df.empty:
        logger.warning("No data returned for %s at interval=%s period=%s", symbol, interval, period)
    return df


def fetch_and_cache(symbol: str, interval: str, period: str, cache_dir: Path) -> pd.DataFrame:
    """
    Fetch fresh data and append/merge into a local Parquet cache so that
    repeated GitHub Actions runs accumulate history beyond yfinance's
    lookback window, rather than losing it every run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol.replace('^', '')}_{interval}.parquet"

    fresh = fetch_intraday(symbol, interval, period)

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        existing.index = pd.to_datetime(existing.index)
        if existing.index.tz is None:
            existing.index = existing.index.tz_localize(IST)
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh

    if not combined.empty:
        combined.to_parquet(cache_path)

    return combined


def filter_market_hours(
    df: pd.DataFrame,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> pd.DataFrame:
    if df.empty:
        return df
    open_t = f"{open_hour:02d}:{open_minute:02d}"
    close_t = f"{close_hour:02d}:{close_minute:02d}"
    return df.between_time(open_t, close_t)
