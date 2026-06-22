"""
Zerodha Kite Connect data feed for real NIFTY futures candles.

Replaces bot/data_feed.py when you want to paper trade with actual
futures data instead of the NIFTY 50 index proxy. Exposes the same
two public functions (fetch_and_cache, filter_market_hours) so no
other file in this codebase needs to change -- the swap is done
entirely in bot/config.py by setting USE_KITE = True.

WHAT THIS SOLVES COMPARED TO yfinance:
  1. Real NIFTY futures price (with the futures basis, not index price)
  2. Real volume data -> true volume-weighted VWAP instead of the
     typical-price approximation the yfinance path falls back to
  3. No 60-day history cap (Kite gives up to 60 days for minute data
     and 2000 days for day data, per Zerodha's API limits)

SETUP REQUIRED:
  1. Generate a Kite Connect API key at: https://developers.kite.trade
     (needs an active Zerodha trading account and a Kite Connect
     subscription -- Rs 2000/month as of writing, verify current price)
  2. Add two GitHub repository secrets (Settings -> Secrets and
     variables -> Actions -> New repository secret):
       KITE_API_KEY     : your API key from step 1
       KITE_ACCESS_TOKEN: a valid daily access token (see below)

ACCESS TOKEN REFRESH (the main operational overhead):
  Kite Connect access tokens expire every day at 06:00 IST. For
  fully-automated operation you have two options:
    a) Manual daily refresh: generate a new token each morning via
       the Kite login flow and update the GitHub secret by hand.
       Simple but requires daily attention on trading days.
    b) Automated refresh via a separate GitHub Actions workflow
       (not included here -- it requires storing your Kite password
       as a secret and running a headless login, which has security
       tradeoffs you should evaluate yourself before doing).
  For a 1-month paper trading trial, option (a) is fine -- it takes
  about 60 seconds each morning.

  To generate a fresh access token:
    pip install kiteconnect
    python -c "
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key='YOUR_API_KEY')
    print(kite.login_url())
    "
    Open the printed URL, log in, copy the 'request_token' from the
    redirect URL, then:
    python -c "
    from kiteconnect import KiteConnect
    kite = KiteConnect(api_key='YOUR_API_KEY')
    session = kite.generate_session('REQUEST_TOKEN', api_secret='YOUR_SECRET')
    print(session['access_token'])
    "
    Paste that access_token into your GitHub secret KITE_ACCESS_TOKEN.

FUTURES SYMBOL / CONTRACT ROLL:
  This module always fetches the front-month (nearest-expiry) NIFTY
  futures contract and rolls automatically on expiry day. The active
  contract instrument token is looked up from Kite's instrument list
  each time rather than hardcoded, so no manual intervention is needed
  on expiry days.
"""

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

IST = "Asia/Kolkata"

# Kite Connect interval strings that correspond to our LTF/HTF config
INTERVAL_MAP = {
    "1m":  "minute",
    "5m":  "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h":  "60minute",
    "1d":  "day",
}


def _get_kite_client():
    """
    Returns an authenticated KiteConnect client using credentials from
    environment variables (populated from GitHub secrets in Actions,
    or from your local shell for manual runs).

    Set these before running locally:
        export KITE_API_KEY=your_api_key
        export KITE_ACCESS_TOKEN=your_access_token
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError(
            "kiteconnect is not installed. Run: pip install kiteconnect\n"
            "Also add it to requirements_kite.txt if using GitHub Actions."
        )

    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        raise EnvironmentError(
            "KITE_API_KEY and KITE_ACCESS_TOKEN must be set as environment "
            "variables (or GitHub repository secrets for Actions runs).\n"
            "See the module docstring for setup instructions."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _get_nifty_futures_instrument(kite) -> dict:
    """
    Looks up the front-month NIFTY futures contract from Kite's live
    instrument list and returns its instrument_token and tradingsymbol.

    Always picks the nearest expiry (front month) unless today IS the
    expiry day, in which case it picks the next contract to avoid
    trading the expiring contract on its final day (liquidity drops
    sharply and the basis widens unpredictably).
    """
    instruments = kite.instruments("NFO")
    nifty_futures = [
        i for i in instruments
        if i["name"] == "NIFTY"
        and i["instrument_type"] == "FUT"
        and i["segment"] == "NFO-FUT"
    ]

    if not nifty_futures:
        raise ValueError("No NIFTY futures contracts found in Kite instrument list.")

    today = date.today()
    nifty_futures.sort(key=lambda i: i["expiry"])

    # Skip the front month if today is its expiry day
    for contract in nifty_futures:
        if contract["expiry"] > today:
            logger.info(
                "Active contract: %s (expiry: %s)",
                contract["tradingsymbol"],
                contract["expiry"],
            )
            return contract

    raise ValueError("Could not find a non-expired NIFTY futures contract.")


def _kite_to_dataframe(candles: list, interval: str) -> pd.DataFrame:
    """
    Convert Kite's historical_data response (a list of dicts with keys
    date/open/high/low/close/volume) into a standardized OHLCV DataFrame
    with a tz-aware IST DatetimeIndex, matching the shape that the rest
    of this codebase expects.
    """
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(candles)
    df = df.rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize(IST)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(IST)

    df = df.set_index("datetime")
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "datetime"
    return df.sort_index()


def fetch_and_cache(
    symbol: str,   # ignored -- we always fetch the front-month NIFTY future
    interval: str, # e.g. "5m" from StrategyConfig.ltf_interval
    period: str,   # e.g. "60d" -- converted to a from_date / to_date range
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Fetch real NIFTY futures candles from Kite Connect and cache them
    locally as Parquet, exactly like the yfinance version does.

    The `symbol` parameter is accepted for interface compatibility but
    ignored -- this function always resolves and uses the current
    front-month NIFTY futures contract.

    `period` must be a string like "60d", "30d", "7d".
    Kite limits: up to 60 days of minute-level data per request.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"NIFTY_FUT_{interval}.parquet"

    kite = _get_kite_client()
    contract = _get_nifty_futures_instrument(kite)

    kite_interval = INTERVAL_MAP.get(interval)
    if kite_interval is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {list(INTERVAL_MAP.keys())}"
        )

    # Convert "60d" period string to actual from/to dates
    days = int(period.rstrip("d"))
    to_date = date.today()
    from_date = to_date - timedelta(days=days)

    # Kite's API caps minute-level requests at 60 days per call -- if
    # the requested period exceeds this, we pull in 60-day chunks and
    # concatenate. In practice "60d" fits in one request.
    all_candles = []
    chunk_start = from_date
    chunk_size = 60  # days per Kite API request for minute data

    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_size), to_date)
        try:
            candles = kite.historical_data(
                instrument_token=contract["instrument_token"],
                from_date=chunk_start,
                to_date=chunk_end,
                interval=kite_interval,
                continuous=False,
                oi=False,
            )
            all_candles.extend(candles)
        except Exception as e:
            logger.warning(
                "Kite historical_data failed for chunk %s to %s: %s",
                chunk_start, chunk_end, e,
            )
        chunk_start = chunk_end + timedelta(days=1)

    fresh = _kite_to_dataframe(all_candles, interval)

    if fresh.empty:
        logger.warning(
            "No data returned from Kite for %s at interval=%s period=%s",
            contract["tradingsymbol"], interval, period,
        )

    # Merge with existing cache, exactly like the yfinance version
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
        logger.info(
            "Cached %d bars for %s (%s to %s)",
            len(combined),
            contract["tradingsymbol"],
            combined.index[0],
            combined.index[-1],
        )

    return combined


def filter_market_hours(
    df: pd.DataFrame,
    open_hour: int,
    open_minute: int,
    close_hour: int,
    close_minute: int,
) -> pd.DataFrame:
    """Identical to the yfinance version -- kept here for interface parity."""
    if df.empty:
        return df
    open_t = f"{open_hour:02d}:{open_minute:02d}"
    close_t = f"{close_hour:02d}:{close_minute:02d}"
    return df.between_time(open_t, close_t)
