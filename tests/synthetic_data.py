"""
Generates synthetic but realistic-shaped 5-minute NIFTY-like OHLCV data
for testing, since we don't have live network access to yfinance in
this sandbox. Includes a deliberate trending stretch so we can verify
that entries actually fire under the right conditions.

IMPORTANT CAVEAT: this generator produces non-zero, randomly varying
volume, which exercises the TRUE volume-weighted VWAP path in
bot/indicators.session_vwap(). Real ^NSEI data from yfinance has
zero/NaN volume almost always, which instead exercises the typical-
price FALLBACK path (see bot/indicators.py and bot/data_feed.py for
why). That fallback path is covered separately and directly by
test_indicators.py::test_session_vwap_resets_each_day, but every
signals/backtester test using this generator is implicitly testing
against "nicer" data than what production paper trading will actually
see. Don't assume backtest behaviour observed here against synthetic
data will look identical once real (effectively volume-less) NIFTY
data is used -- the VWAP-dependent entry conditions specifically may
behave differently.
"""

import numpy as np
import pandas as pd


def make_synthetic_session(
    n_days: int = 5,
    interval_minutes: int = 5,
    start_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price

    for day in range(n_days):
        date = pd.Timestamp("2026-01-05", tz="Asia/Kolkata") + pd.Timedelta(days=day)
        # Skip to a weekday-like cadence (not exact NSE calendar, fine for synthetic test data)
        session_start = date.replace(hour=9, minute=15)
        n_bars = int((6.25 * 60) / interval_minutes)  # ~9:15 to 15:30

        # Inject a trend bias for some days so entries actually trigger
        trend_bias = 0.6 if day % 2 == 0 else -0.6

        for i in range(n_bars):
            ts = session_start + pd.Timedelta(minutes=i * interval_minutes)
            drift = trend_bias * 0.05
            shock = rng.normal(0, 8)
            change = drift + shock
            open_p = price
            close_p = max(price + change, 1)
            high_p = max(open_p, close_p) + abs(rng.normal(0, 4))
            low_p = min(open_p, close_p) - abs(rng.normal(0, 4))
            volume = int(abs(rng.normal(50000, 10000)))

            rows.append(
                {
                    "datetime": ts,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume,
                }
            )
            price = close_p

    df = pd.DataFrame(rows).set_index("datetime")
    return df
