import numpy as np
import pandas as pd

from bot.indicators import atr, ema, rsi, session_vwap
from tests.synthetic_data import make_synthetic_session


def test_ema_matches_pandas_ewm():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    result = ema(s, 3)
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_rsi_bounds():
    df = make_synthetic_session(n_days=2)
    r = rsi(df["close"], 14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 30, dtype=float))  # strictly increasing
    r = rsi(s, 14)
    assert r.iloc[-1] == 100


def test_atr_non_negative():
    df = make_synthetic_session(n_days=2)
    a = atr(df, 14)
    assert (a.dropna() >= 0).all()


def test_session_vwap_resets_each_day():
    df = make_synthetic_session(n_days=2)
    v = session_vwap(df)
    days = df.index.date
    unique_days = sorted(set(days))
    assert len(unique_days) == 2
    # VWAP on the first bar of each day should equal that bar's typical price
    for d in unique_days:
        day_mask = days == d
        first_idx = np.where(day_mask)[0][0]
        typical = (df.iloc[first_idx]["high"] + df.iloc[first_idx]["low"] + df.iloc[first_idx]["close"]) / 3
        assert abs(v.iloc[first_idx] - typical) < 1e-6
