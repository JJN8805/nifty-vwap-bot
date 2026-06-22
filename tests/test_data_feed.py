"""
Tests for bot/data_feed.py's normalization logic.

This module previously had NO test coverage at all -- which is exactly
how a real bug (duplicate "close" columns after rename, breaking
to_parquet) shipped silently and only surfaced on the first live
GitHub Actions run, since my development sandbox had no network access
to yfinance to validate against a real response shape. These tests
reproduce that exact column structure so this class of bug can't slip
through again.
"""

import numpy as np
import pandas as pd

from bot.data_feed import _normalize


def _make_raw_yfinance_multiindex(symbol: str = "^NSEI", n: int = 5) -> pd.DataFrame:
    """
    Reproduces yfinance's actual single-ticker download shape with
    auto_adjust=False: a MultiIndex with BOTH "Adj Close" and "Close"
    as separate top-level fields, names=["Price", "Ticker"]. This is
    the exact structure that broke the live bot (see bot/data_feed.py
    comment on the "adj close" drop).
    """
    idx = pd.date_range("2026-06-01 09:15", periods=n, freq="5min", tz="UTC")
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    columns = pd.MultiIndex.from_product([fields, [symbol]], names=["Price", "Ticker"])
    data = np.random.default_rng(0).uniform(22000, 22100, size=(n, len(fields)))
    data[:, fields.index("Volume")] = 0  # ^NSEI volume is realistically always 0
    return pd.DataFrame(data, index=idx, columns=columns)


def test_normalize_drops_adj_close_without_creating_duplicate_close():
    raw = _make_raw_yfinance_multiindex()
    out = _normalize(raw)

    # The actual bug: a rename of "adj close" -> "close" when "close"
    # already exists silently produces two columns both named "close".
    # pandas tolerates this; pyarrow's to_parquet() does not.
    assert list(out.columns).count("close") == 1
    assert "adj close" not in out.columns
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}


def test_normalize_output_is_parquet_writable(tmp_path):
    """
    The actual failure mode observed on GitHub Actions: _normalize's
    output went straight into df.to_parquet() in fetch_and_cache, and
    pyarrow rejected it with "Duplicate column names found: ['open',
    'high', 'low', 'close', 'close', 'volume']". This test calls
    to_parquet for real (not just checking the column list) so a
    regression here fails loudly instead of just looking fine in
    isolation.
    """
    raw = _make_raw_yfinance_multiindex()
    out = _normalize(raw)

    out_path = tmp_path / "test.parquet"
    out.to_parquet(out_path)  # raises ValueError if duplicate columns slip back in

    reloaded = pd.read_parquet(out_path)
    assert list(reloaded.columns) == list(out.columns)


def test_normalize_handles_flat_columns_without_adj_close():
    """
    Some yfinance versions/call shapes return flat (non-MultiIndex)
    columns with no "Adj Close" at all. Confirm the drop is a no-op
    (errors="ignore") rather than crashing when there's nothing to drop.
    """
    idx = pd.date_range("2026-06-01 09:15", periods=3, freq="5min", tz="UTC")
    raw = pd.DataFrame(
        {
            "Open": [22000.0, 22010.0, 22020.0],
            "High": [22010.0, 22020.0, 22030.0],
            "Low": [21990.0, 22000.0, 22010.0],
            "Close": [22005.0, 22015.0, 22025.0],
            "Volume": [0, 0, 0],
        },
        index=idx,
    )
    out = _normalize(raw)
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    assert list(out.columns).count("close") == 1


def test_normalize_empty_input_returns_empty():
    empty = pd.DataFrame()
    out = _normalize(empty)
    assert out.empty
