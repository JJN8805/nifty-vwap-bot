#!/usr/bin/env python3
"""
Run a backtest of the VWAP V10 Low DD strategy over recent NIFTY 50
index data (used as a proxy for NIFTY futures -- see bot/data_feed.py
for why, and what that approximates).

Usage:
    python run_backtest.py --period 60d --interval 5m
    python run_backtest.py --period 30d --interval 5m --out reports/my_run.csv

Note: yfinance only retains ~60 days of 5m/15m intraday history and
~7 days of 1m history, so a backtest window is capped by that, not by
this script.
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from bot.backtester import run_backtest
from bot.config import USE_KITE, CapitalConfig, InstrumentConfig, StrategyConfig

if USE_KITE:
    from bot.data_feed_kite import fetch_and_cache, filter_market_hours
else:
    from bot.data_feed import fetch_and_cache, filter_market_hours

from bot.signals import compute_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Backtest the VWAP V10 Low DD strategy")
    parser.add_argument("--period", default="60d", help="yfinance lookback period, e.g. 60d")
    parser.add_argument("--interval", default="5m", help="LTF candle interval, e.g. 5m")
    parser.add_argument("--out", default="reports/backtest_trades.csv", help="Trade log output CSV")
    args = parser.parse_args()

    strat_cfg = StrategyConfig(ltf_interval=args.interval)
    instr_cfg = InstrumentConfig()
    cap_cfg = CapitalConfig()

    logger.info("Fetching %s data for %s (period=%s)...", args.interval, instr_cfg.symbol, args.period)
    raw = fetch_and_cache(instr_cfg.symbol, args.interval, args.period, cache_dir=Path("data/cache"))
    raw = filter_market_hours(
        raw,
        strat_cfg.market_open_hour,
        strat_cfg.market_open_minute,
        strat_cfg.market_close_hour,
        strat_cfg.market_close_minute,
    )

    if raw.empty:
        logger.error("No data fetched. Check network access / ticker symbol / yfinance status.")
        return

    logger.info("Fetched %d bars from %s to %s", len(raw), raw.index[0], raw.index[-1])

    df = compute_signals(raw, strat_cfg)
    result = run_backtest(df, strat_cfg, instr_cfg, cap_cfg)

    summary = result.summary()
    logger.info("===== BACKTEST SUMMARY =====")
    for k, v in summary.items():
        logger.info("%s: %s", k, v)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if result.trades:
        trades_df = pd.DataFrame([asdict(t) for t in result.trades])
        trades_df.to_csv(out_path, index=False)
        logger.info("Trade log written to %s", out_path)
    else:
        logger.warning("No trades were generated over this period -- conditions never aligned.")

    equity_path = Path("reports/backtest_equity_curve.csv")
    if not result.equity_curve.empty:
        result.equity_curve.to_csv(equity_path, header=["equity"])
        logger.info("Equity curve written to %s", equity_path)


if __name__ == "__main__":
    main()
