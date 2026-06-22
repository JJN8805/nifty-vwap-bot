#!/usr/bin/env python3
"""
Entry point invoked by the GitHub Actions cron schedule. Each run
performs ONE evaluation step: fetch latest data, check the most
recently closed candle against the strategy rules, update the open
paper position (or open a new one), and persist state.

This is intentionally idempotent-ish: if no new bar has closed since
the last run, it's a no-op (see `last_processed_bar` check in
bot/paper_trader.py).
"""

import logging

from bot.config import CapitalConfig, InstrumentConfig, StrategyConfig
from bot.paper_trader import run_paper_trading_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    run_paper_trading_step(
        strat_cfg=StrategyConfig(),
        instr_cfg=InstrumentConfig(),
        cap_cfg=CapitalConfig(),
    )
