# NIFTY VWAP V10 Low DD — Algo Trading Bot

A Python port of the "VWAP V10 Low DD" Pine Script strategy (originally
built for TradingView), set up for **backtesting** and **automated
paper trading** on NIFTY 50, running entirely on GitHub Actions with
no broker account or paid data subscription required.

## What this strategy does

Multi-timeframe trend filter (15m + chart timeframe EMA 20/50 cross,
with a minimum separation threshold to avoid choppy markets) combined
with a VWAP reclaim trigger as the actual entry signal, confirmed by
RSI and short-term momentum. Risk is controlled with an ATR-based
stop/target (default 1:2.5 reward:risk) and a breakeven stop once
price has moved one risk-unit in favour. All positions are force-closed
at 15:00 IST, and at most 3 new trades are taken per day.

See `bot/signals.py` and `bot/backtester.py` for the exact logic —
every condition is named to match the original Pine script variables
so you can cross-check against TradingView.

## Important: read this before trusting any numbers

This bot uses **yfinance** as a free data source, which has two real
limitations you should know about:

1. **No NIFTY futures data.** yfinance doesn't carry NSE F&O
   (futures & options) contracts. This bot uses the NIFTY 50 *index*
   (`^NSEI`) as a proxy for futures price action. Index and futures
   prices track each other closely intraday, but there's a small
   "basis" difference this bot ignores — backtest/paper P&L will be
   close to, but not identical to, real futures P&L.
2. **No real volume data.** `^NSEI` volume from yfinance is usually
   zero or missing (it's an index, not a traded security), so a true
   volume-weighted VWAP can't be computed. This bot falls back to
   typical price `(high+low+close)/3` on bars with no volume, which
   behaves like a price-based VWAP approximation rather than the
   original volume-weighted indicator. Directionally it should behave
   similarly most of the time, but don't treat it as identical to
   what you'd see in a real volume-aware backtest.

If you later want to remove both approximations, swap
`bot/data_feed.py` for a feed from your Zerodha Kite Connect API
(which you already have set up for another project) — the rest of the
codebase (signals, backtester, paper trader) is written against a
plain OHLCV DataFrame and doesn't care where the candles came from.

`yfinance` also limits how much intraday history it will hand back:
roughly the last 60 days for 5-minute/15-minute candles, and only the
last ~7 days for 1-minute candles. The bot caches fetched data into
`data/cache/*.parquet` and merges new fetches in, so a month of paper
trading will build up a genuine month of history over time even though
any single fetch only returns a 60-day rolling window.

## Project structure

```
bot/
  config.py        Strategy / instrument / capital parameters
  indicators.py     EMA, RSI, ATR, session VWAP
  signals.py        Entry/exit condition logic (the strategy itself)
  data_feed.py      yfinance fetch + local parquet cache
  backtester.py     Bar-by-bar backtest engine
  paper_trader.py   Single-step paper trading logic + state persistence
run_backtest.py             CLI: run a historical backtest
run_paper_trade_step.py     CLI: one paper-trading evaluation step
tests/                       Unit tests using synthetic OHLCV data
.github/workflows/
  paper_trading.yml          Runs every 5 min during market hours
  weekly_backtest.yml        Re-runs a fresh backtest weekly
data/                         Cached price data + paper trading state (git-tracked)
reports/                      Trade logs, equity curves (git-tracked)
```

## Setup

```bash
git clone <your-repo-url>
cd nifty-vwap-bot
pip install -r requirements.txt
pip install pytest   # only needed to run tests yourself
```

Run the test suite (uses synthetic data, no network needed):

```bash
python -m pytest tests/ -v
```

## Running a backtest locally

```bash
python run_backtest.py --period 60d --interval 5m
```

This fetches the last 60 days of 5-minute NIFTY 50 data, runs the
strategy bar-by-bar, and writes:
- `reports/backtest_trades.csv` — every simulated trade
- `reports/backtest_equity_curve.csv` — equity over time

Console output includes a summary: win rate, profit factor, max
drawdown, etc.

## Running paper trading for a month

### Option A: Fully automated on GitHub Actions (recommended)

1. Push this repo to GitHub.
2. In the repo's **Settings → Actions → General**, under "Workflow
   permissions," select **"Read and write permissions"** — the
   scheduled workflow commits updated state/logs back to the repo, so
   it needs write access.
3. That's it. `.github/workflows/paper_trading.yml` is already
   scheduled to run every 5 minutes during NSE market hours
   (09:15–15:30 IST, Monday–Friday) via cron, and
   `weekly_backtest.yml` re-validates against fresh data every
   Saturday.
4. Each run is a single evaluation step: check the latest closed
   candle, manage any open paper position (stop/target/breakeven/
   force-exit), open a new one if conditions align, and commit the
   updated `data/paper_state.json`, `reports/paper_trade_log.csv`, and
   `reports/paper_equity_log.csv` back to the repo.
5. After a month, `reports/paper_trade_log.csv` is your full paper
   trading record — every entry, exit, reason, and P&L in INR (lot
   size accounted for, with an estimated round-trip cost deducted per
   trade).

You can also trigger a run manually any time from the **Actions** tab
using the "Run workflow" button (both workflows have
`workflow_dispatch` enabled).

### Option B: Run it yourself on a schedule

If you'd rather not rely on GitHub Actions' cron (which is best-effort
and can lag anywhere from a few minutes to 20-30+ minutes during
periods of platform load -- GitHub does not guarantee on-time
execution for scheduled workflows -- and which auto-disables on repos
with no activity for 60 days):

```bash
# crontab -e, then add a line like:
*/5 9-15 * * 1-5 cd /path/to/nifty-vwap-bot && python run_paper_trade_step.py
```

State is stored in `data/paper_state.json` either way — there's no
difference in behaviour between the two options other than who's
triggering the script.

## Interpreting paper trading results after a month

- `reports/paper_trade_log.csv`: one row per closed trade —
  direction, entry/exit time and price, exit reason
  (`target` / `stop_loss` / `breakeven_stop` / `force_exit_1500`), and
  P&L in INR.
- `reports/paper_equity_log.csv`: equity value logged at every
  evaluation step, useful for plotting an equity curve.
- Compare this against `reports/backtest_trades.csv` from the same
  date range — if paper trading and backtest results diverge a lot, it
  usually means either a network/data gap during live running, or a
  market holiday the bot didn't account for (NSE holidays aren't
  hardcoded in; the bot just won't get new candles on those days,
  which is harmless but worth knowing).

## Tuning parameters

Everything in the original Pine Script's `input.*()` lines is exposed
in `bot/config.py`'s `StrategyConfig` — EMA lengths, RSI/ATR length,
separation thresholds, ATR stop multiplier, reward:risk ratio, and max
trades per day. Lot size and per-trade cost assumptions are in
`InstrumentConfig`. Edit these directly, or extend `run_backtest.py`
to accept them as CLI flags if you want to sweep parameters.

## Known limitations / honest caveats

- **This Python port is intentionally stricter about lookahead bias
  than the original Pine script.** The original script reads the 15m
  HTF trend via `request.security()` without `barmerge.lookahead_off`
  or an explicit `[1]` offset -- a well-known TradingView gotcha that
  can let a 5m bar see a 15m candle's value slightly before that
  candle has technically closed, especially on the live/right-most
  bar. This Python port deliberately closes that gap: an LTF bar can
  only ever see an HTF candle that closed strictly before it. This
  means **this bot's backtest results will not exactly match a
  TradingView backtest of the same Pine script** -- the Python version
  is the more conservative (and arguably more correct) of the two, but
  if you're cross-checking trade counts or entry timestamps against
  TradingView, expect small differences around 15-minute boundaries,
  not a bug.
- Index-as-futures-proxy and degraded VWAP, as covered above.
- No NSE holiday calendar — the bot just sees no new data on holidays,
  which is harmless (it skips that day) but isn't validated against
  an explicit calendar.
- No slippage modeling beyond the flat estimated round-trip cost
  (`InstrumentConfig.est_cost_per_lot_round_trip`) — real fills,
  especially on the stop side during fast moves, can be worse than
  assumed here.
- **Same-bar stop/breakeven ambiguity is resolved conservatively.** If
  a single 5-minute candle's high/low range touches both the original
  stop level AND the breakeven-trigger level (1R favourable), there's
  no way to know from OHLC data alone which happened first within that
  bar. This bot always assumes the original stop was hit first (the
  worse-case outcome) rather than optimistically granting breakeven --
  see `evaluate_exit()` in `bot/backtester.py`. This makes backtest
  results slightly more pessimistic than a tick-level simulation might
  show, which is the safer direction to be wrong in.
- This is a **paper trading and backtesting tool only**. It places no
  real orders and connects to no brokerage. Treat all P&L figures as
  simulation output, not as a guarantee of how a live-funded version
  would perform.
