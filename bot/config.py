"""
Configuration for the VWAP V10 Low DD strategy.

Every parameter here maps 1:1 to an `input.*()` line in the original
Pine Script (VWAP V10 Low DD). Keeping the names aligned makes it easy
to cross-check behaviour against TradingView if you re-run the Pine
backtest side by side.

DATA SOURCE:
Set USE_KITE = True to use real NIFTY futures data via Zerodha Kite
Connect (requires KITE_API_KEY and KITE_ACCESS_TOKEN env vars / GitHub
secrets). Set USE_KITE = False (default) to use the free yfinance
NIFTY 50 index proxy instead.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------
# Flip this to True once your Kite API credentials are set up.
# See bot/data_feed_kite.py for the full setup instructions.
# ---------------------------------------------------------------
USE_KITE: bool = False


@dataclass(frozen=True)
class StrategyConfig:
    # --- EMA trend filter ---
    ema_fast: int = 20
    ema_slow: int = 50

    # --- Oscillators / volatility ---
    rsi_len: int = 14
    atr_len: int = 14

    # --- Trend strength thresholds (fractional separation, e.g. 0.0010 = 0.10%) ---
    ema_sep_threshold: float = 0.0010      # LTF EMA separation
    htf_sep_threshold: float = 0.0010      # HTF EMA separation

    # --- Risk / reward ---
    atr_stop_mult: float = 0.9
    reward_risk: float = 2.5

    # --- Trade frequency cap ---
    max_trades_per_day: int = 3

    # --- Timeframes ---
    ltf_interval: str = "5m"     # chart timeframe in the original script
    htf_interval: str = "15m"    # HTF trend filter timeframe

    # --- Session control (IST, 24h clock) ---
    force_exit_hour: int = 15
    force_exit_minute: int = 0
    market_open_hour: int = 9
    market_open_minute: int = 15
    market_close_hour: int = 15
    market_close_minute: int = 30

    # --- VWAP reclaim lookback used for VWAP slope check ---
    vwap_slope_lookback: int = 5


@dataclass(frozen=True)
class InstrumentConfig:
    """
    NIFTY futures contract economics, used to convert index point moves
    into simulated P&L for backtesting / paper trading.

    Lot size changes periodically per NSE/SEBI circulars -- NSE revised
    the NIFTY 50 lot size from 75 to 65, effective contracts expiring
    from January 2026 onward (NSE circular NSE/FAOP/70616, 03-Oct-2025).
    This value WILL change again over time. Verify the current lot size
    on the NSE website (nseindia.com) before relying on these P&L
    figures for any real capital decision -- don't trust this constant
    blindly, including the "65" set here.
    """

    symbol: str = "^NSEI"        # yfinance ticker for the NIFTY 50 index
    display_name: str = "NIFTY 50"
    lot_size: int = 65
    tick_size: float = 0.05

    # Flat estimate of round-trip cost (brokerage + STT + exchange + GST + stamp)
    # per lot, in INR. This is a simplification — replace with your actual
    # broker's cost sheet for more realistic backtest P&L.
    est_cost_per_lot_round_trip: float = 40.0


@dataclass(frozen=True)
class CapitalConfig:
    initial_capital: float = 100_000.0
    # Number of lots traded per signal. Kept simple/fixed; you can extend
    # this to volatility-based position sizing later.
    lots_per_trade: int = 1
