"""Configuration constants for Supply & Demand zone strategy."""

from __future__ import annotations
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Symbol / Data ─────────────────────────────────────────────────────────────
SYMBOL = "BTC/USD:USD"
TIMEFRAME = "30m"
ATR_PERIOD = 14

# ── Impulse ───────────────────────────────────────────────────────────────────
IMPULSE_MULT = 1.5
MIN_BODY_RATIO = 0.55

# ── Zone width limits ─────────────────────────────────────────────────────────
MAX_ZONE_WIDTH_PCT = 0.012

# ── Base expansion ────────────────────────────────────────────────────────────
BASE_MAX_CANDLES = 6
BASE_BODY_THRESH = 0.6

# ── Freshness ─────────────────────────────────────────────────────────────────
WICK_TOLERANCE_PCT = 0.002

# ── Volume ────────────────────────────────────────────────────────────────────
VOL_ZSCORE_THRESH = 1.5

# ── Liquidity ─────────────────────────────────────────────────────────────────
EQL_TOLERANCE_PCT = 0.001
EQL_MIN_TOUCHES = 2

# ── Backtest ──────────────────────────────────────────────────────────────────
BT_SL_ATR_MULT = 1.0
BT_TP_RR = 2.0
BT_MIN_RR = 1.5
BT_MAX_OPEN_TRADES = 3
BT_ZONE_CLUSTER_PCT = 0.008
BT_EXCLUDE_MONTHS = []

# ── Sessions (UTC hours) ──────────────────────────────────────────────────────
SESSION_ASIA = (0, 8)
SESSION_LONDON = (8, 13)
SESSION_NY = (13, 21)
# ── Risk / equity simulation ──────────────────────────────────────────────────
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE_PCT = 0.01
RISK_CAP_PCT = 0.02  # max risk per trade as % of initial balance

# ── Fees ──────────────────────────────────────────────────────────────────────
FEE_RATE_MAKER = 0.0002  # Limit post-only (entry + TP)
FEE_RATE_TAKER = 0.0005  # Stop market (SL)
SL_SLIPPAGE_PCT = 0.0005  # 0.05% slippage on stop market orders only

# ── Optimal params (from WFO on BTC 2yr) ─────────────────────────────────────
OPTIMAL_PARAMS = {
    "BT_SL_ATR_MULT": 1.0,
    "BT_TP_RR": 2.5,
    "BT_MAX_OPEN_TRADES": 4,
    "BT_ZONE_CLUSTER_PCT": BT_ZONE_CLUSTER_PCT,
    "BT_MIN_RR": BT_MIN_RR,
}

