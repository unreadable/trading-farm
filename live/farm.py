"""Supply/Demand Trading Farm — live trading on Kraken Futures.

Usage:
    export KRAKEN_FUTURES_API_KEY=your_key
    export KRAKEN_FUTURES_API_SECRET=your_secret
    python -m live.farm [--coins AVAX,ETH] [--max-trades 4] [--risk-pct 0.5]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    OPTIMAL_PARAMS, ATR_PERIOD, WICK_TOLERANCE_PCT,
    FEE_RATE_MAKER, FEE_RATE_TAKER, SL_SLIPPAGE_PCT,
)
from zones import detect_zones
from structure import compute_atr
from backtest import find_dynamic_tp
from data import fetch_ohlcv_full
from live.kraken_client import KrakenFuturesClient

# ── Config ────────────────────────────────────────────────────────────────────

TIMEFRAME = "30m"
CANDLE_SECONDS = 1800  # 30 minutes
POLL_INTERVAL = 30     # seconds between order checks
LOOKBACK_DAYS = 60     # days of history for zone detection (tested: best PF + avg RR)
LEVERAGE = 5           # isolated margin, 5x — enough for our position sizes

# Instrument specs cache (populated at startup)
_instrument_specs = {}


def fmt_price(price: float) -> str:
    """Format price to match Kraken Pro display conventions."""
    p = abs(price)
    if p >= 10_000:
        return f"{price:,.0f}"
    elif p >= 100:
        return f"{price:,.1f}"
    elif p >= 10:
        return f"{price:.2f}"
    else:
        return f"{price:.3f}"

COIN_MAP = {
    "AVAX": "PF_AVAXUSD",
    "UNI":  "PF_UNIUSD",
    "LINK": "PF_LINKUSD",
    "LTC":  "PF_LTCUSD",
    "BNB":  "PF_BNBUSD",
    "SOL":  "PF_SOLUSD",
    "ETH":  "PF_ETHUSD",
    "BTC":  "PF_XBTUSD",
}

# ccxt symbol format for data fetching
CCXT_MAP = {
    "AVAX": "AVAX/USD:USD",
    "UNI":  "UNI/USD:USD",
    "LINK": "LINK/USD:USD",
    "LTC":  "LTC/USD:USD",
    "BNB":  "BNB/USD:USD",
    "SOL":  "SOL/USD:USD",
    "ETH":  "ETH/USD:USD",
    "BTC":  "BTC/USD:USD",
}

STATE_FILE = Path(__file__).parent / "state.json"
LOG_FILE = Path(__file__).parent / "farm.log"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("farm")

# ── State management ─────────────────────────────────────────────────────────


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"trades": [], "completed": [], "total_r": 0.0}


def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.replace(STATE_FILE)  # atomic on POSIX


# ── Zone detection ────────────────────────────────────────────────────────────


def compute_signals(df, coin: str, active_trades: list[dict],
                    params: dict) -> list[dict]:
    """Core signal logic — detect zones, check freshness, find entries.

    Separated from get_signals() for testability (no API dependency).
    Returns list of signal dicts with entry/sl/tp/size info.
    """
    kraken_symbol = COIN_MAP.get(coin, coin)

    zones, struct = detect_zones(df, TIMEFRAME, no_lookahead=True)
    atr_series = compute_atr(df, ATR_PERIOD)
    liq_levels = struct.get("liquidity_levels", [])

    _sl_atr = params.get("BT_SL_ATR_MULT", 1.0)
    _cluster_pct = params.get("BT_ZONE_CLUSTER_PCT", 0.008)

    # Current price and ATR
    current_price = float(df["close"].iloc[-1])
    current_atr = float(atr_series.iloc[-1])
    if current_atr == 0:
        return []

    # Incremental zone activation + freshness walk (matching backtest exactly)
    # Zones are activated on the bar their impulse occurs, then freshness
    # is checked. This interleaving matters for correct clustering.
    zones_sorted = sorted(zones, key=lambda z: z["impulse_idx"])
    active_zones = []
    zone_ptr = 0
    warmup = max(ATR_PERIOD + 5, 20)
    wick_tol = WICK_TOLERANCE_PCT

    for i in range(warmup, len(df)):
        # 1. Activate new zones arriving on this bar
        while zone_ptr < len(zones_sorted) and zones_sorted[zone_ptr]["impulse_idx"] <= i:
            z = dict(zones_sorted[zone_ptr])
            z["_fresh"] = True
            z["_trade_opened"] = False

            clustered = False
            for existing in active_zones:
                if (existing["type"] == z["type"] and existing["_fresh"]
                        and abs(existing["mid"] - z["mid"]) / z["mid"] < _cluster_pct):
                    if z["strength"] > existing["strength"]:
                        existing["_fresh"] = False
                    else:
                        clustered = True
                    break
            if not clustered:
                active_zones.append(z)
            zone_ptr += 1

        # 2. Update freshness on this bar
        bar_close = float(df["close"].iloc[i])
        for z in active_zones:
            if not z["_fresh"]:
                continue
            tolerance = z["mid"] * wick_tol
            if z["type"] == "demand":
                if bar_close < z["bottom"] - tolerance:
                    z["_fresh"] = False
            else:
                if bar_close > z["top"] + tolerance:
                    z["_fresh"] = False

    fresh_count = sum(1 for z in active_zones if z["_fresh"])

    # Find zones touched by the last candle (matching backtest entry logic)
    signals = []
    last_bar_low = float(df["low"].iloc[-1])
    last_bar_high = float(df["high"].iloc[-1])

    # Collect entry prices of active trades on this coin to prevent duplicates
    active_entries = {t["entry_price"] for t in active_trades
                      if t.get("coin") == coin
                      and t.get("status") in ("pending_entry", "active")}

    for z in active_zones:
        if not z["_fresh"]:
            continue

        entry_price = z["top"] if z["type"] == "demand" else z["bottom"]

        # Skip if we already have a trade at this zone's entry price
        if any(abs(entry_price - ep) / entry_price < _cluster_pct
               for ep in active_entries):
            continue

        # Intra-bar touch check — same as backtest:
        # demand: bar_low <= zone_top (price dipped into demand zone)
        # supply: bar_high >= zone_bottom (price rose into supply zone)
        zone_reached = False
        if z["type"] == "demand" and last_bar_low <= z["top"]:
            zone_reached = True
        elif z["type"] == "supply" and last_bar_high >= z["bottom"]:
            zone_reached = True
        if not zone_reached:
            continue

        # Calculate SL
        if z["type"] == "demand":
            sl = z["bottom"] - current_atr * _sl_atr
            risk = entry_price - sl
            side = "buy"
        else:
            sl = z["top"] + current_atr * _sl_atr
            risk = sl - entry_price
            side = "sell"

        if risk <= 0:
            continue

        # Fee filter: skip if estimated round-trip fees exceed 0.3R
        notional = (1.0 / risk) * entry_price  # per $1 of risk
        est_fee_r = notional * (FEE_RATE_MAKER + FEE_RATE_TAKER + SL_SLIPPAGE_PCT)
        if est_fee_r > 0.3:
            continue

        # Calculate TP
        tp = find_dynamic_tp(z, entry_price, risk, active_zones, liq_levels,
                             params=params)
        rr_target = abs(tp - entry_price) / risk

        signals.append({
            "coin": coin,
            "symbol": kraken_symbol,
            "zone_type": z["type"],
            "side": side,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "rr_target": round(rr_target, 2),
            "strength": z.get("strength", 0),
            "atr": round(current_atr, 2),
            "current_price": current_price,
        })

    # Sort by strength (best zones first)
    signals.sort(key=lambda s: -s["strength"])
    return signals, fresh_count, active_zones, current_price


def get_signals(coin: str, active_trades: list[dict],
                params: dict) -> list[dict]:
    """Fetch data + compute signals. Thin wrapper around compute_signals()."""
    ccxt_symbol = CCXT_MAP[coin]
    df = fetch_ohlcv_full(ccxt_symbol, TIMEFRAME, days=LOOKBACK_DAYS)

    signals, fresh_count, active_zones, current_price = compute_signals(
        df, coin, active_trades, params)

    # Single summary line per coin
    base = f"[{coin}] price @ ${fmt_price(current_price)} | {fresh_count} zones ready"
    if signals:
        log.info(f"{base} | {len(signals)} signal(s)")
    elif fresh_count == 0:
        log.info(f"{base} | no signals")
    else:
        nearest = None
        nearest_dist = float("inf")
        for z in active_zones:
            if not z["_fresh"]:
                continue
            ep = z["top"] if z["type"] == "demand" else z["bottom"]
            d = abs(current_price - ep) / current_price
            if d < nearest_dist:
                nearest_dist = d
                nearest = z
        if nearest:
            ep = nearest["top"] if nearest["type"] == "demand" else nearest["bottom"]
            log.info(f"{base} | no signals | nearest: {nearest['type']} @ ${fmt_price(ep)} ({nearest_dist*100:.1f}% away)")
        else:
            log.info(f"{base} | no signals")
    return signals


# ── Order management ──────────────────────────────────────────────────────────


def _round_price(price: float, tick_size: float) -> float:
    """Round price to instrument's tick size."""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(price / tick_size) * tick_size, 10)


def _round_size(size: float, symbol: str) -> float:
    """Round size to instrument's precision, enforce minimum."""
    specs = _instrument_specs.get(symbol, {})
    min_size = specs.get("min_order_size", 1)
    precision = specs.get("contract_value_trade_precision", 0)
    size = round(size, precision)
    return max(min_size, size)


def place_entry(client: KrakenFuturesClient, signal: dict,
                risk_amount: float) -> dict:
    """Place a limit entry order for a signal. Returns trade state dict."""
    trade_id = str(uuid.uuid4())[:8]
    specs = _instrument_specs.get(signal["symbol"], {})
    tick = specs.get("tick_size", 0.01)

    position_size_usd = risk_amount / signal["risk"] * signal["entry_price"]
    size = _round_size(position_size_usd, signal["symbol"])

    entry_price = _round_price(signal["entry_price"], tick)
    sl_price = _round_price(signal["sl"], tick)
    tp_price = _round_price(signal["tp"], tick)

    cli_id = f"sd_{trade_id}_entry"

    result = client.send_order(
        symbol=signal["symbol"],
        side=signal["side"],
        size=size,
        order_type="lmt",
        limit_price=entry_price,
        post_only=True,
        cli_ord_id=cli_id,
    )

    order_id = result.get("sendStatus", {}).get("order_id", "")

    trade = {
        "trade_id": trade_id,
        "coin": signal["coin"],
        "symbol": signal["symbol"],
        "zone_type": signal["zone_type"],
        "side": signal["side"],
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "risk": signal["risk"],
        "risk_amount": round(risk_amount, 2),
        "rr_target": signal["rr_target"],
        "size": size,
        "entry_order_id": order_id,
        "entry_cli_id": cli_id,
        "sl_order_id": None,
        "tp_order_id": None,
        "status": "pending_entry",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_filled_at": None,
        "closed_at": None,
        "result": None,
        "pnl_r": None,
    }

    log.info(f"[{trade_id}] ENTRY PLACED: {signal['side']} {size} {signal['symbol']} "
             f"@ ${fmt_price(signal['entry_price'])} | "
             f"SL=${fmt_price(signal['sl'])} TP=${fmt_price(signal['tp'])} "
             f"RR={signal['rr_target']} | risk=${risk_amount:.2f}")

    return trade


def place_sl_tp(client: KrakenFuturesClient, trade: dict):
    """Place SL and TP orders after entry fill. Handles partial placement."""
    trade_id = trade["trade_id"]
    exit_side = "sell" if trade["side"] == "buy" else "buy"

    # SL = stop market, reduce only (skip if already placed)
    if not trade.get("sl_order_id"):
        sl_cli = f"sd_{trade_id}_sl"
        sl_result = client.send_order(
            symbol=trade["symbol"],
            side=exit_side,
            size=trade["size"],
            order_type="stp",
            stop_price=trade["sl_price"],
            reduce_only=True,
            cli_ord_id=sl_cli,
        )
        trade["sl_order_id"] = sl_result.get("sendStatus", {}).get("order_id", "")

    # TP = limit post-only, reduce only (skip if already placed)
    if not trade.get("tp_order_id"):
        tp_cli = f"sd_{trade_id}_tp"
        tp_result = client.send_order(
            symbol=trade["symbol"],
            side=exit_side,
            size=trade["size"],
            order_type="lmt",
            limit_price=trade["tp_price"],
            reduce_only=True,
            post_only=True,
            cli_ord_id=tp_cli,
        )
        trade["tp_order_id"] = tp_result.get("sendStatus", {}).get("order_id", "")

    trade["status"] = "active"
    log.info(f"[{trade_id}] SL+TP PLACED: SL={trade['sl_price']} TP={trade['tp_price']}")


def check_fills(client: KrakenFuturesClient, state: dict):
    """Check order status and manage trade lifecycle."""
    open_orders = client.get_open_orders()
    open_order_ids = {o.get("order_id") for o in open_orders}

    # Get recent fills — aggregate by cliOrdId (handles partial fills)
    try:
        recent_fills = client.get_fills()
    except Exception as e:
        log.warning(f"Could not fetch fills: {e}")
        recent_fills = []

    fills_by_cli = {}  # cliOrdId -> {"total_size": float, "avg_price": float}
    for f in recent_fills:
        cli = f.get("cliOrdId", "")
        if not cli:
            continue
        size = float(f.get("size", 0))
        price = float(f.get("price", 0))
        if cli in fills_by_cli:
            prev = fills_by_cli[cli]
            total = prev["total_size"] + size
            # Weighted average price
            prev["avg_price"] = (prev["avg_price"] * prev["total_size"] + price * size) / total
            prev["total_size"] = total
        else:
            fills_by_cli[cli] = {"total_size": size, "avg_price": price}

    for trade in state["trades"]:
        if trade["status"] == "pending_entry":
            entry_fill = fills_by_cli.get(trade.get("entry_cli_id", ""))
            order_still_open = trade["entry_order_id"] in open_order_ids

            # Partial fill while order still open — protect the filled portion
            if entry_fill and entry_fill["total_size"] > 0 and order_still_open:
                filled_size = entry_fill["total_size"]
                fill_price = entry_fill["avg_price"]
                log.warning(f"[{trade['trade_id']}] PARTIAL FILL while order open: "
                            f"{filled_size}/{trade['size']} @ {fill_price} — "
                            f"cancelling remainder, placing SL+TP on filled portion")
                try:
                    client.cancel_order(trade["entry_order_id"])
                except Exception as e:
                    log.warning(f"Cancel remaining entry failed: {e}")
                trade["size"] = _round_size(filled_size, trade["symbol"])
                trade["entry_price_actual"] = fill_price
                trade["entry_filled_at"] = datetime.now(timezone.utc).isoformat()
                try:
                    place_sl_tp(client, trade)
                except Exception as e:
                    log.error(f"[{trade['trade_id']}] Failed to place SL+TP: {e}")
                    trade["status"] = "error_no_sl_tp"

            # Order fully gone — check if filled or rejected
            elif not order_still_open:
                if entry_fill and entry_fill["total_size"] > 0:
                    filled_size = entry_fill["total_size"]
                    fill_price = entry_fill["avg_price"]

                    # Handle partial fill: use actual filled size for SL/TP
                    if filled_size < trade["size"] * 0.9:
                        log.warning(f"[{trade['trade_id']}] PARTIAL FILL: "
                                    f"{filled_size}/{trade['size']} — adjusting size")
                        trade["size"] = _round_size(filled_size, trade["symbol"])

                    trade["entry_price_actual"] = fill_price
                    trade["entry_filled_at"] = datetime.now(timezone.utc).isoformat()
                    log.info(f"[{trade['trade_id']}] ENTRY FILLED: "
                             f"{filled_size} @ {fill_price}! Placing SL+TP...")
                    try:
                        place_sl_tp(client, trade)
                    except Exception as e:
                        log.error(f"[{trade['trade_id']}] Failed to place SL+TP: {e}")
                        trade["status"] = "error_no_sl_tp"
                else:
                    # Order gone and no fill = rejected/cancelled
                    log.warning(f"[{trade['trade_id']}] Entry order gone, no fill — "
                                f"likely rejected (post-only)")
                    trade["status"] = "cancelled"
                    trade["closed_at"] = datetime.now(timezone.utc).isoformat()

        elif trade["status"] == "active":
            sl_alive = trade["sl_order_id"] in open_order_ids
            tp_alive = trade["tp_order_id"] in open_order_ids

            if not sl_alive and tp_alive:
                # SL filled — cancel TP
                log.info(f"[{trade['trade_id']}] SL HIT — cancelling TP")
                try:
                    client.cancel_order(trade["tp_order_id"])
                except Exception as e:
                    log.warning(f"Cancel TP failed: {e}")
                trade["status"] = "closed"
                trade["result"] = "loss"
                trade["pnl_r"] = -1.0
                trade["closed_at"] = datetime.now(timezone.utc).isoformat()
                log.info(f"[{trade['trade_id']}] CLOSED: loss (-1R)")

            elif sl_alive and not tp_alive:
                # TP filled — cancel SL
                log.info(f"[{trade['trade_id']}] TP HIT — cancelling SL")
                try:
                    client.cancel_order(trade["sl_order_id"])
                except Exception as e:
                    log.warning(f"Cancel SL failed: {e}")
                trade["status"] = "closed"
                trade["result"] = "win"
                trade["pnl_r"] = trade["rr_target"]
                trade["closed_at"] = datetime.now(timezone.utc).isoformat()
                log.info(f"[{trade['trade_id']}] CLOSED: win (+{trade['rr_target']:.1f}R)")

            elif not sl_alive and not tp_alive:
                # Both gone — determine from fills
                sl_fill = fills_by_cli.get(f"sd_{trade['trade_id']}_sl")
                tp_fill = fills_by_cli.get(f"sd_{trade['trade_id']}_tp")
                trade["status"] = "closed"
                trade["closed_at"] = datetime.now(timezone.utc).isoformat()
                if tp_fill and not sl_fill:
                    trade["result"] = "win"
                    trade["pnl_r"] = trade["rr_target"]
                elif sl_fill and not tp_fill:
                    trade["result"] = "loss"
                    trade["pnl_r"] = -1.0
                else:
                    # Both filled or neither — check fill prices
                    trade["result"] = "unknown"
                    trade["pnl_r"] = 0.0
                log.info(f"[{trade['trade_id']}] CLOSED: {trade['result']} "
                         f"({trade['pnl_r']:+.1f}R)")

    # Move closed trades to completed list
    still_active = []
    for trade in state["trades"]:
        if trade["status"] in ("closed", "cancelled", "error_no_sl_tp"):
            if trade["pnl_r"] is not None:
                state["total_r"] += trade["pnl_r"]
            state["completed"].append(trade)
        else:
            still_active.append(trade)
    state["trades"] = still_active


# ── Stale order cleanup ──────────────────────────────────────────────────────


def cancel_stale_entries(client: KrakenFuturesClient, state: dict,
                         max_age_hours: int = 4):
    """Cancel entry orders that have been pending too long."""
    now = datetime.now(timezone.utc)
    for trade in state["trades"]:
        if trade["status"] != "pending_entry":
            continue
        created = datetime.fromisoformat(trade["created_at"])
        age_hours = (now - created).total_seconds() / 3600
        if age_hours > max_age_hours:
            log.info(f"[{trade['trade_id']}] Cancelling stale entry ({age_hours:.1f}h old)")
            try:
                client.cancel_order(trade["entry_order_id"])
            except Exception as e:
                log.warning(f"Cancel stale entry failed: {e}")
            trade["status"] = "cancelled"
            trade["closed_at"] = now.isoformat()


# ── Main loop ─────────────────────────────────────────────────────────────────


def next_candle_close() -> float:
    """Seconds until next 30m candle close (+ 10s buffer for data)."""
    now = time.time()
    current_slot = now // CANDLE_SECONDS
    next_close = (current_slot + 1) * CANDLE_SECONDS + 10
    return max(0, next_close - now)


def _sync_positions(client: KrakenFuturesClient, state: dict, coins: list[str]):
    """Check exchange positions against farm state. Warn about orphaned positions."""
    try:
        positions = client.get_open_positions()
    except Exception as e:
        log.error(f"Could not fetch positions for sync: {e}")
        return

    real_positions = {}
    for p in positions:
        size = abs(float(p.get("size", 0)))
        if size > 0:
            real_positions[p["symbol"]] = {"size": size, "side": p.get("side", "")}

    # Symbols our state knows about
    tracked_symbols = {t["symbol"] for t in state["trades"]
                       if t["status"] in ("active", "error_no_sl_tp")}

    # Check for exchange positions not in our state
    for symbol, pos in real_positions.items():
        coin = None
        for c, s in COIN_MAP.items():
            if s == symbol:
                coin = c
                break
        if coin not in coins:
            continue
        if symbol not in tracked_symbols:
            log.error(f"ORPHANED POSITION: {symbol} {pos['side']} size={pos['size']} "
                      f"— not tracked by farm. Close manually or add to state.")

    # Check for state trades whose positions no longer exist
    for trade in state["trades"]:
        if trade["status"] == "active" and trade["symbol"] not in real_positions:
            log.warning(f"[{trade['trade_id']}] State says active but no position on exchange. "
                        f"Marking as closed (unknown result).")
            trade["status"] = "closed"
            trade["result"] = "unknown"
            trade["pnl_r"] = 0.0
            trade["closed_at"] = datetime.now(timezone.utc).isoformat()


def _recover_unprotected(client: KrakenFuturesClient, state: dict):
    """On startup, retry SL+TP placement for trades stuck in error_no_sl_tp."""
    for trade in state["trades"]:
        if trade["status"] != "error_no_sl_tp":
            continue
        log.warning(f"[{trade['trade_id']}] Recovering unprotected position — placing SL+TP")
        try:
            place_sl_tp(client, trade)
            log.info(f"[{trade['trade_id']}] Recovery successful")
        except Exception as e:
            log.error(f"[{trade['trade_id']}] Recovery FAILED: {e}. "
                      f"MANUAL ACTION NEEDED: {trade['symbol']} {trade['side']} "
                      f"size={trade['size']}")


def run_dry(coins: list[str], risk_pct: float):
    """Single-cycle dry run — scan signals without placing orders."""
    params = OPTIMAL_PARAMS
    log.info("DRY-RUN — scanning signals, no orders will be placed")

    total_signals = 0
    for coin in coins:
        ccxt_symbol = CCXT_MAP[coin]
        df = fetch_ohlcv_full(ccxt_symbol, TIMEFRAME, days=LOOKBACK_DAYS)
        signals, fresh_count, _, current_price = compute_signals(
            df, coin, [], params)

        base = f"[{coin}] price @ ${fmt_price(current_price)} | {fresh_count} zones ready"
        if not signals:
            log.info(f"{base} | no signals")
            continue

        log.info(f"{base} | {len(signals)} signal(s):")
        for sig in signals:
            log.info(f"  → {sig['side'].upper()} @ ${fmt_price(sig['entry_price'])} "
                     f"SL=${fmt_price(sig['sl'])} TP=${fmt_price(sig['tp'])} "
                     f"RR={sig['rr_target']:.1f} ({sig['zone_type']})")
        total_signals += len(signals)

    log.info(f"DRY-RUN complete — {total_signals} signal(s) across {len(coins)} coins")


def run_farm(coins: list[str], max_trades: int, risk_pct: float):
    """Main farm loop."""
    global _instrument_specs

    api_key = os.environ.get("KRAKEN_FUTURES_API_KEY")
    api_secret = os.environ.get("KRAKEN_FUTURES_API_SECRET")
    if not api_key or not api_secret:
        log.error("Set KRAKEN_FUTURES_API_KEY and KRAKEN_FUTURES_API_SECRET env vars")
        sys.exit(1)

    client = KrakenFuturesClient(api_key, api_secret)
    state = load_state()
    params = OPTIMAL_PARAMS

    # Load instrument specs and set leverage
    for coin in coins:
        symbol = COIN_MAP[coin]
        try:
            _instrument_specs[symbol] = client.get_instrument_specs(symbol)
        except Exception as e:
            log.warning(f"[{coin}] specs failed: {e}")
        try:
            client.set_leverage(symbol, LEVERAGE)
        except Exception as e:
            log.warning(f"[{coin}] leverage failed: {e}")

    # Recover trades stuck without SL/TP from previous crash
    _recover_unprotected(client, state)

    # Sync state with actual exchange positions
    _sync_positions(client, state, coins)
    save_state(state)

    # Startup info
    balance = client.get_balance()
    risk_amount = balance * (risk_pct / 100)
    completed = len(state['completed'])
    status = f"{completed} trades ({state['total_r']:+.1f}R)" if completed else "no trades yet"
    log.info(f"Supply/Demand Trading Farm")
    log.info(f"  Balance: ${balance:,.2f} | Risk/trade: ${risk_amount:,.2f} ({risk_pct}%) | "
             f"Max trades: {max_trades} | Leverage: {LEVERAGE}x")
    log.info(f"  Coins: {', '.join(coins)} | Status: {status}")

    last_signal_check = 0
    _last_status_log = 0

    while True:
        try:
            now = time.time()

            # ── Check fills every POLL_INTERVAL seconds ───────────────
            check_fills(client, state)
            cancel_stale_entries(client, state)
            save_state(state)

            # ── Signal check at candle close ──────────────────────────
            wait = next_candle_close()
            active_count = len([t for t in state["trades"]
                                if t["status"] in ("pending_entry", "active")])

            if wait < POLL_INTERVAL and now - last_signal_check > CANDLE_SECONDS * 0.8:
                last_signal_check = now
                balance = client.get_balance()
                risk_amount = balance * (risk_pct / 100)

                if risk_amount < 1:
                    log.warning(f"Risk amount too small (${risk_amount:.2f}), skipping")
                else:
                    for coin in coins:
                        if active_count >= max_trades:
                            log.info(f"Max trades ({max_trades}) reached, skipping signals")
                            break

                        try:
                            signals = get_signals(coin, state["trades"], params)
                            for sig in signals:
                                if active_count >= max_trades:
                                    break
                                log.info(f"[{coin}] Signal: {sig['side']} @ ${fmt_price(sig['entry_price'])} "
                                         f"({sig['zone_type']}) RR={sig['rr_target']} "
                                         f"str={sig['strength']:.1f}")
                                trade = place_entry(client, sig, risk_amount)
                                state["trades"].append(trade)
                                active_count += 1
                                save_state(state)
                        except Exception as e:
                            log.error(f"[{coin}] Signal error: {e}")

            # ── Heartbeat (once per candle cycle) ────────────────────
            if time.time() - _last_status_log >= CANDLE_SECONDS:
                pending = len([t for t in state["trades"] if t["status"] == "pending_entry"])
                active = len([t for t in state["trades"] if t["status"] == "active"])
                completed = len(state["completed"])
                log.info(f"[HEARTBEAT] {pending} pending | {active} active | "
                         f"{completed} completed | P&L: {state['total_r']:+.1f}R")
                _last_status_log = time.time()

            # Sleep until next poll
            time.sleep(min(POLL_INTERVAL, wait))

        except KeyboardInterrupt:
            log.info("Shutting down gracefully...")
            save_state(state)
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)
            save_state(state)
            time.sleep(10)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply/Demand Trading Farm")
    parser.add_argument("--coins", type=str, required=True,
                        help="Comma-separated coins (e.g. AVAX,ETH)")
    parser.add_argument("--max-trades", type=int, default=4,
                        help="Max simultaneous trades (default: 4)")
    parser.add_argument("--risk-pct", type=float, default=0.5,
                        help="Risk per trade as %% of balance (default: 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan signals without placing orders (single cycle)")
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",")]
    for c in coins:
        if c not in COIN_MAP:
            log.error(f"Unknown coin: {c}. Available: {', '.join(COIN_MAP.keys())}")
            sys.exit(1)

    if args.dry_run:
        run_dry(coins, args.risk_pct)
    else:
        run_farm(coins, args.max_trades, args.risk_pct)
