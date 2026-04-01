"""Walk-forward backtest engine (no lookahead bias) — Optimized"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BT_SL_ATR_MULT, BT_TP_RR, BT_MIN_RR,
    BT_MAX_OPEN_TRADES, BT_ZONE_CLUSTER_PCT,
    BT_EXCLUDE_MONTHS,
    INITIAL_BALANCE, ATR_PERIOD, WICK_TOLERANCE_PCT,
    RISK_PER_TRADE_PCT, RISK_CAP_PCT, FEE_RATE_MAKER, FEE_RATE_TAKER, SL_SLIPPAGE_PCT,
)
from strategy import tag_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_dynamic_tp(
    zone: dict,
    entry_price: float,
    risk: float,
    active_zones: list[dict],
    liq_levels: list[dict] | None,
    params: dict | None = None,
) -> float:
    """
    Find TP at next opposite zone or liquidity level.
    Falls back to fixed RR if no valid target found or RR too low.
    """
    p = params or {}
    _tp_rr  = p.get("BT_TP_RR",  BT_TP_RR)
    _min_rr = p.get("BT_MIN_RR", BT_MIN_RR)

    candidates: list[float] = []

    if zone["type"] == "demand":
        for z in active_zones:
            if z["type"] == "supply" and z.get("_fresh", True) and z["bottom"] > entry_price:
                candidates.append(z["bottom"])
        if liq_levels:
            for lv in liq_levels:
                if lv["type"] == "equal_highs" and lv["price"] > entry_price:
                    candidates.append(lv["price"])
        if candidates:
            tp = min(candidates)
            if abs(tp - entry_price) / risk >= _min_rr:
                return tp
        return entry_price + risk * _tp_rr
    else:
        for z in active_zones:
            if z["type"] == "demand" and z.get("_fresh", True) and z["top"] < entry_price:
                candidates.append(z["top"])
        if liq_levels:
            for lv in liq_levels:
                if lv["type"] == "equal_lows" and lv["price"] < entry_price:
                    candidates.append(lv["price"])
        if candidates:
            tp = max(candidates)
            if abs(entry_price - tp) / risk >= _min_rr:
                return tp
        return entry_price - risk * _tp_rr


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def walkforward_backtest(
    df: pd.DataFrame,
    zones: list[dict],
    atr_series: pd.Series,
    liq_levels: list[dict] | None = None,
    params: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    p = params or {}
    _sl_atr          = p.get("BT_SL_ATR_MULT",       BT_SL_ATR_MULT)
    _max_open        = p.get("BT_MAX_OPEN_TRADES",    BT_MAX_OPEN_TRADES)
    _cluster_pct     = p.get("BT_ZONE_CLUSTER_PCT",   BT_ZONE_CLUSTER_PCT)
    _exclude_months  = p.get("EXCLUDE_MONTHS",        BT_EXCLUDE_MONTHS)
    _max_zone_age    = p.get("MAX_ZONE_AGE_BARS",      None)

    _ambiguous_sl_first   = p.get("AMBIGUOUS_SL_FIRST",       True)
    _entry_slip_pct       = p.get("BT_ENTRY_SLIPPAGE_PCT",    0.0)
    _confirm_delay        = p.get("BT_CONFIRM_DELAY",         False)
    _scale_sl_slip        = p.get("BT_SCALE_SL_SLIPPAGE",     False)
    _max_notional_vol_pct = p.get("BT_MAX_NOTIONAL_VOL_PCT",  0.0)

    # ── Extract DataFrame columns to NumPy arrays once ─────────────────────
    highs   = df["high"].to_numpy(dtype=np.float64)
    lows    = df["low"].to_numpy(dtype=np.float64)
    closes  = df["close"].to_numpy(dtype=np.float64)
    opens   = df["open"].to_numpy(dtype=np.float64)
    volumes = df["volume"].to_numpy(dtype=np.float64) if "volume" in df.columns \
              else np.zeros(len(df), dtype=np.float64)
    times   = df.index

    regime_arr = tag_regime(df).to_numpy()
    atr_arr    = atr_series.to_numpy(dtype=np.float64)

    # ── Sort zones by impulse_idx ───────────────────────────────────────────
    zones_sorted = sorted(zones, key=lambda z: z["impulse_idx"])
    zone_ptr     = 0

    # ── State ──────────────────────────────────────────────────────────────
    active_zones:  list[dict] = []
    open_trades:   list[dict] = []
    closed_trades: list[dict] = []
    equity_curve:  list[dict] = []

    balance       = float(INITIAL_BALANCE)
    peak_balance  = balance
    peak_r        = 0.0
    cumulative_r  = 0.0
    max_dd_r      = 0.0
    max_dd_pct    = 0.0

    warmup = max(ATR_PERIOD + 5, 20)
    n      = len(df)

    for i in range(warmup, n):
        # ── Read bar data from NumPy arrays (no .iloc) ─────────────────────
        bar_high   = highs[i]
        bar_low    = lows[i]
        bar_close  = closes[i]
        bar_open   = opens[i]
        bar_volume = volumes[i]
        bar_time   = times[i]
        bar_range  = bar_high - bar_low

        # ── 1. Activate new zones ──────────────────────────────────────────
        while zone_ptr < len(zones_sorted) and zones_sorted[zone_ptr]["impulse_idx"] <= i:
            z = dict(zones_sorted[zone_ptr])
            z["_fresh"]         = True
            z["_mitigated"]     = False
            z["_trade_opened"]  = False
            z["trade_score"]    = z["strength"]

            clustered = False
            for existing in active_zones:
                if (
                    existing["type"] == z["type"]
                    and existing["_fresh"]
                    and abs(existing["mid"] - z["mid"]) / z["mid"] < _cluster_pct
                ):
                    if z["strength"] > existing["strength"]:
                        existing["_fresh"] = False
                    else:
                        clustered = True
                    break

            if not clustered:
                active_zones.append(z)
            zone_ptr += 1

        # ── 2. Update zone freshness + PRUNE stale zones inline ────────────
        #
        # OPTIMIZATION vs original:
        #   The original loop kept every zone in active_zones forever, even
        #   after _fresh became False.  This caused the list to grow to
        #   O(total_zones) and iterate that many dicts per bar.
        #
        #   We now rebuild active_zones in a single pass:
        #   • Hard-broken zones (close breaks through) are dropped immediately.
        #   • All other zones (fresh, mitigated-but-fresh) are kept.
        #   Result: active_zones only ever contains zones that are still live.
        #
        #   Behavioural identity:
        #   • Stale zones (_fresh=False) never participate in entries (guarded
        #     by `not z["_fresh"]`) or TP targeting (guarded by z.get("_fresh")).
        #     Dropping them early is therefore semantically a no-op.
        #   • Zones made stale by MAX_ZONE_AGE are also dropped on the same bar.
        # ──────────────────────────────────────────────────────────────────
        next_active: list[dict] = []
        for z in active_zones:
            if not z["_fresh"]:
                # Already stale from a prior bar — drop it.
                continue

            # Age-based expiry (optional, same bar as original)
            if _max_zone_age is not None and (i - z["impulse_idx"]) > _max_zone_age:
                z["_fresh"] = False
                continue          # drop

            tolerance = z["mid"] * WICK_TOLERANCE_PCT
            if z["type"] == "demand":
                if bar_close < z["bottom"] - tolerance:
                    z["_fresh"] = False
                    continue      # drop — hard break below demand
                elif bar_low <= z["top"] and not z["_mitigated"]:
                    z["_mitigated"] = True
            else:
                if bar_close > z["top"] + tolerance:
                    z["_fresh"] = False
                    continue      # drop — hard break above supply
                elif bar_high >= z["bottom"] and not z["_mitigated"]:
                    z["_mitigated"] = True

            next_active.append(z)   # still live → keep

        active_zones = next_active

        # ── 3. Check for new trade entries ─────────────────────────────────
        if len(open_trades) < _max_open:
            candidates: list[dict] = []
            for z in active_zones:
                if z["_trade_opened"] or not z["_fresh"]:
                    continue

                zone_reached = False
                if z["type"] == "demand" and bar_low <= z["top"]:
                    zone_reached = True
                elif z["type"] == "supply" and bar_high >= z["bottom"]:
                    zone_reached = True

                if not zone_reached:
                    continue

                if _confirm_delay:
                    if not z.get("_touched_bar"):
                        z["_touched_bar"] = i
                        continue
                    elif z["_touched_bar"] == i:
                        continue

                candidates.append(z)

            if candidates:
                candidates.sort(key=lambda z: -z.get("trade_score", 0))

                month = bar_time.month
                for z in candidates:
                    if len(open_trades) >= _max_open:
                        break
                    if _exclude_months and month in _exclude_months:
                        continue

                    if z["type"] == "demand":
                        entry_price = z["top"]
                        if _entry_slip_pct:
                            entry_price *= (1.0 - _entry_slip_pct)
                    else:
                        entry_price = z["bottom"]
                        if _entry_slip_pct:
                            entry_price *= (1.0 + _entry_slip_pct)

                    atr_val = atr_arr[i]
                    if np.isnan(atr_val) or atr_val == 0.0:
                        continue

                    if z["type"] == "demand":
                        sl   = z["bottom"] - atr_val * _sl_atr
                        risk = entry_price - sl
                    else:
                        sl   = z["top"] + atr_val * _sl_atr
                        risk = sl - entry_price

                    if risk <= 0.0:
                        continue

                    tp = find_dynamic_tp(z, entry_price, risk, active_zones, liq_levels, params=p)
                    rr_target = abs(tp - entry_price) / risk

                    risk_amount = min(balance * RISK_PER_TRADE_PCT, INITIAL_BALANCE * RISK_CAP_PCT)
                    if risk_amount <= 0.0:
                        continue

                    position_size = risk_amount / risk

                    if _max_notional_vol_pct > 0.0 and bar_volume > 0.0:
                        max_notional = bar_volume * bar_close * _max_notional_vol_pct
                        capped_size  = max_notional / entry_price
                        if capped_size < position_size:
                            position_size = capped_size
                            risk_amount   = position_size * risk

                    est_fee = position_size * entry_price * (FEE_RATE_MAKER + FEE_RATE_TAKER + SL_SLIPPAGE_PCT)
                    if est_fee / risk_amount > 0.3:
                        continue

                    if z["type"] == "demand":
                        entry_penetration = (entry_price - bar_low) / entry_price
                    else:
                        entry_penetration = (bar_high - entry_price) / entry_price

                    trade = {
                        "zone":             z,
                        "zone_type":        z["type"],
                        "trade_score":      z.get("trade_score", z.get("strength", 0)),
                        "entry_bar":        i,
                        "entry_time":       bar_time,
                        "entry_price":      round(entry_price, 1),
                        "sl":               round(sl, 1),
                        "tp":               round(tp, 1),
                        "risk":             round(risk, 1),
                        "risk_amount":      round(risk_amount, 2),
                        "position_size":    position_size,
                        "rr_target":        round(rr_target, 2),
                        "with_trend":       z.get("with_trend", False),
                        "has_bos":          z.get("has_bos", False),
                        "liq_swept":        z.get("liq_swept", False),
                        "timeframe":        z.get("timeframe", ""),
                        "regime":           regime_arr[i],
                        "_mae":             0.0,
                        "_mfe":             0.0,
                        "entry_penetration":    entry_penetration,
                        "entry_bar_volume":     bar_volume,
                        "position_notional":    position_size * entry_price,
                    }
                    open_trades.append(trade)
                    z["_trade_opened"] = True

        # ── 4. Manage open trades (SL first = conservative by default) ─────
        still_open: list[dict] = []
        for trade in open_trades:
            risk_t = trade["risk"]

            if risk_t > 0.0:
                if trade["zone_type"] == "demand":
                    adverse   = (trade["entry_price"] - bar_low)  / risk_t
                    favorable = (bar_high - trade["entry_price"]) / risk_t
                else:
                    adverse   = (bar_high - trade["entry_price"]) / risk_t
                    favorable = (trade["entry_price"] - bar_low)  / risk_t
                if adverse  > trade["_mae"]: trade["_mae"] = adverse
                if favorable > trade["_mfe"]: trade["_mfe"] = favorable

            sl_hit = tp_hit = False
            if trade["zone_type"] == "demand":
                sl_hit = bar_low  <= trade["sl"]
                tp_hit = bar_high >= trade["tp"]
            else:
                sl_hit = bar_high >= trade["sl"]
                tp_hit = bar_low  <= trade["tp"]

            if sl_hit and tp_hit:
                if _ambiguous_sl_first:
                    tp_hit = False
                else:
                    if trade["zone_type"] == "demand":
                        if bar_open >= trade["entry_price"]:
                            sl_hit = False
                        else:
                            tp_hit = False
                    else:
                        if bar_open <= trade["entry_price"]:
                            sl_hit = False
                        else:
                            tp_hit = False

            resolved = False
            if sl_hit:
                if _scale_sl_slip and bar_range > 0.0 and trade["entry_price"] > 0.0:
                    dynamic_slip = max(SL_SLIPPAGE_PCT,
                                       bar_range / trade["entry_price"] * 0.1)
                else:
                    dynamic_slip = SL_SLIPPAGE_PCT

                trade["result"] = "loss"
                trade["rr"]     = -1.0
                if trade["zone_type"] == "demand":
                    trade["exit_price"] = trade["sl"] * (1.0 - dynamic_slip)
                else:
                    trade["exit_price"] = trade["sl"] * (1.0 + dynamic_slip)
                resolved = True

            elif tp_hit:
                trade["result"]     = "win"
                trade["rr"]         = trade["rr_target"]
                trade["exit_price"] = trade["tp"]
                resolved            = True

            if resolved:
                trade["mae"]  = round(trade.pop("_mae"), 3)
                trade["mfe"]  = round(trade.pop("_mfe"), 3)
                trade["exit_bar"]        = i
                trade["exit_time"]       = bar_time
                trade["exit_bar_volume"] = bar_volume

                if trade["result"] == "win" and trade["tp"] > 0.0:
                    if trade["zone_type"] == "demand":
                        trade["tp_penetration"] = (bar_high - trade["tp"]) / trade["tp"]
                    else:
                        trade["tp_penetration"] = (trade["tp"] - bar_low)  / trade["tp"]
                else:
                    trade["tp_penetration"] = 0.0

                pos       = trade["position_size"]
                entry_fee = pos * trade["entry_price"] * FEE_RATE_MAKER
                if trade["result"] == "win":
                    exit_fee = pos * trade["exit_price"] * FEE_RATE_MAKER
                else:
                    exit_fee = pos * trade["exit_price"] * (FEE_RATE_TAKER + SL_SLIPPAGE_PCT)

                total_fee = entry_fee + exit_fee
                fee_in_r  = total_fee / trade["risk_amount"] if trade["risk_amount"] > 0.0 else 0.0

                trade["fee"]      = round(total_fee, 2)
                trade["fee_r"]    = round(fee_in_r, 3)
                trade["rr_gross"] = trade["rr"]
                trade["rr"]       = round(trade["rr"] - fee_in_r, 3)

                pnl = trade["risk_amount"] * trade["rr"]
                trade["pnl"]           = round(pnl, 2)
                balance               += pnl
                trade["balance_after"] = round(balance, 2)
                cumulative_r          += trade["rr"]
                closed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── 5. Equity tracking ─────────────────────────────────────────────
        if cumulative_r > peak_r:
            peak_r = cumulative_r
        if balance > peak_balance:
            peak_balance = balance

        dd_r   = peak_r - cumulative_r
        dd_pct = (peak_balance - balance) / peak_balance * 100.0 if peak_balance > 0.0 else 0.0
        if dd_r   > max_dd_r:   max_dd_r   = dd_r
        if dd_pct > max_dd_pct: max_dd_pct = dd_pct

        equity_curve.append({
            "bar":          i,
            "time":         bar_time,
            "cumulative_r": round(cumulative_r, 2),
            "balance":      round(balance, 2),
            "drawdown_r":   round(dd_r, 2),
            "drawdown_pct": round(dd_pct, 2),
        })

    # ── Mark remaining open trades ─────────────────────────────────────────
    final_close = float(closes[-1])
    final_time  = times[-1]
    for trade in open_trades:
        trade["result"]          = "open"
        trade["rr"]              = 0.0
        trade["pnl"]             = 0.0
        trade["fee"]             = 0.0
        trade["fee_r"]           = 0.0
        trade["rr_gross"]        = 0.0
        trade["mae"]             = trade.pop("_mae", 0.0)
        trade["mfe"]             = trade.pop("_mfe", 0.0)
        trade["balance_after"]   = round(balance, 2)
        trade["exit_bar"]        = n - 1
        trade["exit_time"]       = final_time
        trade["exit_price"]      = round(final_close, 1)
        closed_trades.append(trade)

    return closed_trades, equity_curve