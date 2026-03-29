"""Walk-forward backtest engine (no lookahead bias)."""

from __future__ import annotations

import pandas as pd
from config import (
    SESSION_ASIA, SESSION_LONDON, SESSION_NY,
    BT_SL_ATR_MULT, BT_TP_RR, BT_MIN_RR,
    BT_MAX_OPEN_TRADES, BT_ZONE_CLUSTER_PCT,
    BT_EXCLUDE_MONTHS,
    INITIAL_BALANCE, ATR_PERIOD, WICK_TOLERANCE_PCT,
    RISK_PER_TRADE_PCT, RISK_CAP_PCT, FEE_RATE_MAKER, FEE_RATE_TAKER, SL_SLIPPAGE_PCT,
)
from structure import tag_market_regime


def detect_session(timestamp) -> str:
    """Classify bar into trading session based on UTC hour."""
    hour = timestamp.hour
    if SESSION_ASIA[0] <= hour < SESSION_ASIA[1]:
        return "Asia"
    elif SESSION_LONDON[0] <= hour < SESSION_LONDON[1]:
        return "London"
    elif SESSION_NY[0] <= hour < SESSION_NY[1]:
        return "New York"
    return "Asia"


def find_dynamic_tp(zone: dict, entry_price: float, risk: float,
                    active_zones: list[dict], liq_levels: list[dict] | None,
                    params: dict | None = None) -> float:
    """
    Find TP at next opposite zone or liquidity level.
    Falls back to fixed RR if no valid target found or RR too low.
    """
    p = params or {}
    _tp_rr = p.get("BT_TP_RR", BT_TP_RR)
    _min_rr = p.get("BT_MIN_RR", BT_MIN_RR)

    candidates = []

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


def walkforward_backtest(df: pd.DataFrame, zones: list[dict], atr_series: pd.Series,
                         liq_levels: list[dict] | None = None,
                         params: dict | None = None) -> tuple[list[dict], list[dict]]:
    """
    Walk-forward backtest WITHOUT lookahead bias.

    Zones are pre-detected with no_lookahead=True (no future freshness).
    This function processes bar-by-bar:
      1. Activate zones as they form (impulse_idx), cluster duplicates
      2. Track freshness in real-time (body close through zone = invalidated)
      3. Open trades when conditions met (max open limit)
      4. Manage SL/TP on each bar (SL checked first = conservative)
      5. Track equity curve with real balance simulation

    Returns: (trades, equity_curve)
    """
    zones_sorted = sorted(zones, key=lambda z: z["impulse_idx"])

    p = params or {}
    _sl_atr = p.get("BT_SL_ATR_MULT", BT_SL_ATR_MULT)
    _max_open = p.get("BT_MAX_OPEN_TRADES", BT_MAX_OPEN_TRADES)
    _cluster_pct = p.get("BT_ZONE_CLUSTER_PCT", BT_ZONE_CLUSTER_PCT)
    _exclude_months = p.get("EXCLUDE_MONTHS", BT_EXCLUDE_MONTHS)
    _max_zone_age = p.get("MAX_ZONE_AGE_BARS", None)

    active_zones = []
    open_trades = []
    closed_trades = []
    equity_curve = []
    cumulative_r = 0.0
    balance = INITIAL_BALANCE
    peak_balance = balance
    peak_r = 0.0
    max_dd_r = 0.0
    max_dd_pct = 0.0
    zone_ptr = 0

    regime_series = tag_market_regime(df)
    warmup = max(ATR_PERIOD + 5, 20)

    for i in range(warmup, len(df)):
        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]
        bar_close = df["close"].iloc[i]
        bar_time = df.index[i]
        bar_volume = df["volume"].iloc[i] if "volume" in df.columns else 0

        # -- 1. Activate new zones, cluster duplicates --
        while zone_ptr < len(zones_sorted) and zones_sorted[zone_ptr]["impulse_idx"] <= i:
            z = dict(zones_sorted[zone_ptr])
            z["_fresh"] = True
            z["_mitigated"] = False
            z["_trade_opened"] = False
            z["trade_score"] = z["strength"]

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

        # -- 2. Update zone freshness (real-time, no lookahead) --
        for z in active_zones:
            if not z["_fresh"]:
                continue
            # Expire zones older than max age
            if _max_zone_age is not None and (i - z["impulse_idx"]) > _max_zone_age:
                z["_fresh"] = False
                continue
            tolerance = z["mid"] * WICK_TOLERANCE_PCT
            if z["type"] == "demand":
                if bar_close < z["bottom"] - tolerance:
                    z["_fresh"] = False
                elif bar_low <= z["top"] and not z["_mitigated"]:
                    z["_mitigated"] = True
            else:
                if bar_close > z["top"] + tolerance:
                    z["_fresh"] = False
                elif bar_high >= z["bottom"] and not z["_mitigated"]:
                    z["_mitigated"] = True

        # -- 3. Check for new trade entries --
        candidates = []
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

            candidates.append(z)

        candidates.sort(key=lambda z: -z.get("trade_score", 0))

        for z in candidates:
            if len(open_trades) >= _max_open:
                break

            if _exclude_months and bar_time.month in _exclude_months:
                continue

            session = detect_session(bar_time)

            entry_price = z["top"] if z["type"] == "demand" else z["bottom"]

            atr_val = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else 0
            if atr_val == 0:
                continue

            if z["type"] == "demand":
                sl = z["bottom"] - atr_val * _sl_atr
                risk = entry_price - sl
            else:
                sl = z["top"] + atr_val * _sl_atr
                risk = sl - entry_price

            if risk <= 0:
                continue

            tp = find_dynamic_tp(z, entry_price, risk, active_zones, liq_levels, params=params)
            rr_target = abs(tp - entry_price) / risk

            risk_amount = min(balance * RISK_PER_TRADE_PCT, INITIAL_BALANCE * RISK_CAP_PCT)
            if risk_amount <= 0:
                continue
            position_size = risk_amount / risk

            # Skip if estimated round-trip fees exceed 0.3R
            # Worst case: maker entry + taker SL + SL slippage
            est_fee = position_size * entry_price * (FEE_RATE_MAKER + FEE_RATE_TAKER + SL_SLIPPAGE_PCT)
            if est_fee / risk_amount > 0.3:
                continue

            trade = {
                "zone": z,
                "zone_type": z["type"],
                "trade_score": z.get("trade_score", z.get("strength", 0)),
                "entry_bar": i,
                "entry_time": bar_time,
                "entry_price": round(entry_price, 1),
                "sl": round(sl, 1),
                "tp": round(tp, 1),
                "risk": round(risk, 1),
                "risk_amount": round(risk_amount, 2),
                "position_size": position_size,
                "rr_target": round(rr_target, 2),
                "session": session,
                "with_trend": z.get("with_trend", False),
                "has_bos": z.get("has_bos", False),
                "liq_swept": z.get("liq_swept", False),
                "timeframe": z.get("timeframe", ""),
                "regime": regime_series.iloc[i],
            }
            trade["_mae"] = 0.0  # max adverse excursion (in R)
            trade["_mfe"] = 0.0  # max favorable excursion (in R)
            # Limit order feasibility tracking
            if entry_price > 0:
                if z["type"] == "demand":
                    trade["entry_penetration"] = (entry_price - bar_low) / entry_price
                else:
                    trade["entry_penetration"] = (bar_high - entry_price) / entry_price
            else:
                trade["entry_penetration"] = 0.0
            trade["entry_bar_volume"] = bar_volume
            trade["position_notional"] = position_size * entry_price
            open_trades.append(trade)
            z["_trade_opened"] = True

        # -- 4. Manage open trades (SL first = conservative) --
        still_open = []
        for trade in open_trades:
            # Track MAE/MFE before resolution
            risk = trade["risk"]
            if risk > 0:
                if trade["zone_type"] == "demand":
                    adverse = (trade["entry_price"] - bar_low) / risk
                    favorable = (bar_high - trade["entry_price"]) / risk
                else:
                    adverse = (bar_high - trade["entry_price"]) / risk
                    favorable = (trade["entry_price"] - bar_low) / risk
                trade["_mae"] = max(trade["_mae"], adverse)
                trade["_mfe"] = max(trade["_mfe"], favorable)

            resolved = False
            if trade["zone_type"] == "demand":
                if bar_low <= trade["sl"]:
                    trade["result"] = "loss"
                    trade["rr"] = -1.0
                    trade["exit_price"] = trade["sl"]
                    resolved = True
                elif bar_high >= trade["tp"]:
                    trade["result"] = "win"
                    trade["rr"] = trade["rr_target"]
                    trade["exit_price"] = trade["tp"]
                    resolved = True
            else:
                if bar_high >= trade["sl"]:
                    trade["result"] = "loss"
                    trade["rr"] = -1.0
                    trade["exit_price"] = trade["sl"]
                    resolved = True
                elif bar_low <= trade["tp"]:
                    trade["result"] = "win"
                    trade["rr"] = trade["rr_target"]
                    trade["exit_price"] = trade["tp"]
                    resolved = True

            if resolved:
                trade["mae"] = round(trade.pop("_mae"), 3)
                trade["mfe"] = round(trade.pop("_mfe"), 3)
                trade["exit_bar"] = i
                trade["exit_time"] = bar_time
                trade["exit_bar_volume"] = bar_volume
                # TP penetration (how far past TP did price go on exit bar)
                if trade["result"] == "win" and trade["tp"] > 0:
                    if trade["zone_type"] == "demand":
                        trade["tp_penetration"] = (bar_high - trade["tp"]) / trade["tp"]
                    else:
                        trade["tp_penetration"] = (trade["tp"] - bar_low) / trade["tp"]
                else:
                    trade["tp_penetration"] = 0.0
                pos = trade["position_size"]
                entry_fee = pos * trade["entry_price"] * FEE_RATE_MAKER  # Entry = limit, no slippage
                if trade["result"] == "win":
                    exit_fee = pos * trade["exit_price"] * FEE_RATE_MAKER  # TP = limit, no slippage
                else:
                    exit_fee = pos * trade["exit_price"] * (FEE_RATE_TAKER + SL_SLIPPAGE_PCT)  # SL = stop market + slippage
                total_fee = entry_fee + exit_fee
                fee_in_r = total_fee / trade["risk_amount"] if trade["risk_amount"] > 0 else 0
                trade["fee"] = round(total_fee, 2)
                trade["fee_r"] = round(fee_in_r, 3)
                trade["rr_gross"] = trade["rr"]
                trade["rr"] = round(trade["rr"] - fee_in_r, 3)
                pnl = trade["risk_amount"] * trade["rr"]
                trade["pnl"] = round(pnl, 2)
                balance += pnl
                trade["balance_after"] = round(balance, 2)
                cumulative_r += trade["rr"]
                closed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # -- 5. Equity tracking --
        peak_r = max(peak_r, cumulative_r)
        peak_balance = max(peak_balance, balance)
        dd_r = peak_r - cumulative_r
        dd_pct = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd_r = max(max_dd_r, dd_r)
        max_dd_pct = max(max_dd_pct, dd_pct)
        equity_curve.append({
            "bar": i, "time": bar_time,
            "cumulative_r": round(cumulative_r, 2),
            "balance": round(balance, 2),
            "drawdown_r": round(dd_r, 2),
            "drawdown_pct": round(dd_pct, 2),
        })

    # Mark remaining open trades
    for trade in open_trades:
        trade["result"] = "open"
        trade["rr"] = 0.0
        trade["pnl"] = 0.0
        trade["fee"] = 0.0
        trade["fee_r"] = 0.0
        trade["rr_gross"] = 0.0
        trade["mae"] = trade.pop("_mae", 0.0)
        trade["mfe"] = trade.pop("_mfe", 0.0)
        trade["balance_after"] = round(balance, 2)
        trade["exit_bar"] = len(df) - 1
        trade["exit_time"] = df.index[-1]
        trade["exit_price"] = round(float(df["close"].iloc[-1]), 1)
        closed_trades.append(trade)

    return closed_trades, equity_curve
