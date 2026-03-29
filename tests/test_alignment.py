"""Alignment test: verify farm signal logic matches backtest entries.

Tests that given the SAME zones, ATR, and data, the farm's freshness walk +
entry detection produces identical results to the backtest engine.

This directly compares the core logic paths without confounding factors
(different zone detection, lookback windows, etc).

Usage: python -m tests.test_alignment [--coin AVAX] [--sample 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from cache_data import load_coin
from config import (OPTIMAL_PARAMS, TIMEFRAME, ATR_PERIOD,
                    WICK_TOLERANCE_PCT, FEE_RATE_MAKER, FEE_RATE_TAKER,
                    SL_SLIPPAGE_PCT)
from zones import detect_zones
from structure import compute_atr
from backtest import walkforward_backtest


def farm_signals_at_bar(bar_idx, df, zones_sorted, atr_series, params):
    """Replicate farm's compute_signals() logic for a single bar.

    Uses pre-computed zones (same as backtest) to isolate the logic comparison.
    """
    _sl_atr = params.get("BT_SL_ATR_MULT", 1.0)
    _cluster_pct = params.get("BT_ZONE_CLUSTER_PCT", 0.008)

    current_atr = float(atr_series.iloc[bar_idx])
    if current_atr == 0:
        return []

    # Incremental activation + freshness (matching backtest exactly)
    active_zones = []
    zone_ptr = 0
    warmup = max(ATR_PERIOD + 5, 20)

    for i in range(warmup, bar_idx + 1):
        # Activate new zones on this bar
        while zone_ptr < len(zones_sorted) and zones_sorted[zone_ptr]["impulse_idx"] <= i:
            z = dict(zones_sorted[zone_ptr])
            z["_fresh"] = True

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

        # Update freshness
        bar_close = float(df["close"].iloc[i])
        for z in active_zones:
            if not z["_fresh"]:
                continue
            tolerance = z["mid"] * WICK_TOLERANCE_PCT
            if z["type"] == "demand":
                if bar_close < z["bottom"] - tolerance:
                    z["_fresh"] = False
            else:
                if bar_close > z["top"] + tolerance:
                    z["_fresh"] = False

    # Entry check on this bar (same as farm)
    bar_low = float(df["low"].iloc[bar_idx])
    bar_high = float(df["high"].iloc[bar_idx])
    signals = []

    for z in active_zones:
        if not z["_fresh"]:
            continue

        entry_price = z["top"] if z["type"] == "demand" else z["bottom"]

        zone_reached = False
        if z["type"] == "demand" and bar_low <= z["top"]:
            zone_reached = True
        elif z["type"] == "supply" and bar_high >= z["bottom"]:
            zone_reached = True
        if not zone_reached:
            continue

        if z["type"] == "demand":
            sl = z["bottom"] - current_atr * _sl_atr
            risk = entry_price - sl
        else:
            sl = z["top"] + current_atr * _sl_atr
            risk = sl - entry_price

        if risk <= 0:
            continue

        notional = (1.0 / risk) * entry_price
        est_fee_r = notional * (FEE_RATE_MAKER + FEE_RATE_TAKER + SL_SLIPPAGE_PCT)
        if est_fee_r > 0.3:
            continue

        signals.append({
            "entry_price": round(entry_price, 1),
            "zone_type": z["type"],
        })

    return signals


def run_alignment_test(coin: str, sample_size: int = 200):
    """Compare farm logic vs backtest entries bar-by-bar."""
    print(f"\n{'=' * 70}")
    print(f"  ALIGNMENT TEST — {coin}")
    print(f"{'=' * 70}")

    df = load_coin(coin)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    print(f"  {len(df)} bars | {df.index[0].strftime('%Y-%m-%d')} -> "
          f"{df.index[-1].strftime('%Y-%m-%d')}")

    # Run zone detection + backtest (same data for both paths)
    zones, struct = detect_zones(df, TIMEFRAME, no_lookahead=True)
    atr = compute_atr(df, ATR_PERIOD)
    liq = struct.get("liquidity_levels", [])
    zones_sorted = sorted(zones, key=lambda z: z["impulse_idx"])

    bt_trades, _ = walkforward_backtest(df, zones, atr, liq_levels=liq,
                                         params=OPTIMAL_PARAMS)

    bt_entries = [t for t in bt_trades if "entry_bar" in t]
    print(f"  Backtest: {len(bt_entries)} entries")

    bt_by_bar = {}
    for t in bt_entries:
        bt_by_bar.setdefault(t["entry_bar"], []).append(t)

    # Sample entry bars evenly
    all_entry_bars = sorted(bt_by_bar.keys())
    if len(all_entry_bars) <= sample_size:
        test_bars = all_entry_bars
    else:
        indices = np.linspace(0, len(all_entry_bars) - 1, sample_size, dtype=int)
        test_bars = [all_entry_bars[i] for i in indices]

    print(f"  Testing {len(test_bars)} sampled bars...", flush=True)

    matched = 0
    missed = 0
    extra = 0
    mismatches = []

    for idx, bar_idx in enumerate(test_bars):
        if idx % 50 == 0 and idx > 0:
            print(f"    {idx}/{len(test_bars)}...", flush=True)

        # Farm logic path (same zones, same data)
        farm_sigs = farm_signals_at_bar(
            bar_idx, df, zones_sorted, atr, OPTIMAL_PARAMS)

        bt_on_bar = bt_by_bar[bar_idx]

        farm_prices = {s["entry_price"] for s in farm_sigs}
        bt_prices = {round(t["entry_price"], 1) for t in bt_on_bar}

        for bt_t in bt_on_bar:
            bt_ep = round(bt_t["entry_price"], 1)
            if any(abs(bt_ep - fp) / max(abs(bt_ep), 0.01) < 0.005
                   for fp in farm_prices):
                matched += 1
            else:
                missed += 1
                if len(mismatches) < 10:
                    mismatches.append({
                        "bar": bar_idx,
                        "time": df.index[bar_idx],
                        "bt_entry": bt_t["entry_price"],
                        "bt_type": bt_t["zone_type"],
                        "farm_prices": sorted(farm_prices),
                    })

        for fp in farm_prices:
            if not any(abs(fp - bp) / max(abs(fp), 0.01) < 0.005
                       for bp in bt_prices):
                extra += 1

    total = matched + missed
    rate = matched / total * 100 if total > 0 else 0

    print(f"\n  --- Results ---")
    print(f"  Matched:  {matched}/{total} ({rate:.1f}%)")
    print(f"  Missed:   {missed} (backtest entry, no farm signal)")
    print(f"  Extra:    {extra} (farm signal, no backtest entry)")

    if mismatches:
        print(f"\n  --- Sample mismatches ---")
        for m in mismatches[:10]:
            print(f"  bar {m['bar']} ({m['time']}) | BT: {m['bt_type']} "
                  f"@ {m['bt_entry']} | farm: {m['farm_prices']}")
        if missed > 0:
            print(f"\n  Expected: _trade_opened (backtest blocks re-entry on same zone)")
            print(f"  + MAX_OPEN_TRADES (backtest caps at {OPTIMAL_PARAMS.get('BT_MAX_OPEN_TRADES', 4)})")

    if rate >= 95:
        print(f"\n  PASS — {rate:.1f}% alignment")
    elif rate >= 85:
        print(f"\n  WARN — {rate:.1f}% alignment")
    else:
        print(f"\n  FAIL — {rate:.1f}% alignment")

    return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Farm vs backtest alignment test")
    parser.add_argument("--coin", type=str, default="AVAX",
                        help="Coin to test (default: AVAX)")
    parser.add_argument("--all", action="store_true",
                        help="Test all portfolio coins")
    parser.add_argument("--sample", type=int, default=200,
                        help="Bars to sample (default: 200)")
    args = parser.parse_args()

    coins = ["AVAX", "UNI", "LINK", "LTC", "BNB", "SOL", "ETH", "BTC"] if args.all else [args.coin]

    results = {}
    for coin in coins:
        rate = run_alignment_test(coin, sample_size=args.sample)
        results[coin] = rate

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY")
        print(f"{'=' * 70}")
        for coin, rate in results.items():
            status = "PASS" if rate >= 95 else ("WARN" if rate >= 85 else "FAIL")
            print(f"  {coin:<6} {rate:>5.1f}% {status}")
