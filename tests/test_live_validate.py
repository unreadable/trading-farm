"""Validate live farm signals against backtest on post-cache data.

Downloads fresh data from Kraken (cache period + new bars), runs backtest
on the full range, and compares what the backtest would have done after
the cache end date with what the farm actually logged.

Usage:
    python -m tests.test_live_validate              # all portfolio coins
    python -m tests.test_live_validate --coins AVAX # specific coin(s)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from cache_data import PORTFOLIO, END_DATE, load_coin
from config import OPTIMAL_PARAMS, TIMEFRAME, ATR_PERIOD
from data import fetch_ohlcv_full
from zones import detect_zones
from structure import compute_atr
from backtest import walkforward_backtest

FARM_LOG = Path(__file__).parent.parent / "live" / "farm.log"


def parse_farm_log(coins: list[str]) -> dict:
    """Parse farm.log for scan results and signals per coin.

    Returns {coin: [{time, fresh_count, total_zones, price, signals, nearest}]}
    """
    if not FARM_LOG.exists():
        print(f"  No farm log found at {FARM_LOG}")
        return {}

    scans = {c: [] for c in coins}

    # Match both old and new log formats
    # Old: 2026-03-27 15:29:52 [INFO] [AVAX] 10/84 zones fresh | price=8.8270
    # New: 2026-03-27 18:29:57 [INFO] [AVAX] price @ $8.7720 | 10 zones ready | ...
    old_pat = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[(\w+)\] (\d+)/(\d+) zones fresh \| price=([\d.]+)"
    )
    new_pat = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\[(\w+)\] price @ \$?([\d.,]+) \| (\d+) zones ready"
    )
    nearest_pat = re.compile(
        r"nearest.*?(?:fresh|): (\w+) @ \$?([\d.,]+)"
    )

    with open(FARM_LOG) as f:
        for line in f:
            # Old format
            m = old_pat.search(line)
            if m:
                ts, coin, fresh, total, price = m.groups()
                if coin in scans:
                    entry = {
                        "time": ts,
                        "fresh_count": int(fresh),
                        "total_zones": int(total),
                        "price": float(price),
                    }
                    nm = nearest_pat.search(line)
                    if nm:
                        entry["nearest_type"] = nm.group(1)
                        entry["nearest_price"] = float(nm.group(2).replace(",", ""))
                    scans[coin].append(entry)
                continue

            # New format
            m = new_pat.search(line)
            if m:
                ts, coin, price, fresh = m.groups()
                if coin in scans:
                    entry = {
                        "time": ts,
                        "fresh_count": int(fresh),
                        "price": float(price.replace(",", "")),
                    }
                    nm = nearest_pat.search(line)
                    if nm:
                        entry["nearest_type"] = nm.group(1)
                        entry["nearest_price"] = float(nm.group(2).replace(",", ""))
                    scans[coin].append(entry)

    return scans


def run_validation(coins: list[str]):
    """Run live vs backtest validation."""
    print(f"\n{'=' * 70}")
    print(f"  LIVE vs BACKTEST VALIDATION")
    print(f"{'=' * 70}")

    cache_end = END_DATE.tz_localize(None) if END_DATE.tz else END_DATE
    print(f"  Cache ends: {cache_end}")

    # Parse farm logs
    farm_scans = parse_farm_log(coins)
    total_farm_scans = sum(len(v) for v in farm_scans.values())
    print(f"  Farm log: {total_farm_scans} scan entries")

    for coin in coins:
        print(f"\n  {'─' * 66}")
        print(f"  {coin}")
        print(f"  {'─' * 66}")

        # Load cached data as base
        info = PORTFOLIO.get(coin)
        if not info:
            print(f"    Not in PORTFOLIO, skipping")
            continue

        # Load cache + fetch only new bars since END_DATE
        try:
            df_cached = load_coin(coin)
            if df_cached.index.tz is not None:
                df_cached.index = df_cached.index.tz_localize(None)
        except FileNotFoundError:
            print(f"    No cache — skipping (run cache_data.py first)")
            continue

        # Fetch only bars after cache end (~few days, not 4 years)
        days_since = (pd.Timestamp.utcnow().tz_localize(None) - cache_end).days + 2
        print(f"    Fetching {days_since}d of new bars...", flush=True)
        try:
            df_new = fetch_ohlcv_full(info["symbol"], TIMEFRAME,
                                       days=days_since)
        except Exception as e:
            print(f"    Error fetching: {e}")
            continue

        if df_new.index.tz is not None:
            df_new.index = df_new.index.tz_localize(None)

        # Concatenate: cached + new (deduplicate overlap)
        df_fresh = pd.concat([df_cached, df_new])
        df_fresh = df_fresh[~df_fresh.index.duplicated(keep="last")]
        df_fresh.sort_index(inplace=True)

        df_new_only = df_fresh[df_fresh.index > cache_end]

        print(f"    Total: {len(df_fresh)} bars "
              f"({df_fresh.index[0].strftime('%Y-%m-%d')} → "
              f"{df_fresh.index[-1].strftime('%Y-%m-%d')})")
        print(f"    New bars (post-cache): {len(df_new_only)}")

        if len(df_new_only) == 0:
            print(f"    No new data beyond cache — nothing to validate")
            continue

        # Run backtest on FULL data (cached + new)
        zones, struct = detect_zones(df_fresh, TIMEFRAME, no_lookahead=True)
        atr = compute_atr(df_fresh, ATR_PERIOD)
        liq = struct.get("liquidity_levels", [])

        trades, _ = walkforward_backtest(df_fresh, zones, atr,
                                          liq_levels=liq, params=OPTIMAL_PARAMS)

        # Filter to trades that ENTERED after cache end
        # Cache contains data UP TO end date, so new bars start the day after
        new_trades = [t for t in trades
                      if t.get("result") in ("win", "loss", "open")
                      and t.get("entry_time") is not None
                      and pd.Timestamp(t["entry_time"]) > cache_end]

        closed_new = [t for t in new_trades
                      if t.get("result") in ("win", "loss")]

        print(f"    Backtest entries post-cache: {len(new_trades)} "
              f"({len(closed_new)} closed)")

        if new_trades:
            for t in new_trades[:10]:
                etime = pd.Timestamp(t["entry_time"]).strftime("%m-%d %H:%M")
                result = t.get("result", "?")
                rr = t.get("rr", 0)
                print(f"      {etime} | {t['zone_type']:<7} "
                      f"@ {t['entry_price']:<10} "
                      f"→ {result} {rr:+.2f}R" if result != "open"
                      else f"      {etime} | {t['zone_type']:<7} "
                      f"@ {t['entry_price']:<10} → (open)")
            if len(new_trades) > 10:
                print(f"      ... and {len(new_trades) - 10} more")

        # Compare with farm scans
        farm_coin = farm_scans.get(coin, [])
        if farm_coin:
            print(f"\n    Farm scans: {len(farm_coin)} entries")

            # Compare zone counts at similar timestamps
            for scan in farm_coin[:5]:
                scan_time = scan["time"]
                farm_fresh = scan["fresh_count"]
                farm_price = scan["price"]

                # Find the closest bar in backtest data
                try:
                    scan_ts = pd.Timestamp(scan_time)
                    bar_idx = df_fresh.index.get_indexer([scan_ts],
                                                         method="nearest")[0]
                    if bar_idx < 0:
                        continue
                except Exception:
                    continue

                nearest_info = ""
                if "nearest_type" in scan:
                    nearest_info = (f" | nearest: {scan['nearest_type']} "
                                    f"@ {scan['nearest_price']}")

                print(f"      {scan_time} | farm: {farm_fresh} fresh, "
                      f"price={farm_price}{nearest_info}")

        # Summary stats for post-cache backtest
        if closed_new:
            wins = sum(1 for t in closed_new if t["result"] == "win")
            total_r = sum(t["rr"] for t in closed_new)
            print(f"\n    Post-cache backtest: {len(closed_new)} trades, "
                  f"{wins} wins, {total_r:+.1f}R")
        elif new_trades:
            print(f"\n    Post-cache: {len(new_trades)} trades still open")
        else:
            print(f"\n    Post-cache: no backtest entries (consistent with "
                  f"farm seeing 0 signals)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate live farm vs backtest on post-cache data")
    parser.add_argument("--coins", type=str, default=None,
                        help="Coins to validate (default: all portfolio)")
    args = parser.parse_args()

    coins = ([c.strip().upper() for c in args.coins.split(",")]
             if args.coins else list(PORTFOLIO.keys()))
    run_validation(coins)
