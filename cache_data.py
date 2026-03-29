#!/usr/bin/env python3
"""Download and cache OHLCV data for all portfolio coins.

Saves 30m candles as parquet files in cache/ directory.
Also provides a loader function for fast reads.

Usage:
  python3 cache_data.py                    # Download all coins
  python3 cache_data.py --coins=AVAX,ETH   # Specific coins only
  python3 cache_data.py --check-trades     # Download + quick backtest on AVAX/ETH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import fetch_ohlcv_full
from config import TIMEFRAME

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# End date: 2026-03-26 (locked)
END_DATE = pd.Timestamp("2026-03-26", tz="UTC")

# Portfolio coins with their ccxt symbols and history depth
PORTFOLIO = {
    "AVAX": {"symbol": "AVAX/USD:USD", "days": 1460},
    "UNI":  {"symbol": "UNI/USD:USD",  "days": 1460},
    "LTC":  {"symbol": "LTC/USD:USD",  "days": 1460},
    "LINK": {"symbol": "LINK/USD:USD", "days": 1460},
    "SOL":  {"symbol": "SOL/USD:USD",  "days": 1460},
    "ETH":  {"symbol": "ETH/USD:USD",  "days": 1460},
    "BTC":  {"symbol": "BTC/USD:USD",  "days": 1460},
    "BNB":  {"symbol": "BNB/USD:USD",  "days": 1460},
}


def cache_path(coin: str) -> Path:
    """Return parquet file path for a coin."""
    return CACHE_DIR / f"{coin}_30m.parquet"


def download_coin(coin: str, info: dict) -> pd.DataFrame:
    """Download and cache a single coin's data."""
    print(f"\n  [{coin}] Downloading {info['days']}d of {TIMEFRAME} from Kraken...", flush=True)
    df = fetch_ohlcv_full(info["symbol"], TIMEFRAME, days=info["days"])

    # Trim to end date
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df[df.index <= END_DATE]

    path = cache_path(coin)
    df.to_parquet(path)
    start = df.index[0].strftime("%Y-%m-%d")
    end = df.index[-1].strftime("%Y-%m-%d")
    print(f"  [{coin}] Saved {len(df)} bars ({start} -> {end}) -> {path.name}")
    return df


def load_coin(coin: str) -> pd.DataFrame:
    """Load cached parquet data for a coin."""
    path = cache_path(coin)
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {coin}. Run: python3 cache_data.py --coins={coin}")
    df = pd.read_parquet(path)
    return df


def download_all(coins: list[str] | None = None):
    """Download and cache data for specified coins (or all)."""
    targets = coins or list(PORTFOLIO.keys())
    print(f"{'='*60}")
    print(f"  CACHE DATA — {TIMEFRAME} candles")
    print(f"  End date: {END_DATE.strftime('%Y-%m-%d')}")
    print(f"  Coins: {', '.join(targets)}")
    print(f"{'='*60}")

    for coin in targets:
        if coin not in PORTFOLIO:
            print(f"  [{coin}] Unknown coin, skipping")
            continue
        download_coin(coin, PORTFOLIO[coin])

    print(f"\n  Done! Files saved in {CACHE_DIR}/")


def check_recent_trades(coins: list[str] | None = None):
    """Run backtest on cached data and show recent trade activity."""
    from zones import detect_zones
    from structure import compute_atr
    from backtest import walkforward_backtest
    from config import OPTIMAL_PARAMS, ATR_PERIOD

    targets = coins or ["AVAX", "ETH"]

    print(f"\n{'='*60}")
    print(f"  BACKTEST CHECK — recent trade activity")
    print(f"{'='*60}")

    for coin in targets:
        df = load_coin(coin)
        # Remove tz for backtest compatibility
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        print(f"\n  [{coin}] {len(df)} bars | "
              f"{df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")

        zones, struct = detect_zones(df, TIMEFRAME, no_lookahead=True)
        atr = compute_atr(df, ATR_PERIOD)
        liq = struct.get("liquidity_levels", [])

        print(f"  [{coin}] {len(zones)} zones detected")

        trades, equity = walkforward_backtest(df, zones, atr, liq_levels=liq,
                                               params=OPTIMAL_PARAMS)
        closed = [t for t in trades if t["result"] in ("win", "loss")]

        if not closed:
            print(f"  [{coin}] NO TRADES in entire period!")
            continue

        wins = sum(1 for t in closed if t["result"] == "win")
        total_r = sum(t["rr"] for t in closed)
        pf_w = sum(t["rr"] for t in closed if t["result"] == "win")
        pf_l = abs(sum(t["rr"] for t in closed if t["result"] == "loss"))
        pf = pf_w / pf_l if pf_l > 0 else 0

        print(f"  [{coin}] Total: {len(closed)} trades | {wins}W/{len(closed)-wins}L | "
              f"{total_r:+.1f}R | PF={pf:.2f}")

        # Show last 10 trades with dates
        print(f"\n  [{coin}] Last 10 trades:")
        print(f"  {'Entry Time':<20} {'Type':<8} {'Side':<5} {'Entry':>10} {'Result':>7} {'RR':>7}")
        print(f"  {'-'*62}")
        for t in closed[-10:]:
            entry_t = t.get("entry_time", "?")
            if hasattr(entry_t, "strftime"):
                entry_t = entry_t.strftime("%Y-%m-%d %H:%M")
            print(f"  {str(entry_t):<20} {t.get('zone_type','?'):<8} "
                  f"{t.get('side', '?'):<5} {t.get('entry_price',0):>10.2f} "
                  f"{t['result']:>7} {t['rr']:>+6.2f}R")

        # Check last 7 days specifically
        recent_cutoff = df.index[-1] - pd.Timedelta(days=7)
        recent = [t for t in closed
                  if hasattr(t.get("entry_time"), "__gt__") and t["entry_time"] > recent_cutoff]
        if recent:
            print(f"\n  [{coin}] Trades in last 7 days: {len(recent)}")
            for t in recent:
                entry_t = t["entry_time"].strftime("%Y-%m-%d %H:%M")
                print(f"    {entry_t} | {t.get('zone_type','?')} {t.get('side','?')} "
                      f"@ {t.get('entry_price',0):.2f} -> {t['result']} ({t['rr']:+.2f}R)")
        else:
            print(f"\n  [{coin}] NO trades in last 7 days (before {df.index[-1].strftime('%Y-%m-%d')})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache OHLCV data for portfolio coins")
    parser.add_argument("--coins", type=str, default=None,
                        help="Comma-separated coins (default: all)")
    parser.add_argument("--check-trades", action="store_true",
                        help="Run backtest check on AVAX/ETH after download")
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(",")] if args.coins else None

    download_all(coins)

    if args.check_trades:
        check_coins = coins if coins else ["AVAX", "ETH"]
        check_recent_trades(check_coins)
