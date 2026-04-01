import time
import argparse

from main import load_cached, compute_metrics, print_table, compare_and_save_baseline

from config import ATR_PERIOD, OPTIMAL_PARAMS
from experimental.strategy import detect_zones, compute_atr
from backtest import walkforward_backtest

COINS = ["AVAX", "UNI", "LINK", "LTC", "BNB", "SOL", "ETH", "BTC"]

def run_backtest(args):
    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",")]
    else:
        coins = COINS

    print(f"{'=' * 95}")
    print(f"  STRATEGY BACKTEST — 30m | strategy.py | Cached data")
    print(f"{'=' * 95}")

    results = []

    for coin in coins:
        print(f"{coin}")

        try:
            df = load_cached(coin)
        except FileNotFoundError as e:
            print(str(e))
            continue

        start_time = time.perf_counter()
        zones, struct = detect_zones(df, no_lookahead=True)
        end_time = time.perf_counter()
        print(f"Zones detected in {end_time - start_time:.6f} sec")

        atr = compute_atr(df, ATR_PERIOD)
        liq = struct.get("liquidity_levels", [])

        start_time = time.perf_counter()
        trades, _ = walkforward_backtest(
            df,
            zones,
            atr,
            liq_levels=liq,
            params=OPTIMAL_PARAMS
        )
        end_time = time.perf_counter()
        print(f"Backtest completed in {end_time - start_time:.6f} sec")

        m = compute_metrics(coin, trades, df)

        if m is None:
            print("NO TRADES")
            continue

        print(f"Trades: {m['trades']:,} | Total R: {m['total_r']:+.0f}R | PF={m['pf']:.2f}\n")

        results.append(m)

    # ── Post-processing ─────────────────────────────
    for r in results:
        r["r_dd"] = r["total_r"] / r["max_dd"] if r["max_dd"] > 0 else 0
        r["r_mo"] = r["total_r"] / r["total_months"] if r["total_months"] > 0 else 0

    results.sort(key=lambda r: -r["r_mo"])

    print_table("PORTFOLIO (sorted by R/Mo)", results)

    # ── Baseline ───────────────────────────────────
    compare_and_save_baseline(results)


def main():
    parser = argparse.ArgumentParser(description="Supply & Demand Strategy Backtest")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated coins (default: all)")
    args = parser.parse_args()

    run_backtest(args)


if __name__ == "__main__":
    main()