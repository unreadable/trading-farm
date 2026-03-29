#!/usr/bin/env python3
"""
Supply & Demand Zone Strategy — CLI

Usage:
  python3 main.py                          # BTC scan + chart (365 days)
  python3 main.py --test                   # Test strategy on full portfolio (cached)
  python3 main.py --test --coins=BTC,ETH   # Test specific coins
  python3 main.py --test --mc              # + Monte Carlo stress test
  python3 main.py --wfo --days=1460        # Walk-forward optimization (BTC)
  python3 main.py --wfo --serial           # WFO without multiprocessing
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress only known noisy warnings (matplotlib, pandas futurewarnings)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="matplotlib")

logging.basicConfig(
    level=logging.WARNING,
    format="  %(levelname)s [%(name)s] %(message)s",
)

CACHE_DIR = Path(__file__).parent / "cache"

# Portfolio coins ordered by R/Mo (best to worst)
PORTFOLIO_COINS = ["AVAX", "UNI", "LINK", "LTC", "BNB", "SOL", "ETH", "BTC"]


def parse_args():
    parser = argparse.ArgumentParser(description="Supply & Demand Zone Strategy")
    parser.add_argument("--test", action="store_true",
                        help="Test strategy on portfolio (uses cached data)")
    parser.add_argument("--wfo", action="store_true",
                        help="Walk-forward optimization mode")
    parser.add_argument("--serial", action="store_true",
                        help="Disable multiprocessing in WFO")
    parser.add_argument("--days", type=int, default=365,
                        help="Days of data to fetch (default: 365)")
    parser.add_argument("--coins", type=str, default=None,
                        help="Comma-separated coins for --test (default: all)")
    parser.add_argument("--exclude-months", type=str, default="",
                        help="Comma-separated months to exclude (e.g. 12 or 11,12)")
    parser.add_argument("--mc", action="store_true",
                        help="Run Monte Carlo stress test after --test")
    parser.add_argument("--mc-sims", type=int, default=10_000,
                        help="Number of MC simulations (default: 10000)")
    parser.add_argument("--limit-analysis", action="store_true",
                        help="Run limit order feasibility analysis")
    return parser.parse_args()


# ── Test mode: strategy on portfolio ─────────────────────────────────────────

def load_cached(coin: str) -> pd.DataFrame:
    """Load cached parquet. Run cache_data.py first."""
    path = CACHE_DIR / f"{coin}_30m.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cache for {coin}. Run: python3 cache_data.py --coins={coin}")
    df = pd.read_parquet(path)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def compute_metrics(coin: str, trades: list[dict], df: pd.DataFrame) -> dict | None:
    """Compute all metrics for one coin."""
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    if not closed:
        return None

    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]

    total_r = sum(t["rr"] for t in closed)
    gross_win = sum(t["rr"] for t in wins)
    gross_loss = abs(sum(t["rr"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    # MaxDD
    cumul = pd.Series([t["rr"] for t in closed]).cumsum()
    max_dd = (cumul.cummax() - cumul).max()

    # Avg RR (average winning trade size)
    avg_rr = np.mean([t["rr"] for t in wins]) if wins else 0

    # Fees
    total_fees_r = sum(t.get("fee_r", 0) for t in closed)

    # Duration (median hours)
    durations = []
    for t in closed:
        et, xt = t.get("entry_time"), t.get("exit_time")
        if hasattr(et, "timestamp") and hasattr(xt, "timestamp"):
            durations.append((xt - et).total_seconds() / 3600)
    median_dur = np.median(durations) if durations else 0

    # Monthly consistency
    df_t = pd.DataFrame(closed)
    df_t["entry_time"] = pd.to_datetime(df_t["entry_time"])
    monthly = df_t.groupby(df_t["entry_time"].dt.to_period("M"))["rr"].sum()
    neg_months = int((monthly < 0).sum())
    total_months = len(monthly)

    # Yearly breakdown
    yearly = df_t.groupby(df_t["entry_time"].dt.year)["rr"].sum().to_dict()

    days = (df.index[-1] - df.index[0]).days

    return {
        "coin": coin, "days": days,
        "trades": len(closed), "wins": len(wins), "losses": len(losses),
        "wr": len(wins) / len(closed) * 100,
        "total_r": total_r, "avg_rr": avg_rr, "pf": pf,
        "max_dd": max_dd, "fees_r": total_fees_r,
        "median_dur_h": median_dur,
        "neg_months": neg_months, "total_months": total_months,
        "yearly": yearly,
    }


def print_table(title: str, results: list[dict]):
    """Print unified metrics table with yearly breakdown."""
    if not results:
        return

    print(f"\n{'=' * 109}")
    print(f"  {title}")
    print(f"{'=' * 109}")
    print(f"  {'Coin':<6} {'Trades':>7} {'WR':>6} {'Total R':>9} {'Avg RR':>7} "
          f"{'PF':>6} {'MaxDD':>7} {'R/DD':>6} {'R/Mo':>6} {'Fees':>7} {'Dur':>6} {'Neg Mo':>7}")
    print(f"  {'-' * 107}")

    for r in results:
        r_dd = r.get("r_dd", r["total_r"] / r["max_dd"] if r["max_dd"] > 0 else 0)
        r_mo = r.get("r_mo", r["total_r"] / r["total_months"] if r["total_months"] > 0 else 0)
        print(f"  {r['coin']:<6} {r['trades']:>7,} {r['wr']:>5.1f}% {r['total_r']:>+8.0f}R "
              f"{r['avg_rr']:>+6.2f} {r['pf']:>5.2f} {r['max_dd']:>6.1f}R "
              f"{r_dd:>5.1f}x {r_mo:>5.1f}R {r['fees_r']:>6.1f}R {r['median_dur_h']:>5.1f}h "
              f"{r['neg_months']:>2}/{r['total_months']:<2}")

    # Aggregate row
    if len(results) > 1:
        tot_trades = sum(r["trades"] for r in results)
        tot_wins = sum(r["wins"] for r in results)
        tot_r = sum(r["total_r"] for r in results)
        tot_win_r = sum(r["avg_rr"] * r["wins"] for r in results)
        tot_losses = sum(r["losses"] for r in results)
        agg_pf = tot_win_r / tot_losses if tot_losses > 0 else 0
        agg_avg_rr = tot_win_r / tot_wins if tot_wins > 0 else 0

        agg_max_dd = max(r['max_dd'] for r in results)
        agg_r_dd = tot_r / agg_max_dd if agg_max_dd > 0 else 0
        tot_months = sum(r['total_months'] for r in results)
        agg_r_mo = tot_r / tot_months if tot_months > 0 else 0
        print(f"  {'-' * 107}")
        print(f"  {'TOTAL':<6} {tot_trades:>7,} {tot_wins / tot_trades * 100:>5.1f}% "
              f"{tot_r:>+8.0f}R {agg_avg_rr:>+6.2f} {agg_pf:>5.2f} "
              f"{agg_max_dd:>6.1f}R "
              f"{agg_r_dd:>5.1f}x {agg_r_mo:>5.1f}R "
              f"{sum(r['fees_r'] for r in results):>6.1f}R "
              f"{np.median([r['median_dur_h'] for r in results]):>5.1f}h "
              f"{sum(r['neg_months'] for r in results):>2}/"
              f"{tot_months:<2}")

    # Yearly breakdown
    all_years = sorted(set(y for r in results for y in r["yearly"]))
    if all_years:
        print(f"\n  {'Coin':<6}", end="")
        for y in all_years:
            print(f" {y:>8}", end="")
        print()
        print(f"  {'-' * (6 + 9 * len(all_years))}")
        for r in results:
            print(f"  {r['coin']:<6}", end="")
            for y in all_years:
                print(f" {r['yearly'].get(y, 0):>+7.0f}R", end="")
            print()
        if len(results) > 1:
            print(f"  {'TOTAL':<6}", end="")
            for y in all_years:
                print(f" {sum(r['yearly'].get(y, 0) for r in results):>+7.0f}R", end="")
            print()


def run_test(args):
    from config import OPTIMAL_PARAMS, TIMEFRAME, ATR_PERIOD
    from zones import detect_zones
    from structure import compute_atr
    from backtest import walkforward_backtest

    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",")]
    else:
        coins = PORTFOLIO_COINS

    print(f"{'=' * 95}")
    print(f"  STRATEGY TEST — {TIMEFRAME} | OPTIMAL_PARAMS | Cached data")
    print(f"{'=' * 95}")

    results = []
    all_rr = []          # for Monte Carlo
    coin_rrs = {}

    for coin in coins:
        print(f"  {coin}...", end=" ", flush=True)
        try:
            df = load_cached(coin)
        except FileNotFoundError as e:
            print(str(e))
            continue

        zones, struct = detect_zones(df, TIMEFRAME, no_lookahead=True)
        atr = compute_atr(df, ATR_PERIOD)
        liq = struct.get("liquidity_levels", [])
        trades, _ = walkforward_backtest(df, zones, atr, liq_levels=liq,
                                          params=OPTIMAL_PARAMS)

        m = compute_metrics(coin, trades, df)
        if m is None:
            print("NO TRADES")
            continue

        print(f"{m['trades']:,} trades, {m['total_r']:+.0f}R, PF={m['pf']:.2f}")

        # Collect R-multiples for MC
        rr_list = [t["rr"] for t in trades if t["result"] in ("win", "loss")]
        coin_rrs[coin] = np.array(rr_list)
        all_rr.extend(rr_list)
        results.append(m)

    # Compute R/DD and R/Mo, sort by R/Mo (time-normalized)
    for r in results:
        r["r_dd"] = r["total_r"] / r["max_dd"] if r["max_dd"] > 0 else 0
        r["r_mo"] = r["total_r"] / r["total_months"] if r["total_months"] > 0 else 0
    results.sort(key=lambda r: -r["r_mo"])

    print_table("PORTFOLIO (sorted by R/Mo)", results)

    # ── Baseline comparison ──────────────────────────────────────────────
    compare_and_save_baseline(results)

    if args.mc and all_rr:
        run_monte_carlo(np.array(all_rr), coin_rrs, n_sims=args.mc_sims)


# ── Baseline comparison ──────────────────────────────────────────────────────

BASELINE_PATH = Path(__file__).parent / "output" / "baseline.json"


def _snapshot(results: list[dict]) -> dict:
    """Extract deterministic metrics for comparison."""
    coins = {}
    for r in results:
        coins[r["coin"]] = {
            "trades": r["trades"],
            "total_r": round(r["total_r"], 1),
            "pf": round(r["pf"], 2),
            "wr": round(r["wr"], 1),
            "max_dd": round(r["max_dd"], 1),
        }
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "coins": coins,
        "total_trades": sum(r["trades"] for r in results),
        "total_r": round(sum(r["total_r"] for r in results), 1),
    }


def compare_and_save_baseline(results: list[dict]):
    """Compare current run against saved baseline, then save new baseline."""
    current = _snapshot(results)

    if BASELINE_PATH.exists():
        prev = json.loads(BASELINE_PATH.read_text())
        prev_coins = prev.get("coins", {})

        print(f"\n  {'─' * 80}")
        print(f"  BASELINE COMPARISON  (vs {prev.get('timestamp', '?')})")
        print(f"  {'─' * 80}")
        print(f"  {'Coin':<6} {'Trades':>9} {'Total R':>12} {'PF':>10} {'WR':>10} {'MaxDD':>10}")
        print(f"  {'─' * 80}")

        any_diff = False
        for coin, cur in current["coins"].items():
            old = prev_coins.get(coin)
            if not old:
                print(f"  {coin:<6}  NEW")
                any_diff = True
                continue

            diffs = []
            for key, label, fmt, invert in [
                ("trades", "Trades", "{:+d}", False),
                ("total_r", "Total R", "{:+.1f}", False),
                ("pf", "PF", "{:+.2f}", False),
                ("wr", "WR", "{:+.1f}", False),
                ("max_dd", "MaxDD", "{:+.1f}", True),
            ]:
                delta = cur[key] - old[key]
                if abs(delta) < 0.05:
                    diffs.append(f"{'=':>9}")
                else:
                    sign_bad = delta > 0 if invert else delta < 0
                    marker = " !!!" if sign_bad and abs(delta) > abs(old[key]) * 0.02 else ""
                    diffs.append(f"{fmt.format(delta):>9}{marker}")
                    any_diff = True

            print(f"  {coin:<6} {diffs[0]:>9} {diffs[1]:>12} {diffs[2]:>10} {diffs[3]:>10} {diffs[4]:>10}")

        # Missing coins (in previous but not current)
        for coin in prev_coins:
            if coin not in current["coins"]:
                print(f"  {coin:<6}  MISSING (was in previous baseline)")
                any_diff = True

        # Totals
        d_trades = current["total_trades"] - prev.get("total_trades", 0)
        d_r = current["total_r"] - prev.get("total_r", 0)
        print(f"  {'─' * 80}")
        if abs(d_trades) < 1 and abs(d_r) < 0.5:
            print(f"  IDENTICAL — no changes detected")
        elif not any_diff:
            print(f"  IDENTICAL")
        else:
            print(f"  TOTAL: trades {d_trades:+d}, R {d_r:+.1f}")
            if d_r < -50:
                print(f"  ⚠ REGRESSION — Total R dropped significantly!")
    else:
        print(f"\n  First run — saving baseline to {BASELINE_PATH.name}")

    # Save current as new baseline
    BASELINE_PATH.write_text(json.dumps(current, indent=2) + "\n")
    print(f"  Baseline saved → {BASELINE_PATH.name}")


# ── Monte Carlo stress test ──────────────────────────────────────────────────

def _mc_batch(args):
    """Run a batch of MC sims on a numpy array of R-multiples."""
    rr_array, start_seed, n_sims = args
    results = []
    for i in range(n_sims):
        rng = np.random.RandomState(start_seed + i)
        shuffled = rng.permutation(rr_array)
        cumul_r = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cumul_r)
        dd = peak - cumul_r
        max_dd = dd.max()

        max_streak = 0
        streak = 0
        for r in shuffled:
            if r < 0:
                streak += 1
                if streak > max_streak:
                    max_streak = streak
            else:
                streak = 0

        # Worst 500-trade chunk
        chunk_size = 500
        if len(shuffled) >= chunk_size:
            cs = np.cumsum(shuffled)
            chunk_sums = cs[chunk_size:] - cs[:-chunk_size]
            worst_chunk = chunk_sums.min()
        else:
            worst_chunk = cumul_r[-1]

        results.append((max_dd, max_streak, worst_chunk))
    return results


def _mc_coin_batch(args):
    """Run MC sims for a single coin (DD only)."""
    coin_rr, n_sims, base_seed = args
    results = []
    for i in range(n_sims):
        rng = np.random.RandomState(base_seed + i)
        shuffled = rng.permutation(coin_rr)
        cumul_r = np.cumsum(shuffled)
        peak = np.maximum.accumulate(cumul_r)
        dd = peak - cumul_r
        results.append(dd.max())
    return results


def run_monte_carlo(rr_array, coin_rrs, n_sims=10_000):
    """Run Monte Carlo stress test and print results."""
    from multiprocessing import Pool, cpu_count

    n_trades = len(rr_array)
    n_workers = max(1, cpu_count() - 1)

    print(f"\n{'=' * 70}")
    print(f"  MONTE CARLO STRESS TEST")
    print(f"  {n_trades:,} trades, total R: {rr_array.sum():+,.1f}R")
    print(f"  {n_sims:,} sims on {n_workers} cores")
    print(f"{'=' * 70}")

    batch_size = n_sims // n_workers
    batches = []
    for w in range(n_workers):
        start = w * batch_size
        n = batch_size if w < n_workers - 1 else n_sims - start
        batches.append((rr_array, start, n))

    with Pool(n_workers) as pool:
        batch_results = pool.map(_mc_batch, batches)

    all_results = []
    for br in batch_results:
        all_results.extend(br)

    df_mc = pd.DataFrame(all_results, columns=["max_dd", "max_loss_streak", "worst_chunk"])

    # Max Drawdown
    print(f"\n  --- Max Drawdown Distribution ---")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p:02d}: {df_mc['max_dd'].quantile(p / 100):>7.1f}R")
    print(f"  Worst: {df_mc['max_dd'].max():>7.1f}R")

    # Consecutive Losses
    print(f"\n  --- Max Consecutive Losses ---")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p:02d}: {df_mc['max_loss_streak'].quantile(p / 100):>5.0f}")
    print(f"  Worst: {df_mc['max_loss_streak'].max():>5.0f}")

    # Worst chunk
    print(f"\n  --- Worst 500-Trade Chunk ---")
    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p:02d}: {df_mc['worst_chunk'].quantile(p / 100):>+8.1f}R")

    # Risk of Ruin table
    p95_dd = df_mc["max_dd"].quantile(0.95)
    p99_dd = df_mc["max_dd"].quantile(0.99)

    print(f"\n  --- Position Sizing (P95 DD={p95_dd:.1f}R, P99 DD={p99_dd:.1f}R) ---")
    print(f"  {'Max Capital Loss':<18} {'Risk/Trade (P95)':>18} {'Risk/Trade (P99)':>18}")
    print(f"  {'-' * 56}")
    for target in [10, 15, 20, 25, 30]:
        print(f"  {target:>3}% of capital    {target / p95_dd:>17.2f}%  {target / p99_dd:>17.2f}%")

    # Per-coin MC
    if len(coin_rrs) > 1:
        print(f"\n  --- Per-Coin MaxDD (MC P50 / P90 / P95 / P99) ---")
        print(f"  {'Coin':<6} {'Trades':>7} {'Total R':>9} {'Actual':>8} "
              f"{'P50':>7} {'P90':>7} {'P95':>7} {'P99':>7}")
        print(f"  {'-' * 65}")

        for coin, coin_rr in coin_rrs.items():
            actual_dd = (np.maximum.accumulate(np.cumsum(coin_rr)) - np.cumsum(coin_rr)).max()
            with Pool(n_workers) as pool:
                dds = np.array(pool.map(_mc_coin_batch, [(coin_rr, 1000, 0)])[0])
            print(f"  {coin:<6} {len(coin_rr):>7,} {coin_rr.sum():>+8.0f}R {actual_dd:>7.1f}R "
                  f"{np.percentile(dds, 50):>6.1f}R {np.percentile(dds, 90):>6.1f}R "
                  f"{np.percentile(dds, 95):>6.1f}R {np.percentile(dds, 99):>6.1f}R")


# ── WFO mode ─────────────────────────────────────────────────────────────────

def run_wfo(args):
    from config import SYMBOL, TIMEFRAME, OUTPUT_DIR
    from data import fetch_ohlcv_full
    from wfo import WFO_PARAM_GRID, anchored_walkforward_optimization
    from reporting import print_oos_summary, export_trade_journal

    print("=" * 70)
    print("  WALK-FORWARD OPTIMIZATION MODE")
    print(f"  Fetching {args.days} days of {TIMEFRAME} data...")
    print("=" * 70)

    exclude_months = []
    if args.exclude_months:
        exclude_months = [int(x) for x in args.exclude_months.split(",")]

    df_full = fetch_ohlcv_full(SYMBOL, TIMEFRAME, days=args.days)
    print(f"  Data: {len(df_full)} bars | "
          f"{df_full.index[0].strftime('%Y-%m-%d')} -> "
          f"{df_full.index[-1].strftime('%Y-%m-%d')}")
    if exclude_months:
        from calendar import month_abbr
        print(f"  Excluding months: {', '.join(month_abbr[m] for m in exclude_months)}")

    param_grid = WFO_PARAM_GRID
    if exclude_months:
        param_grid = [{**c, "EXCLUDE_MONTHS": exclude_months} for c in WFO_PARAM_GRID]

    use_parallel = not args.serial
    results = anchored_walkforward_optimization(df_full, param_grid, parallel=use_parallel)
    print_oos_summary(results)

    oos_path = str(OUTPUT_DIR / "oos_journal.csv")
    export_trade_journal(results["oos_trades"], oos_path)

    fold_rows = []
    for f in results["folds"]:
        row = {
            "fold": f["fold"],
            "train_period": f["train_period"],
            "test_period": f["test_period"],
            **{f"is_{k}": v for k, v in f["is_metrics"].items()},
            **{f"oos_{k}": v for k, v in f["oos_metrics"].items()},
            **{k: v for k, v in f["best_params"].items()},
        }
        fold_rows.append(row)
    fold_path = str(OUTPUT_DIR / "wfo_folds.csv")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    print(f"\n   Fold summary -> {fold_path}")


# ── Normal mode: BTC scan + backtest ─────────────────────────────────────────

def run_normal(args):
    from config import (SYMBOL, TIMEFRAME, ATR_PERIOD,
                        BT_SL_ATR_MULT, BT_TP_RR, BT_MAX_OPEN_TRADES,
                        FEE_RATE_MAKER, FEE_RATE_TAKER, SL_SLIPPAGE_PCT, OUTPUT_DIR)
    from data import fetch_ohlcv_full
    from zones import detect_zones
    from structure import compute_atr
    from backtest import walkforward_backtest
    from reporting import (print_backtest_report, export_trade_journal,
                           plot_equity_curve, plot_zones, print_limit_order_analysis)

    print("=" * 70)
    print("  BTC/USD Supply & Demand Zone Scanner -- Production")
    print(f"  Kraken Futures | {TIMEFRAME}")
    print("  Structure + Liquidity + Walk-Forward Backtest (fee-adjusted)")
    print("=" * 70)

    # 1. Fetch
    print(f"\n  Fetching {args.days}d of {TIMEFRAME}...", end="")
    df = fetch_ohlcv_full(SYMBOL, TIMEFRAME, days=args.days)
    print(f" {len(df)} candles | "
          f"{df.index[0].strftime('%Y-%m-%d %H:%M')} -> "
          f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}")

    # 2. Detect zones (with lookahead for display)
    zones_display, structure = detect_zones(df, TIMEFRAME)
    fresh = sum(1 for z in zones_display if z["fresh"])
    liq = sum(1 for z in zones_display if z.get("liq_swept"))
    print(f"\n  {len(zones_display)} zones | {fresh} fresh | {liq} liq-swept")
    for z in zones_display[:5]:
        tag = "F" if z["fresh"] else ("M" if z.get("mitigated") else "T")
        tr = "+" if z.get("with_trend") else ("-" if z.get("trend") != "ranging" and not z.get("with_trend") else "~")
        bos = "B" if z["has_bos"] else " "
        lq = "L" if z.get("liq_swept") else " "
        print(f"   [{tag}{tr}{bos}{lq}] {z['type'].upper():>6} | "
              f"${z['bottom']:>9,.0f} - ${z['top']:>9,.0f} "
              f"| str:{z['strength']:>5.2f} | "
              f"vol_z:{z['vol_zscore']:>5.1f} | "
              f"base:{z['base_candles']}c | "
              f"trend:{z['trend']}")

    # 3. Nearest zones
    current_price = df["close"].iloc[-1]
    supply_z = [z for z in zones_display if z["type"] == "supply" and z["fresh"]]
    demand_z = [z for z in zones_display if z["type"] == "demand" and z["fresh"]]

    print(f"\n{'_' * 70}")
    print(f"  Price: ${current_price:,.1f}")
    below_demand = [z for z in demand_z if z["top"] <= current_price * 1.005]
    above_supply = [z for z in supply_z if z["bottom"] >= current_price * 0.995]

    if below_demand:
        nd = max(below_demand, key=lambda z: z["top"])
        d = current_price - nd["top"]
        print(f"  Nearest DEMAND: ${nd['bottom']:,.0f}-${nd['top']:,.0f}  "
              f"(${d:,.0f} below)  str:{nd['strength']:.1f}")
    if above_supply:
        ns = min(above_supply, key=lambda z: z["bottom"])
        d = ns["bottom"] - current_price
        print(f"  Nearest SUPPLY: ${ns['bottom']:,.0f}-${ns['top']:,.0f}  "
              f"(${d:,.0f} above)  str:{ns['strength']:.1f}")

    # 4. Walk-forward backtest
    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD BACKTEST")
    print(f"  SL={BT_SL_ATR_MULT}x | TP={BT_TP_RR}x | "
          f"MaxTrades={BT_MAX_OPEN_TRADES} | Maker={FEE_RATE_MAKER*100:.2f}% | Taker+Slip={FEE_RATE_TAKER*100:.2f}%+{SL_SLIPPAGE_PCT*100:.2f}%")
    print(f"{'=' * 70}")

    atr = compute_atr(df, ATR_PERIOD)
    bt_zones, bt_structure = detect_zones(df, TIMEFRAME, no_lookahead=True)
    bt_liq = bt_structure.get("liquidity_levels", [])

    trades, equity = walkforward_backtest(df, bt_zones, atr, liq_levels=bt_liq)

    print_backtest_report(trades, equity)

    if args.limit_analysis:
        print_limit_order_analysis(trades)

    # 5. Export
    journal_path = str(OUTPUT_DIR / "trade_journal.csv")
    export_trade_journal(trades, journal_path)

    equity_path = str(OUTPUT_DIR / "equity_curve.png")
    plot_equity_curve(equity, equity_path)

    # 6. Chart
    print(f"\n{'=' * 70}")
    print("  Generating chart...")
    plot_zones(df, zones_display, structure, TIMEFRAME)


def main():
    args = parse_args()

    if args.wfo:
        run_wfo(args)
    elif args.test:
        run_test(args)
    else:
        run_normal(args)


if __name__ == "__main__":
    main()
