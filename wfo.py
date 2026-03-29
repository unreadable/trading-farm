"""Walk-forward optimization with multiprocessing grid search."""

from __future__ import annotations

import os
import multiprocessing

from config import ATR_PERIOD, BT_ZONE_CLUSTER_PCT, BT_MIN_RR, TIMEFRAME
from zones import detect_zones
from structure import compute_atr
from backtest import walkforward_backtest


# ── Parameter grid (30 combinations) ─────────────────────────────────────────
WFO_PARAM_GRID = [
    {"BT_SL_ATR_MULT": sl, "BT_TP_RR": tp, "BT_MAX_OPEN_TRADES": mt,
     "BT_ZONE_CLUSTER_PCT": BT_ZONE_CLUSTER_PCT, "BT_MIN_RR": BT_MIN_RR,
     }
    for sl in [1.0, 1.2, 1.5]
    for tp in [1.5, 2.0, 2.5, 3.0, 3.5]
    for mt in [3, 4]
]


# ── Multiprocessing shared state (MUST stay in this module for pickle) ────────
_mp_train_slice = None
_mp_train_zones = None
_mp_train_atr = None
_mp_train_liq = None


def _mp_init(train_slice, train_zones, train_atr, train_liq):
    """Initializer for multiprocessing pool -- stores shared data."""
    global _mp_train_slice, _mp_train_zones, _mp_train_atr, _mp_train_liq
    _mp_train_slice = train_slice
    _mp_train_zones = train_zones
    _mp_train_atr = train_atr
    _mp_train_liq = train_liq


def _mp_eval_combo(combo):
    """Worker: run backtest for one param combo, return (score, combo, metrics)."""
    trades, _ = walkforward_backtest(
        _mp_train_slice, _mp_train_zones, _mp_train_atr,
        liq_levels=_mp_train_liq, params=combo,
    )
    score = _score_backtest(trades)
    metrics = _backtest_metrics(trades)
    return (score, combo, metrics)


# ── Scoring & metrics ─────────────────────────────────────────────────────────

def _score_backtest(trades: list[dict]) -> float:
    """Edge score = total_r / sqrt(trades). Higher = better risk-adjusted."""
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    if len(closed) < 5:
        return -999.0
    total_r = sum(t["rr"] for t in closed)
    return total_r / (len(closed) ** 0.5)


def _backtest_metrics(trades: list[dict]) -> dict:
    """Compute summary metrics from a list of trades."""
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    if not closed:
        return {"trades": 0, "winrate": 0, "total_r": 0, "avg_r": 0,
                "pf": 0, "edge_score": 0, "max_dd_r": 0}
    wins = [t for t in closed if t["result"] == "win"]
    total_r = sum(t["rr"] for t in closed)
    gross_win = sum(t["rr"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["rr"] for t in closed if t["result"] == "loss")) or 0.001
    max_dd = 0.0
    cumul_r = 0.0
    peak_r = 0.0
    for t in closed:
        cumul_r += t["rr"]
        if cumul_r > peak_r:
            peak_r = cumul_r
        dd = peak_r - cumul_r
        if dd > max_dd:
            max_dd = dd
    return {
        "trades": len(closed),
        "winrate": round(len(wins) / len(closed) * 100, 1),
        "total_r": round(total_r, 1),
        "avg_r": round(total_r / len(closed), 3),
        "pf": round(gross_win / gross_loss, 2) if gross_loss > 0 else 0,
        "edge_score": round(total_r / (len(closed) ** 0.5), 2),
        "max_dd_r": round(max_dd, 1),
    }


# ── Walk-forward optimization ────────────────────────────────────────────────

def anchored_walkforward_optimization(
    df_1h,
    param_grid: list[dict] | None = None,
    min_train_bars: int = 2000,
    test_window_bars: int = 500,
    step_bars: int = 500,
    parallel: bool = True,
) -> dict:
    """
    Anchored walk-forward optimization.
    Train on expanding window, test on next unseen window, repeat.
    Zone detection runs once on full dataset; backtest runs per param combo.
    """
    if param_grid is None:
        param_grid = WFO_PARAM_GRID

    n = len(df_1h)
    folds = []

    train_end = min_train_bars
    while train_end + test_window_bars <= n:
        test_end = train_end + test_window_bars
        folds.append((train_end, test_end))
        train_end += step_bars

    if not folds:
        print(f"  Not enough data for WFO. Need {min_train_bars + test_window_bars} bars, have {n}.")
        return {"folds": [], "oos_trades": [], "summary": {}}

    print(f"\n  WFO: {len(folds)} folds | train_min={min_train_bars} | "
          f"test={test_window_bars} | step={step_bars}")
    print(f"  Param grid: {len(param_grid)} combinations")
    print(f"  Total backtest runs: {len(folds) * len(param_grid)}", flush=True)

    # Pre-compute zones & ATR once on full dataset
    print("  Detecting zones on full dataset (once)...", end="", flush=True)
    all_zones, all_struct = detect_zones(df_1h, TIMEFRAME, no_lookahead=True)
    full_atr = compute_atr(df_1h, ATR_PERIOD)
    full_liq = all_struct.get("liquidity_levels", [])
    print(f" {len(all_zones)} zones found.", flush=True)

    n_workers = max(1, (os.cpu_count() or 4) - 1) if parallel else 1
    if parallel:
        print(f"  Parallel mode: {n_workers} workers", flush=True)

    all_oos_trades = []
    fold_results = []

    for fi, (tr_end, te_end) in enumerate(folds):
        train_zones = [z for z in all_zones if z["impulse_idx"] < tr_end]
        test_zones = [z for z in all_zones if z["impulse_idx"] < te_end]
        train_liq = [lv for lv in full_liq if lv.get("last_idx", 0) < tr_end]
        test_liq = [lv for lv in full_liq if lv.get("last_idx", 0) < te_end]

        train_slice = df_1h.iloc[:tr_end]
        train_atr = full_atr.iloc[:tr_end]

        best_score = -999.0
        best_params = param_grid[0]
        best_is_metrics = {}

        if parallel and len(param_grid) > 1:
            with multiprocessing.Pool(
                processes=n_workers,
                initializer=_mp_init,
                initargs=(train_slice, train_zones, train_atr, train_liq),
            ) as pool:
                results_list = pool.map(_mp_eval_combo, param_grid)
            for score, combo, metrics in results_list:
                if score > best_score:
                    best_score = score
                    best_params = combo
                    best_is_metrics = metrics
        else:
            for combo in param_grid:
                trades, _ = walkforward_backtest(
                    train_slice, train_zones, train_atr,
                    liq_levels=train_liq, params=combo,
                )
                score = _score_backtest(trades)
                if score > best_score:
                    best_score = score
                    best_params = combo
                    best_is_metrics = _backtest_metrics(trades)

        # Test on unseen window with best params
        full_slice = df_1h.iloc[:te_end]
        test_atr = full_atr.iloc[:te_end]

        test_trades, _ = walkforward_backtest(
            full_slice, test_zones, test_atr,
            liq_levels=test_liq, params=best_params,
        )

        oos_trades = [t for t in test_trades
                      if t["result"] in ("win", "loss") and t.get("entry_bar", 0) >= tr_end]
        oos_metrics = _backtest_metrics(oos_trades)
        all_oos_trades.extend(oos_trades)

        train_start_date = df_1h.index[0].strftime("%Y-%m-%d")
        train_end_date = df_1h.index[tr_end - 1].strftime("%Y-%m-%d")
        test_start_date = df_1h.index[tr_end].strftime("%Y-%m-%d")
        test_end_date = df_1h.index[te_end - 1].strftime("%Y-%m-%d")

        fold_results.append({
            "fold": fi + 1,
            "train_period": f"{train_start_date} -> {train_end_date}",
            "test_period": f"{test_start_date} -> {test_end_date}",
            "best_params": best_params,
            "is_metrics": best_is_metrics,
            "oos_metrics": oos_metrics,
            "oos_trades": oos_trades,
        })

        params_short = (f"SL={best_params['BT_SL_ATR_MULT']} "
                        f"TP={best_params['BT_TP_RR']} "
                        f"MT={best_params['BT_MAX_OPEN_TRADES']}")
        print(f"  Fold {fi+1:>2}/{len(folds)}: "
              f"IS {best_is_metrics.get('total_r', 0):+6.1f}R | "
              f"OOS {oos_metrics.get('total_r', 0):+6.1f}R "
              f"({oos_metrics.get('trades', 0)} trades, "
              f"WR {oos_metrics.get('winrate', 0):.0f}%) | {params_short}",
              flush=True)

    summary = _backtest_metrics(all_oos_trades)

    return {
        "folds": fold_results,
        "oos_trades": all_oos_trades,
        "summary": summary,
    }
