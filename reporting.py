"""Reports, charts, and CSV export."""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
from collections import defaultdict

from config import (
    INITIAL_BALANCE, RISK_PER_TRADE_PCT, FEE_RATE_MAKER, FEE_RATE_TAKER, SL_SLIPPAGE_PCT,
    BT_MAX_OPEN_TRADES, BT_ZONE_CLUSTER_PCT, OUTPUT_DIR,
)


def export_trade_journal(trades: list[dict], path: str):
    """Export trade journal to CSV."""
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.get("entry_time", ""),
            "exit_time": t.get("exit_time", ""),
            "zone_type": t.get("zone_type", ""),
            "trade_score": t.get("trade_score", 0),
            "session": t.get("session", ""),
            "entry_price": t.get("entry_price", 0),
            "sl": t.get("sl", 0),
            "tp": t.get("tp", 0),
            "exit_price": t.get("exit_price", 0),
            "risk": t.get("risk", 0),
            "risk_amount": t.get("risk_amount", 0),
            "result": t.get("result", ""),
            "rr": t.get("rr", 0),
            "rr_target": t.get("rr_target", 0),
            "pnl": t.get("pnl", 0),
            "balance_after": t.get("balance_after", 0),
            "with_trend": t.get("with_trend", False),
            "has_bos": t.get("has_bos", False),
            "liq_swept": t.get("liq_swept", False),
            "timeframe": t.get("timeframe", ""),
            "regime": t.get("regime", ""),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\n   Trade journal -> {path}")


def plot_equity_curve(equity_curve: list[dict], path: str):
    """Save equity curve chart with balance + drawdown."""
    if not equity_curve:
        return

    times = [e["time"] for e in equity_curve]
    balances = [e["balance"] for e in equity_curve]
    dd_pct = [e["drawdown_pct"] for e in equity_curve]
    cum_r = [e["cumulative_r"] for e in equity_curve]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.patch.set_facecolor("#1a1a1a")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.plot(times, balances, color="#26a69a", linewidth=1.2)
    ax1.axhline(INITIAL_BALANCE, color="#555555", linewidth=0.5, linestyle="--")
    ax1.fill_between(times, balances, INITIAL_BALANCE,
                      where=[b >= INITIAL_BALANCE for b in balances], alpha=0.1, color="#26a69a")
    ax1.fill_between(times, balances, INITIAL_BALANCE,
                      where=[b < INITIAL_BALANCE for b in balances], alpha=0.1, color="#ef5350")
    ax1.set_ylabel("Balance ($)", color="#888888")
    ax1.set_title(f"Equity Curve | Start: {INITIAL_BALANCE:,.0f} | "
                  f"Risk: {RISK_PER_TRADE_PCT*100:.0f}% per trade | "
                  f"End: {balances[-1]:,.0f}",
                  color="#cccccc", fontsize=11)

    ax2.plot(times, cum_r, color="#4fc3f7", linewidth=1.0)
    ax2.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Cumulative R", color="#888888")

    ax3.fill_between(times, dd_pct, 0, alpha=0.4, color="#ef5350")
    ax3.set_ylabel("DD %", color="#888888")
    ax3.invert_yaxis()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close(fig)
    print(f"   Equity curve -> {path}")


def _print_monthly_breakdown(closed: list[dict], label: str = "BY MONTH"):
    """Print monthly performance breakdown from closed trades."""
    from calendar import month_abbr
    monthly = defaultdict(list)
    for t in closed:
        entry = t.get("entry_time")
        if entry is not None:
            monthly[entry.month].append(t)
    if not monthly:
        return
    print(f"\n  {'_' * 50}")
    print(f"  {label}:")
    print(f"    {'Mon':<5} | {'Trades':>6} | {'WR':>6} | {'Avg R':>7} | {'Total R':>8}")
    print(f"    {'_'*5}_|_{'_'*6}_|_{'_'*6}_|_{'_'*7}_|_{'_'*8}")
    worst_month = None
    worst_r = float('inf')
    for m in range(1, 13):
        trades_m = monthly.get(m, [])
        if not trades_m:
            continue
        wins = sum(1 for t in trades_m if t["result"] == "win")
        wr = wins / len(trades_m) * 100
        total_r = sum(t["rr"] for t in trades_m)
        avg_r = total_r / len(trades_m)
        if total_r < worst_r:
            worst_r = total_r
            worst_month = m
        print(f"    {month_abbr[m]:<5} | {len(trades_m):>6} | {wr:>5.0f}% | "
              f"{avg_r:>+6.3f} | {total_r:>+7.1f}R")
    if worst_month is not None:
        print(f"    ** Weakest: {month_abbr[worst_month]} ({worst_r:+.1f}R)")


def print_limit_order_analysis(trades: list[dict]):
    """Analyze limit order feasibility: penetration buffers, volume fill ratios, fee savings."""
    closed = [t for t in trades if t["result"] in ("win", "loss")]
    if not closed:
        print("\n  No trades for limit order analysis.")
        return

    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]
    total = len(closed)

    print(f"\n{'=' * 70}")
    print(f"  LIMIT ORDER FEASIBILITY ANALYSIS")
    print(f"{'=' * 70}")

    # --- 1. Entry penetration analysis ---
    entry_pens = [t.get("entry_penetration", 0) for t in closed]
    print(f"\n  ENTRY PENETRATION (how far past limit level price went on entry bar)")
    print(f"  Total trades: {total}")
    print(f"  {'Metric':<20} | {'Value':>10}")
    print(f"  {'_'*20}_|_{'_'*10}")
    for label, fn in [("Min", np.min), ("P10", lambda x: np.percentile(x, 10)),
                      ("P25 (Q1)", lambda x: np.percentile(x, 25)),
                      ("Median", np.median), ("P75 (Q3)", lambda x: np.percentile(x, 75)),
                      ("Mean", np.mean), ("Max", np.max)]:
        print(f"  {label:<20} | {fn(entry_pens)*100:>9.3f}%")

    # --- 2. TP penetration analysis (wins only) ---
    tp_pens = [t.get("tp_penetration", 0) for t in wins]
    if tp_pens:
        print(f"\n  TP PENETRATION (how far past TP price went on exit bar — wins only)")
        print(f"  Winning trades: {len(wins)}")
        print(f"  {'Metric':<20} | {'Value':>10}")
        print(f"  {'_'*20}_|_{'_'*10}")
        for label, fn in [("Min", np.min), ("P10", lambda x: np.percentile(x, 10)),
                          ("P25 (Q1)", lambda x: np.percentile(x, 25)),
                          ("Median", np.median), ("P75 (Q3)", lambda x: np.percentile(x, 75)),
                          ("Mean", np.mean), ("Max", np.max)]:
            print(f"  {label:<20} | {fn(tp_pens)*100:>9.3f}%")

    # --- 3. Buffer impact: how many trades lost at each buffer ---
    buffers = [0.0, 0.0001, 0.0002, 0.0005, 0.001]
    print(f"\n  FILL PROBABILITY AT DIFFERENT PENETRATION BUFFERS")
    print(f"  (trade only fills if price goes PAST level by >= buffer)")
    print(f"  {'Buffer':<10} | {'Entry fill':>11} | {'TP fill':>11} | "
          f"{'Both fill':>11} | {'Lost trades':>12} | {'Lost R':>10}")
    print(f"  {'_'*10}_|_{'_'*11}_|_{'_'*11}_|_{'_'*11}_|_{'_'*12}_|_{'_'*10}")

    # Baseline total R for comparison
    baseline_r = sum(t["rr"] for t in closed)

    for buf in buffers:
        entry_fills = sum(1 for t in closed if t.get("entry_penetration", 0) >= buf)
        tp_fills = sum(1 for t in wins if t.get("tp_penetration", 0) >= buf)
        # Trades that wouldn't enter at all
        no_entry = [t for t in closed if t.get("entry_penetration", 0) < buf]
        # Wins that enter but TP doesn't fill (becomes uncertain — conservative: count as loss)
        uncertain_wins = [t for t in wins
                          if t.get("entry_penetration", 0) >= buf
                          and t.get("tp_penetration", 0) < buf]

        # Lost R from no-entry trades (we lose their contribution, good or bad)
        lost_r_no_entry = sum(t["rr"] for t in no_entry)
        # Lost R from uncertain wins (worst case: they become -1R losses instead of wins)
        # Net change per trade = current rr - (-1) = rr + 1 (fee already baked into rr)
        lost_r_uncertain = sum(t["rr"] + 1.0 for t in uncertain_wins)

        both_ok = entry_fills - len(uncertain_wins)
        total_lost = len(no_entry) + len(uncertain_wins)
        total_lost_r = lost_r_uncertain - lost_r_no_entry  # positive = net R lost

        print(f"  {buf*100:>8.2f}% | {entry_fills:>5}/{total:<5} | "
              f"{tp_fills:>5}/{len(wins):<5} | {both_ok:>5}/{total:<5} | "
              f"{total_lost:>6} ({total_lost/total*100:>4.1f}%) | {total_lost_r:>+9.1f}R")

    # --- 4. Volume fill ratio analysis ---
    print(f"\n  VOLUME FILL RATIO (position size vs bar volume)")
    print(f"  (what % of the bar's volume your order would consume)")

    entry_ratios = []
    exit_ratios = []
    for t in closed:
        notional = t.get("position_notional", 0)
        entry_vol = t.get("entry_bar_volume", 0)
        exit_vol = t.get("exit_bar_volume", 0)
        entry_price = t.get("entry_price", 0)
        exit_price = t.get("exit_price", 0)

        if entry_vol > 0 and entry_price > 0:
            entry_vol_usd = entry_vol * entry_price
            entry_ratios.append(notional / entry_vol_usd)
        if exit_vol > 0 and exit_price > 0:
            exit_vol_usd = exit_vol * exit_price
            exit_ratios.append(notional / exit_vol_usd)

    if entry_ratios:
        print(f"\n  Entry fill ratios ({len(entry_ratios)} trades):")
        print(f"  {'Metric':<20} | {'Value':>10}")
        print(f"  {'_'*20}_|_{'_'*10}")
        for label, fn in [("Median", np.median), ("P90", lambda x: np.percentile(x, 90)),
                          ("P99", lambda x: np.percentile(x, 99)),
                          ("Max", np.max)]:
            print(f"  {label:<20} | {fn(entry_ratios)*100:>9.4f}%")

        thresholds = [0.01, 0.05, 0.10]
        for thr in thresholds:
            n = sum(1 for r in entry_ratios if r > thr)
            print(f"  Entries > {thr*100:.0f}% of bar vol: {n} ({n/len(entry_ratios)*100:.1f}%)")

    if exit_ratios:
        print(f"\n  Exit fill ratios ({len(exit_ratios)} trades):")
        print(f"  {'Metric':<20} | {'Value':>10}")
        print(f"  {'_'*20}_|_{'_'*10}")
        for label, fn in [("Median", np.median), ("P90", lambda x: np.percentile(x, 90)),
                          ("P99", lambda x: np.percentile(x, 99)),
                          ("Max", np.max)]:
            print(f"  {label:<20} | {fn(exit_ratios)*100:>9.4f}%")

    # --- 5. Fee summary ---
    total_fees_r = sum(t.get("fee_r", 0) for t in closed)
    print(f"\n  FEE MODEL (maker entry/TP={FEE_RATE_MAKER*100:.2f}%, taker SL={FEE_RATE_TAKER*100:.2f}%+{SL_SLIPPAGE_PCT*100:.2f}% slip)")
    print(f"  {'_'*50}")
    print(f"  Total fees paid:  {total_fees_r:>8.1f}R  ({total_fees_r/total:.3f}R/trade)")
    print(f"  Total R (net):    {baseline_r:>+8.1f}R")


def print_backtest_report(trades: list[dict], equity_curve: list[dict] | None = None):
    """Print detailed walk-forward backtest statistics."""
    if not trades:
        print("\n  No trades to backtest.")
        return

    closed = [t for t in trades if t["result"] in ("win", "loss")]
    if not closed:
        print(f"\n  {len(trades)} zones identified, none resolved yet (all open).")
        return

    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]
    opens = [t for t in trades if t["result"] == "open"]

    total = len(closed)
    winrate = len(wins) / total * 100
    total_rr = sum(t["rr"] for t in closed)
    avg_rr = total_rr / total

    avg_win_rr = np.mean([t["rr"] for t in wins]) if wins else 0
    avg_loss_rr = abs(np.mean([t["rr"] for t in losses])) if losses else 0
    expectancy = (winrate / 100 * avg_win_rr) - ((100 - winrate) / 100 * avg_loss_rr)

    max_consec_loss = 0
    curr_consec = 0
    for t in closed:
        if t["result"] == "loss":
            curr_consec += 1
            max_consec_loss = max(max_consec_loss, curr_consec)
        else:
            curr_consec = 0

    final_balance = closed[-1].get("balance_after", INITIAL_BALANCE) if closed else INITIAL_BALANCE
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    ret_pct = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    max_dd = 0.0
    max_dd_pct_val = 0.0
    if equity_curve:
        max_dd = max(e.get("drawdown_r", e.get("drawdown", 0)) for e in equity_curve)
        max_dd_pct_val = max(e.get("drawdown_pct", 0) for e in equity_curve)

    print(f"\n{'=' * 70}")
    print(f"  WALK-FORWARD BACKTEST  (no lookahead bias)")
    print(f"  Max {BT_MAX_OPEN_TRADES} trades | Zone cluster {BT_ZONE_CLUSTER_PCT*100:.1f}%")
    print(f"{'=' * 70}")
    print(f"\n  Total trades: {total} closed + {len(opens)} open")
    print(f"  Wins: {len(wins)}  |  Losses: {len(losses)}  |  Winrate: {winrate:.1f}%")

    total_fees_r = sum(t.get("fee_r", 0) for t in closed)
    total_fees_usd = sum(t.get("fee", 0) for t in closed)
    avg_fee_r = total_fees_r / total if total else 0

    print(f"  Total R (net): {total_rr:+.1f}R  |  Avg R/trade: {avg_rr:+.2f}R")
    print(f"  Avg win: {avg_win_rr:+.2f}R  |  Avg loss: {-avg_loss_rr:+.2f}R")
    print(f"  Expectancy: {expectancy:+.3f}R per trade")
    print(f"  Fees paid: {total_fees_r:.1f}R (${total_fees_usd:,.0f}) | Avg: {avg_fee_r:.3f}R/trade")
    print(f"  Max DD: {max_dd:.1f}R ({max_dd_pct_val:.1f}%)  |  Max consec losses: {max_consec_loss}")

    print(f"\n  {'_' * 50}")
    print(f"  EQUITY SIMULATION ({RISK_PER_TRADE_PCT*100:.0f}% risk/trade, maker={FEE_RATE_MAKER*100:.2f}%, SL taker+slip={FEE_RATE_TAKER*100:.2f}%+{SL_SLIPPAGE_PCT*100:.2f}%):")
    print(f"    Start: ${INITIAL_BALANCE:,.0f}  ->  End: ${final_balance:,.0f}")
    print(f"    Return: {ret_pct:+.1f}%  |  P&L: ${total_pnl:+,.0f}  |  Fees: ${total_fees_usd:,.0f}")

    # By feature
    print(f"\n  {'_' * 50}")
    print(f"  BY FEATURE:")
    features = [
        ("with_trend", "With trend"),
        ("has_bos", "BOS confirmed"),
        ("liq_swept", "Liquidity swept"),
    ]
    for key, label in features:
        yes = [t for t in closed if t.get(key)]
        no = [t for t in closed if not t.get(key)]
        if yes:
            y_wr = sum(1 for t in yes if t["result"] == "win") / len(yes) * 100
            y_rr = sum(t["rr"] for t in yes) / len(yes)
            print(f"    {label:>20} YES: {len(yes)} trades | WR: {y_wr:.0f}% | Avg R: {y_rr:+.2f}")
        if no:
            n_wr = sum(1 for t in no if t["result"] == "win") / len(no) * 100
            n_rr = sum(t["rr"] for t in no) / len(no)
            print(f"    {label:>20}  NO: {len(no)} trades | WR: {n_wr:.0f}% | Avg R: {n_rr:+.2f}")

    # By session
    print(f"\n  {'_' * 50}")
    print(f"  BY SESSION:")
    for sess in ["Asia", "London", "New York"]:
        st = [t for t in closed if t.get("session") == sess]
        if not st:
            continue
        sw = sum(1 for t in st if t["result"] == "win")
        swr = sw / len(st) * 100
        srr = sum(t["rr"] for t in st) / len(st)
        print(f"    {sess:>10}: {len(st)} trades | WR: {swr:.0f}% | Avg R: {srr:+.2f}")

    # By regime
    print(f"\n  {'_' * 50}")
    print(f"  BY REGIME:")
    for reg in ["bull", "bear", "sideways"]:
        rt = [t for t in closed if t.get("regime") == reg]
        if not rt:
            continue
        rw = sum(1 for t in rt if t["result"] == "win")
        rwr = rw / len(rt) * 100
        rrr = sum(t["rr"] for t in rt) / len(rt)
        print(f"    {reg:>10}: {len(rt)} trades | WR: {rwr:.0f}% | Avg R: {rrr:+.2f}")

    # Score buckets
    print(f"\n  {'_' * 50}")
    print(f"  BY SCORE BUCKET:")
    buckets = [(0, 4, "<4"), (4, 6, "4-6"), (6, 8, "6-8"), (8, 20, "8+")]
    for lo, hi, label in buckets:
        bt = [t for t in closed if lo <= t.get("trade_score", 0) < hi]
        if not bt:
            continue
        bwr = sum(1 for t in bt if t["result"] == "win") / len(bt) * 100
        brr = sum(t["rr"] for t in bt) / len(bt)
        print(f"    score {label:>4}: {len(bt)} trades | WR: {bwr:.0f}% | Avg R: {brr:+.2f}")

    _print_monthly_breakdown(closed)


def plot_zones(df, zones: list[dict], structure: dict, timeframe: str = "1h"):
    plot_df = df.tail(120).copy()
    plot_start_idx = len(df) - len(plot_df)

    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350", edge="inherit",
        wick={"up": "#26a69a", "down": "#ef5350"}, volume="in",
    )
    style = mpf.make_mpf_style(
        marketcolors=mc, base_mpf_style="nightclouds",
        gridstyle=":", gridcolor="#2a2a2a",
    )

    fig, axes = mpf.plot(
        plot_df, type="candle", style=style, volume=True,
        title=f"\nBTC/USD Supply & Demand -- {timeframe} | Structure + Liquidity",
        figsize=(22, 12), returnfig=True, warn_too_much_data=500,
    )

    ax = axes[0]
    price_min = plot_df["low"].min()
    price_max = plot_df["high"].max()
    price_range = price_max - price_min
    margin = price_range * 0.05

    # Zones
    drawn_labels = []
    for z in zones:
        if z["bottom"] > price_max + margin or z["top"] < price_min - margin:
            continue

        is_supply = z["type"] == "supply"
        color = "#ef5350" if is_supply else "#26a69a"

        alpha = 0.20
        if z["fresh"]:
            alpha += 0.10
        if z["with_trend"]:
            alpha += 0.05
        lw = 1.5

        ax.axhspan(z["bottom"], z["top"], alpha=alpha, facecolor=color,
                    edgecolor=color, linewidth=lw,
                    linestyle="-" if z["fresh"] else "--")

        label_y = z["top"] if is_supply else z["bottom"]
        too_close = any(abs(label_y - ly) < price_range * 0.015 for ly in drawn_labels)
        if too_close:
            label_y += price_range * 0.015 * (1 if is_supply else -1)
        drawn_labels.append(label_y)

        tags = []
        if z["fresh"]:
            tags.append("FRESH")
        if z["with_trend"]:
            tags.append("TREND")
        if z["has_bos"]:
            tags.append("BOS")
        if z.get("liq_swept"):
            tags.append("LIQ")
        tag_str = " ".join(tags)
        label = f'{z["type"].upper()} [{tag_str}] {z["strength"]:.1f}'

        ax.text(len(plot_df) + 1, label_y, f" {label}", fontsize=6,
                color=color, fontweight="bold",
                va="bottom" if is_supply else "top")

        ax.text(-1, z["top"], f"${z['top']:,.0f} ", fontsize=6, color=color, va="center", ha="right")
        ax.text(-1, z["bottom"], f"${z['bottom']:,.0f} ", fontsize=6, color=color, va="center", ha="right")

    # Trend background
    for seg_start, seg_end, trend in structure.get("trend_segments", []):
        x_s = seg_start - plot_start_idx
        x_e = seg_end - plot_start_idx
        if x_e < 0 or x_s >= len(plot_df):
            continue
        x_s, x_e = max(0, x_s), min(len(plot_df) - 1, x_e)
        if trend == "bullish":
            ax.axvspan(x_s, x_e, alpha=0.03, facecolor="#26a69a", linewidth=0)
        elif trend == "bearish":
            ax.axvspan(x_s, x_e, alpha=0.03, facecolor="#ef5350", linewidth=0)

    # Swing points
    for (idx, body_p, wick_p, label) in structure.get("swing_highs", []):
        x = idx - plot_start_idx
        if x < 0 or x >= len(plot_df):
            continue
        color = "#4fc3f7" if label == "HH" else "#ff8a65" if label == "LH" else "#666666"
        ax.plot(x, wick_p, marker="v", markersize=4, color=color, zorder=5, alpha=0.8)
        ax.text(x, wick_p + price_range * 0.006, f"{label}\n${body_p:,.0f}",
                fontsize=5, color=color, ha="center", va="bottom", fontweight="bold", alpha=0.9)

    for (idx, body_p, wick_p, label) in structure.get("swing_lows", []):
        x = idx - plot_start_idx
        if x < 0 or x >= len(plot_df):
            continue
        color = "#4fc3f7" if label == "HL" else "#ff8a65" if label == "LL" else "#666666"
        ax.plot(x, wick_p, marker="^", markersize=4, color=color, zorder=5, alpha=0.8)
        ax.text(x, wick_p - price_range * 0.006, f"{label}\n${body_p:,.0f}",
                fontsize=5, color=color, ha="center", va="top", fontweight="bold", alpha=0.9)

    # Trend lines
    vis_h = [(idx - plot_start_idx, bp) for idx, bp, wp, _ in structure.get("swing_highs", [])
             if 0 <= idx - plot_start_idx < len(plot_df)]
    vis_l = [(idx - plot_start_idx, bp) for idx, bp, wp, _ in structure.get("swing_lows", [])
             if 0 <= idx - plot_start_idx < len(plot_df)]
    if len(vis_h) >= 2:
        ax.plot([p[0] for p in vis_h], [p[1] for p in vis_h],
                color="#ff8a65", linewidth=0.8, alpha=0.5, linestyle="--", zorder=3)
    if len(vis_l) >= 2:
        ax.plot([p[0] for p in vis_l], [p[1] for p in vis_l],
                color="#4fc3f7", linewidth=0.8, alpha=0.5, linestyle="--", zorder=3)

    # BOS
    vis_bos = [(idx, p, d) for idx, p, d in structure.get("bos_events", [])
               if 0 <= idx - plot_start_idx < len(plot_df)
               and price_min - margin <= p <= price_max + margin][-6:]
    bos_lp = []
    for (idx, price, direction) in vis_bos:
        x = idx - plot_start_idx
        bc = "#4fc3f7" if direction == "bullish" else "#ff8a65"
        ax.plot([x, min(x + 12, len(plot_df) - 1)], [price, price],
                color=bc, linewidth=1.0, linestyle="-.", alpha=0.6, zorder=4)
        if not any(abs(price - bp) < price_range * 0.012 for bp in bos_lp):
            ax.text(x + 1, price, " BOS", fontsize=5.5, color=bc, fontweight="bold",
                    va="bottom" if direction == "bullish" else "top",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#1a1a1a",
                              edgecolor=bc, alpha=0.6, linewidth=0.5))
            bos_lp.append(price)

    # Liquidity levels
    for lv in structure.get("liquidity_levels", []):
        p = lv["price"]
        if p < price_min - margin or p > price_max + margin:
            continue
        lv_color = "#ffd700"
        ax.axhline(p, color=lv_color, linewidth=0.7, linestyle=":", alpha=0.4)
        side = "EQH" if lv["type"] == "equal_highs" else "EQL"
        ax.text(0, p, f" {side}({lv['touches']}x) ${p:,.0f}", fontsize=5,
                color=lv_color, va="center", alpha=0.7)

    # Current price
    last_close = plot_df["close"].iloc[-1]
    ax.axhline(last_close, color="#ffd700", linewidth=1, linestyle=":", alpha=0.6)
    ax.text(len(plot_df) + 1, last_close, f" ${last_close:,.0f}",
            fontsize=9, color="#ffd700", va="center", fontweight="bold")

    # Legend
    ax.text(0.005, 0.98,
            "Zones: Solid=FRESH  Dashed=TESTED  |  EQH/EQL=liquidity\n"
            "Structure (body): HH/HL=bull(blue)  LH/LL=bear(orange)  BOS=close breaks swing\n"
            "Score = impulse + volume + trend + BOS + fresh + liquidity",
            transform=ax.transAxes, fontsize=5.5, color="#888888", va="top",
            bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.85))

    chart_path = str(OUTPUT_DIR / "sd_zones_chart.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n   Chart saved -> {chart_path}")


def print_oos_summary(results: dict):
    """Print walk-forward optimization results with IS vs OOS comparison."""
    folds = results.get("folds", [])
    summary = results.get("summary", {})
    oos_trades = results.get("oos_trades", [])

    if not folds:
        print("\n  No WFO results to display.")
        return

    print(f"\n{'=' * 78}")
    print(f"  ANCHORED WALK-FORWARD OPTIMIZATION RESULTS")
    print(f"{'=' * 78}")
    print(f"  Folds: {len(folds)} | OOS trades: {summary.get('trades', 0)}")

    print(f"\n  {'Fold':>4} | {'Train Period':^25} | {'Test Period':^25} | "
          f"{'IS R':>6} | {'OOS R':>6} | {'OOS WR':>6} | {'OOS PF':>6}")
    print(f"  {'_' * 4}_|_{'_' * 25}_|_{'_' * 25}_|_{'_' * 6}_|_{'_' * 6}_|_{'_' * 6}_|_{'_' * 6}")

    for f in folds:
        is_m = f["is_metrics"]
        oos_m = f["oos_metrics"]
        print(f"  {f['fold']:>4} | {f['train_period']:^25} | {f['test_period']:^25} | "
              f"{is_m.get('total_r', 0):>+6.1f} | {oos_m.get('total_r', 0):>+6.1f} | "
              f"{oos_m.get('winrate', 0):>5.0f}% | {oos_m.get('pf', 0):>6.2f}")

    print(f"\n  {'=' * 60}")
    print(f"  AGGREGATE OUT-OF-SAMPLE (unseen data only):")
    print(f"    Trades: {summary.get('trades', 0)}  |  "
          f"Winrate: {summary.get('winrate', 0):.1f}%  |  "
          f"Total R: {summary.get('total_r', 0):+.1f}  |  "
          f"PF: {summary.get('pf', 0):.2f}")
    print(f"    Avg R/trade: {summary.get('avg_r', 0):+.3f}  |  "
          f"Max DD: {summary.get('max_dd_r', 0):.1f}R  |  "
          f"Edge score: {summary.get('edge_score', 0):.2f}")

    is_totals = [f["is_metrics"] for f in folds if f["is_metrics"].get("trades", 0) > 0]
    if is_totals:
        avg_is_wr = np.mean([m["winrate"] for m in is_totals])
        avg_is_pf = np.mean([m["pf"] for m in is_totals])
        avg_is_avg_r = np.mean([m["avg_r"] for m in is_totals])
        oos_wr = summary.get("winrate", 0)
        oos_pf = summary.get("pf", 0)
        oos_avg_r = summary.get("avg_r", 0)

        def pct_change(old, new):
            if abs(old) < 0.001:
                return 0
            return (new - old) / abs(old) * 100

        print(f"\n  {'_' * 60}")
        print(f"  IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON:")
        print(f"    {'Metric':<20} | {'In-Sample':>12} | {'Out-of-Sample':>14} | {'Change':>8}")
        print(f"    {'_' * 20}_|_{'_' * 12}_|_{'_' * 14}_|_{'_' * 8}")
        print(f"    {'Winrate':<20} | {avg_is_wr:>11.1f}% | {oos_wr:>13.1f}% | "
              f"{pct_change(avg_is_wr, oos_wr):>+7.1f}%")
        print(f"    {'Profit Factor':<20} | {avg_is_pf:>12.2f} | {oos_pf:>14.2f} | "
              f"{pct_change(avg_is_pf, oos_pf):>+7.1f}%")
        print(f"    {'Avg R/trade':<20} | {avg_is_avg_r:>+12.3f} | {oos_avg_r:>+14.3f} | "
              f"{pct_change(avg_is_avg_r, oos_avg_r):>+7.1f}%")

    closed_oos = [t for t in oos_trades if t["result"] in ("win", "loss")]
    if closed_oos:
        print(f"\n  {'_' * 60}")
        print(f"  BY REGIME (out-of-sample):")
        for reg in ["bull", "bear", "sideways"]:
            rt = [t for t in closed_oos if t.get("regime") == reg]
            if not rt:
                continue
            rw = sum(1 for t in rt if t["result"] == "win")
            rwr = rw / len(rt) * 100
            rrr = sum(t["rr"] for t in rt) / len(rt)
            print(f"    {reg:>10}: {len(rt)} trades | WR: {rwr:.0f}% | Avg R: {rrr:+.3f}")

        _print_monthly_breakdown(closed_oos, label="BY MONTH (out-of-sample)")

    print(f"\n  {'_' * 60}")
    print(f"  PARAMETER STABILITY:")
    param_keys = ["BT_SL_ATR_MULT", "BT_TP_RR", "BT_MAX_OPEN_TRADES"]
    for key in param_keys:
        counts = defaultdict(int)
        for f in folds:
            val = f["best_params"].get(key)
            counts[val] += 1
        most_common_val = max(counts, key=counts.get)
        most_common_pct = counts[most_common_val] / len(folds) * 100
        print(f"    {key}: {most_common_val} chosen {counts[most_common_val]}/{len(folds)} "
              f"folds ({most_common_pct:.0f}%)")
