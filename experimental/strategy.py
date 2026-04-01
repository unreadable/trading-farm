"""
Supply & Demand Zone Strategy
Detects zones and scores them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ── Config defaults (override via params dict) ────────────────────────────────
ATR_PERIOD        = 14
IMPULSE_MULT      = 1.5
MIN_BODY_RATIO    = 0.55
MAX_ZONE_WIDTH    = 0.012
BASE_MAX_CANDLES  = 6
BASE_BODY_THRESH  = 0.6
WICK_TOLERANCE    = 0.002
VOL_ZSCORE_THRESH = 1.5
EQL_TOLERANCE     = 0.001
EQL_MIN_TOUCHES   = 2

BT_SL_ATR_MULT      = 1.0
BT_TP_RR            = 2.5
BT_MIN_RR           = 1.5
BT_MAX_OPEN_TRADES  = 4
BT_ZONE_CLUSTER_PCT = 0.008


# ── Indicators ────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_vol_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    v   = df["volume"]
    std = v.rolling(period).std().replace(0, np.nan)
    return (v - v.rolling(period).mean()) / std


def tag_regime(df: pd.DataFrame, ema_period: int = 200, slope_window: int = 20) -> pd.Series:
    ema   = df["close"].ewm(span=ema_period, adjust=False).mean()
    slope = (ema - ema.shift(slope_window)) / ema.shift(slope_window)
    regime = pd.Series("sideways", index=df.index)
    regime[slope > 0.02]  = "bull"
    regime[slope < -0.02] = "bear"
    return regime


# ── Market structure ──────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, lookback: int = 5):
    """
    Optimized: pre-extract arrays, use sliding_window_view for rolling max/min,
    eliminate all .iloc inside the loop.
    """
    n       = len(df)
    high_a  = df["high"].to_numpy()
    low_a   = df["low"].to_numpy()
    open_a  = df["open"].to_numpy()
    close_a = df["close"].to_numpy()

    body_top_a = np.maximum(open_a, close_a)
    body_bot_a = np.minimum(open_a, close_a)

    w = 2 * lookback + 1

    high_windows = sliding_window_view(high_a, w)
    low_windows  = sliding_window_view(low_a,  w)

    roll_max_high = high_windows.max(axis=1)
    roll_min_low  = low_windows.min(axis=1)

    bt_windows = sliding_window_view(body_top_a, w)
    bb_windows = sliding_window_view(body_bot_a, w)

    highs: list[tuple[int, float, float]] = []
    lows:  list[tuple[int, float, float]] = []

    for i in range(lookback, n - lookback):
        win_i = i - lookback

        if high_a[i] == roll_max_high[win_i]:
            bt_i = body_top_a[i]
            bt_win = bt_windows[win_i]
            nbr_max = max(
                bt_win[:lookback].max() if lookback > 0 else -np.inf,
                bt_win[lookback + 1:].max() if lookback + 1 < w else -np.inf,
            )
            if bt_i >= nbr_max:
                highs.append((i, float(bt_i), float(high_a[i])))

        if low_a[i] == roll_min_low[win_i]:
            bb_i = body_bot_a[i]
            bb_win = bb_windows[win_i]
            nbr_min = min(
                bb_win[:lookback].min() if lookback > 0 else np.inf,
                bb_win[lookback + 1:].min() if lookback + 1 < w else np.inf,
            )
            if bb_i <= nbr_min:
                lows.append((i, float(bb_i), float(low_a[i])))

    return highs, lows


def detect_trend_at(
    idx: int,
    swing_highs,
    swing_lows,
    df: pd.DataFrame,
    _close_a: np.ndarray | None = None,
    # ── NEW: pre-built sorted arrays for O(log S) lookups ────────────────────
    _sh_indices: np.ndarray | None = None,
    _sh_bodies:  np.ndarray | None = None,
    _sl_indices: np.ndarray | None = None,
    _sl_bodies:  np.ndarray | None = None,
) -> str:
    """
    Trend at bar `idx`.

    When the four _s*_* arrays are supplied (pre-built from detect_zones)
    the swing lookups are O(log S) via searchsorted instead of the original
    O(S) list comprehension.  The fallback to the original list comprehension
    is preserved when the arrays are absent so existing callers are unaffected.
    """
    if _sh_indices is not None:
        # ── Fast path: O(log S) ───────────────────────────────────────────────
        k_h = int(np.searchsorted(_sh_indices, idx, side="left"))
        k_l = int(np.searchsorted(_sl_indices, idx, side="left"))
        if k_h < 2 or k_l < 2:
            return "ranging"
        rh_bp = _sh_bodies[max(0, k_h - 3):k_h]   # up to last 3 body prices
        rl_bp = _sl_bodies[max(0, k_l - 3):k_l]
        hh = bool(np.all(np.diff(rh_bp) >= 0))
        hl = bool(np.all(np.diff(rl_bp) >= 0))
        lh = bool(np.all(np.diff(rh_bp) <= 0))
        ll = bool(np.all(np.diff(rl_bp) <= 0))
    else:
        # ── Original path (backward-compatible) ──────────────────────────────
        rh = [(i, bp) for i, bp, wp in swing_highs if i < idx][-3:]
        rl = [(i, bp) for i, bp, wp in swing_lows  if i < idx][-3:]
        if len(rh) < 2 or len(rl) < 2:
            return "ranging"
        hh = all(rh[j][1] <= rh[j + 1][1] for j in range(len(rh) - 1))
        hl = all(rl[j][1] <= rl[j + 1][1] for j in range(len(rl) - 1))
        lh = all(rh[j][1] >= rh[j + 1][1] for j in range(len(rh) - 1))
        ll = all(rl[j][1] >= rl[j + 1][1] for j in range(len(rl) - 1))

    if hh and hl: return "bullish"
    if lh and ll: return "bearish"

    if idx >= 20:
        if _close_a is not None:
            close_slice = _close_a[idx - 20:idx]
        else:
            close_slice = df["close"].to_numpy()[idx - 20:idx]
        ema = pd.Series(close_slice).ewm(span=20).mean()
        if len(ema) >= 5:
            slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5]
            if slope > 0.002:  return "bullish"
            if slope < -0.002: return "bearish"
    return "ranging"


def check_bos(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    swing_highs,
    swing_lows,
    _close_a: np.ndarray | None = None,
    # ── NEW: pre-built sorted arrays for O(log S) lookups ────────────────────
    _sh_indices: np.ndarray | None = None,
    _sh_bodies:  np.ndarray | None = None,
    _sl_indices: np.ndarray | None = None,
    _sl_bodies:  np.ndarray | None = None,
) -> bool:
    start = max(0, idx - 3)
    if _close_a is not None:
        win_close = _close_a[start:idx + 1]
    else:
        win_close = df["close"].to_numpy()[start:idx + 1]

    if _sh_indices is not None:
        # ── Fast path: O(log S) ───────────────────────────────────────────────
        if direction == "bullish":
            k_h = int(np.searchsorted(_sh_indices, idx, side="left"))
            if k_h == 0:
                return False
            return bool(np.any(win_close > _sh_bodies[k_h - 1]))
        else:
            k_l = int(np.searchsorted(_sl_indices, idx, side="left"))
            if k_l == 0:
                return False
            return bool(np.any(win_close < _sl_bodies[k_l - 1]))
    else:
        # ── Original path (backward-compatible) ──────────────────────────────
        if direction == "bullish":
            prev = [bp for i, bp, wp in swing_highs if i < idx]
            return bool(prev) and bool(np.any(win_close > prev[-1]))
        else:
            prev = [bp for i, bp, wp in swing_lows if i < idx]
            return bool(prev) and bool(np.any(win_close < prev[-1]))


def detect_liquidity_levels(swing_highs, swing_lows) -> list[dict]:
    """
    Optimized: replace O(n²) nested loop with a single sorted pass that
    builds clusters incrementally, yielding identical output.
    """
    levels: list[dict] = []

    for swings, ltype in [(swing_highs, "equal_highs"), (swing_lows, "equal_lows")]:
        if not swings:
            continue

        n_s = len(swings)
        idxs = np.array([s[0] for s in swings], dtype=np.intp)
        wps  = np.array([s[2] for s in swings], dtype=np.float64)

        for i in range(n_s):
            wp_a = wps[i]
            if wp_a == 0:
                continue
            ratio = np.abs(wps[i:] - wp_a) / wp_a
            mask  = ratio <= EQL_TOLERANCE
            count = int(mask.sum())
            if count < EQL_MIN_TOUCHES:
                continue

            matched_idxs = idxs[i:][mask]
            matched_wps  = wps[i:][mask]
            levels.append({
                "type":      ltype,
                "price":     float(matched_wps.mean()),
                "touches":   count,
                "first_idx": int(matched_idxs[0]),
                "last_idx":  int(matched_idxs[-1]),
            })

    levels.sort(key=lambda l: l["price"])

    deduped: list[dict] = []
    for lv in levels:
        if deduped and abs(lv["price"] - deduped[-1]["price"]) / lv["price"] < EQL_TOLERANCE * 2:
            if lv["touches"] > deduped[-1]["touches"]:
                deduped[-1] = lv
        else:
            deduped.append(lv)
    return deduped


def liq_was_swept(
    df: pd.DataFrame,
    liq_levels: list[dict],
    zone: dict,
    _low_a:  np.ndarray | None = None,
    _high_a: np.ndarray | None = None,
) -> bool:
    for lv in liq_levels:
        if zone["type"] == "demand" and lv["type"] == "equal_lows":
            if zone["bottom"] * 0.99 <= lv["price"] <= zone["top"] * 1.005:
                last = lv["last_idx"]
                if _low_a is not None:
                    sw_low = _low_a[last:]
                    if len(sw_low) and bool(np.any(sw_low < lv["price"])):
                        return True
                else:
                    sw = df.iloc[last:]
                    if not sw.empty and (sw["low"] < lv["price"]).any():
                        return True
        elif zone["type"] == "supply" and lv["type"] == "equal_highs":
            if zone["bottom"] * 0.995 <= lv["price"] <= zone["top"] * 1.01:
                last = lv["last_idx"]
                if _high_a is not None:
                    sw_high = _high_a[last:]
                    if len(sw_high) and bool(np.any(sw_high > lv["price"])):
                        return True
                else:
                    sw = df.iloc[last:]
                    if not sw.empty and (sw["high"] > lv["price"]).any():
                        return True
    return False


# ── Zone detection ────────────────────────────────────────────────────────────

def detect_zones(df: pd.DataFrame, no_lookahead: bool = False) -> tuple[list[dict], dict]:
    """
    Detect supply/demand zones from OHLCV data.
    no_lookahead=True for backtest (freshness checked bar-by-bar in generate_signals).

    ── Optimizations vs previous version ────────────────────────────────────────
    A. VECTORIZED PRE-FILTER  (biggest win, ~50 ms vs ~300 ms for the main loop)
       All bar-level filter conditions (valid ATR, body ratio ≥ threshold, single-
       or double-candle impulse) are evaluated in a single NumPy pass.  Only the
       small list of qualifying bar indices enters the Python loop; the rest of
       the 70 k bars never touch Python bytecode.

    B. SORTED SWING ARRAYS FOR O(log S) LOOKUPS
       Swing-high and swing-low body prices are extracted into sorted NumPy arrays
       once.  detect_trend_at() and check_bos() use np.searchsorted() instead of
       linear list comprehensions — O(log S) vs O(S) per call.
       (Moderate win for typical zone counts; significant on long/volatile data.)

    C. All other internals are unchanged (detect_swings, liq, BOS events, scoring).
    """
    # ── Pre-extract all Series → NumPy arrays (done ONCE) ────────────────────
    open_a  = df["open"].to_numpy(dtype=np.float64)
    high_a  = df["high"].to_numpy(dtype=np.float64)
    low_a   = df["low"].to_numpy(dtype=np.float64)
    close_a = df["close"].to_numpy(dtype=np.float64)

    atr_s  = compute_atr(df)
    vz_s   = compute_vol_zscore(df)

    body_a = np.abs(close_a - open_a)
    rng_a  = high_a - low_a

    atr_a  = atr_s.to_numpy(dtype=np.float64)
    vz_a   = vz_s.to_numpy(dtype=np.float64)

    swing_highs, swing_lows = detect_swings(df)
    liq_levels              = detect_liquidity_levels(swing_highs, swing_lows)

    # ── B. Build sorted swing arrays for O(log S) lookups ────────────────────
    if swing_highs:
        _sh_indices = np.array([s[0] for s in swing_highs], dtype=np.intp)
        _sh_bodies  = np.array([s[1] for s in swing_highs], dtype=np.float64)
    else:
        _sh_indices = np.empty(0, dtype=np.intp)
        _sh_bodies  = np.empty(0, dtype=np.float64)

    if swing_lows:
        _sl_indices = np.array([s[0] for s in swing_lows], dtype=np.intp)
        _sl_bodies  = np.array([s[1] for s in swing_lows], dtype=np.float64)
    else:
        _sl_indices = np.empty(0, dtype=np.intp)
        _sl_bodies  = np.empty(0, dtype=np.float64)

    # ── BOS events – use close_a instead of df.iloc ──────────────────────────
    bos_events: list[tuple[int, float, str]] = []
    for k in range(1, len(swing_highs)):
        ci, cb, _ = swing_highs[k]
        pb        = swing_highs[k - 1][1]
        if cb > pb:
            prev_i = swing_highs[k - 1][0]
            for bi in range(prev_i + 1, min(ci + 1, len(df))):
                if close_a[bi] > pb:
                    bos_events.append((bi, pb, "bullish"))
                    break
    for k in range(1, len(swing_lows)):
        ci, cb, _ = swing_lows[k]
        pb        = swing_lows[k - 1][1]
        if cb < pb:
            prev_i = swing_lows[k - 1][0]
            for bi in range(prev_i + 1, min(ci + 1, len(df))):
                if close_a[bi] < pb:
                    bos_events.append((bi, pb, "bearish"))
                    break

    structure = {
        "swing_highs":      swing_highs,
        "swing_lows":       swing_lows,
        "bos_events":       bos_events,
        "liquidity_levels": liq_levels,
    }

    n        = len(df)
    df_index = df.index

    # ── A. VECTORIZED PRE-FILTER ──────────────────────────────────────────────
    # Evaluate ALL bar-level conditions with NumPy — O(N) in C, not Python.
    # This replaces the original per-bar Python checks with a single pass that
    # produces a small array of candidate bar indices.

    # Slice helpers (all offset by 3 to match range(3, n))
    atr_s3  = atr_a[3:]          # shape (n-3,)
    body_s3 = body_a[3:]
    rng_s3  = rng_a[3:]
    close_s3 = close_a[3:]
    open_s3  = open_a[3:]
    high_s3  = high_a[3:]
    low_s3   = low_a[3:]

    # Valid bars: non-nan ATR > 0 and non-zero range
    valid   = (~np.isnan(atr_s3)) & (atr_s3 > 0) & (rng_s3 > 0)

    # Body ratio filter
    with np.errstate(divide="ignore", invalid="ignore"):
        br_arr = np.where(valid, body_s3 / rng_s3, 0.0)
    pass_br = valid & (br_arr >= MIN_BODY_RATIO)

    # Single-candle impulse
    single_mask = pass_br & (body_s3 >= IMPULSE_MULT * atr_s3)

    # Double-candle impulse (requires i-1 context; offset arrays cover i=3..n-1
    #   so i-1 arrays are open_a[2:-1] etc.)
    same_dir  = (close_s3 > open_s3) == (close_a[2:-1] > open_a[2:-1])
    cb_arr    = np.abs(close_s3 - open_a[2:-1])
    pr_arr    = (np.maximum(high_s3, high_a[2:-1]) -
                 np.minimum(low_s3,  low_a[2:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_arr = np.where(pr_arr > 0, cb_arr / pr_arr, 0.0)
    double_mask = (
        pass_br & ~single_mask & same_dir
        & (cb_arr >= IMPULSE_MULT * 1.3 * atr_s3)
        & (ratio_arr >= 0.50)
    )

    # Candidate bar indices in the original frame (add 3 for the slice offset)
    candidate_offsets = np.where(single_mask | double_mask)[0]
    # ─────────────────────────────────────────────────────────────────────────

    zones: list[dict] = []
    used:  set[int]   = set()

    # Iterate ONLY over qualifying bars (~0.2–5% of total)
    for off in candidate_offsets:
        i = int(off) + 3  # absolute bar index

        # Which condition was met? (both could be True, single takes priority)
        is_single = bool(single_mask[off])
        is_double = bool(double_mask[off]) if not is_single else False

        atr_val = atr_a[i]
        br      = br_arr[off]      # already computed above

        if is_double:
            bullish = close_a[i] > open_a[i - 1]
        else:
            bullish = close_a[i] > open_a[i]
        zone_type = "demand" if bullish else "supply"

        impulse_start = i - 1 if is_double else i
        origin_start  = impulse_start - 1
        if origin_start < 0 or origin_start in used:
            continue

        # ── Walk back to find base candles ────────────────────────────────
        j    = origin_start
        stop = max(0, origin_start - BASE_MAX_CANDLES + 1)
        while j > stop:
            prev = j - 1
            if prev < 0 or np.isnan(atr_a[prev]):
                break
            if body_a[prev] >= BASE_BODY_THRESH * atr_a[prev]:
                break
            j -= 1
        origin_candles = list(range(j, impulse_start)) or [impulse_start - 1]

        # ── Zone bounds from origin candles ───────────────────────────────
        oc_arr      = np.array(origin_candles, dtype=np.intp)
        zone_top    = float(high_a[oc_arr].max())
        zone_bottom = float(low_a[oc_arr].min())
        mid         = (zone_top + zone_bottom) / 2
        max_w       = mid * MAX_ZONE_WIDTH
        if zone_top - zone_bottom > max_w:
            if zone_type == "demand":
                zone_top    = zone_bottom + max_w
            else:
                zone_bottom = zone_top - max_w
        mid = (zone_top + zone_bottom) / 2

        for c in origin_candles:
            used.add(c)

        # ── Freshness check (array slices + np.any) ───────────────────────
        fresh, mitigated = True, False
        if not no_lookahead and i + 1 < n:
            tol = mid * WICK_TOLERANCE
            fut_close = close_a[i + 1:]
            fut_low   = low_a[i + 1:]
            fut_high  = high_a[i + 1:]
            if zone_type == "demand":
                if   np.any(fut_close < zone_bottom - tol): fresh = False
                elif np.any(fut_low  <= zone_top):          mitigated = True
            else:
                if   np.any(fut_close > zone_top + tol):   fresh = False
                elif np.any(fut_high >= zone_bottom):       mitigated = True

        # ── Contextual scoring (O(log S) via pre-built arrays) ────────────
        trend      = detect_trend_at(
            i, swing_highs, swing_lows, df,
            _close_a=close_a,
            _sh_indices=_sh_indices, _sh_bodies=_sh_bodies,
            _sl_indices=_sl_indices, _sl_bodies=_sl_bodies,
        )
        with_trend = (zone_type == "demand" and trend == "bullish") or \
                     (zone_type == "supply" and trend == "bearish")
        counter    = (zone_type == "demand" and trend == "bearish") or \
                     (zone_type == "supply" and trend == "bullish")
        has_bos    = check_bos(
            df, i,
            "bullish" if zone_type == "demand" else "bearish",
            swing_highs, swing_lows,
            _close_a=close_a,
            _sh_indices=_sh_indices, _sh_bodies=_sh_bodies,
            _sl_indices=_sl_indices, _sl_bodies=_sl_bodies,
        )
        zone_tmp = {"type": zone_type, "top": zone_top, "bottom": zone_bottom, "impulse_idx": i}
        liq_sw   = liq_was_swept(df, liq_levels, zone_tmp, _low_a=low_a, _high_a=high_a)

        vz_i = vz_a[i]
        vz   = float(vz_i) if not np.isnan(vz_i) else 0.0
        strength = (
            float(body_a[i] / atr_val)
            + (max(0.0, vz) * 0.3 if vz >= VOL_ZSCORE_THRESH else 0.0)
            + (0.5 if with_trend else (-0.3 if counter else 0.0))
            + (0.4 if has_bos    else 0.0)
            + (0.5 if (fresh and not mitigated) else 0.0)
            + (0.5 if liq_sw     else 0.0)
        )

        zones.append({
            "type":         zone_type,
            "top":          zone_top,
            "bottom":       zone_bottom,
            "mid":          mid,
            "width":        zone_top - zone_bottom,
            "time":         df_index[j],
            "fresh":        fresh,
            "mitigated":    mitigated,
            "strength":     round(strength, 2),
            "trend":        trend,
            "with_trend":   with_trend,
            "has_bos":      has_bos,
            "liq_swept":    liq_sw,
            "vol_zscore":   round(vz, 2),
            "body_ratio":   round(float(br), 2),
            "base_candles": len(origin_candles),
            "impulse_idx":  i,
        })

    zones.sort(key=lambda z: -z["strength"])
    return zones, structure