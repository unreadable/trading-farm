"""Zone detection from OHLCV data."""

from __future__ import annotations

import pandas as pd

from config import (
    ATR_PERIOD, IMPULSE_MULT, MIN_BODY_RATIO, MAX_ZONE_WIDTH_PCT,
    BASE_MAX_CANDLES, BASE_BODY_THRESH, WICK_TOLERANCE_PCT, VOL_ZSCORE_THRESH,
)
from structure import (
    compute_atr, compute_vol_zscore,
    detect_swing_points, detect_trend_at, check_bos,
    detect_liquidity_levels, check_liquidity_swept,
)


def detect_zones(df: pd.DataFrame, timeframe: str = "1h", no_lookahead: bool = False) -> tuple[list[dict], dict]:
    atr = compute_atr(df, ATR_PERIOD)
    vol_z = compute_vol_zscore(df, 20)
    body = (df["close"] - df["open"]).abs()
    candle_range = df["high"] - df["low"]
    mult = IMPULSE_MULT
    max_width_pct = MAX_ZONE_WIDTH_PCT

    swing_highs, swing_lows = detect_swing_points(df, lookback=5)

    # Liquidity levels
    liq_levels = detect_liquidity_levels(df, swing_highs, swing_lows)

    # Classify swings
    labeled_highs = []
    for k, (i, body_p, wick_p) in enumerate(swing_highs):
        if k == 0:
            labeled_highs.append((i, body_p, wick_p, "SH"))
        else:
            prev_body = swing_highs[k - 1][1]
            labeled_highs.append((i, body_p, wick_p, "HH" if body_p > prev_body else "LH"))

    labeled_lows = []
    for k, (i, body_p, wick_p) in enumerate(swing_lows):
        if k == 0:
            labeled_lows.append((i, body_p, wick_p, "SL"))
        else:
            prev_body = swing_lows[k - 1][1]
            labeled_lows.append((i, body_p, wick_p, "HL" if body_p > prev_body else "LL"))

    # BOS events
    bos_events = []
    for k in range(1, len(swing_highs)):
        curr_i, curr_body, _ = swing_highs[k]
        prev_body = swing_highs[k - 1][1]
        if curr_body > prev_body:
            for bi in range(swing_highs[k - 1][0] + 1, min(curr_i + 1, len(df))):
                if df["close"].iloc[bi] > prev_body:
                    bos_events.append((bi, prev_body, "bullish"))
                    break

    for k in range(1, len(swing_lows)):
        curr_i, curr_body, _ = swing_lows[k]
        prev_body = swing_lows[k - 1][1]
        if curr_body < prev_body:
            for bi in range(swing_lows[k - 1][0] + 1, min(curr_i + 1, len(df))):
                if df["close"].iloc[bi] < prev_body:
                    bos_events.append((bi, prev_body, "bearish"))
                    break

    # Trend segments
    trend_segments = []
    seg_start = 0
    prev_trend = "ranging"
    for i in range(5, len(df), 5):
        t = detect_trend_at(df, i, swing_highs, swing_lows)
        if t != prev_trend:
            if seg_start < i:
                trend_segments.append((seg_start, i, prev_trend))
            seg_start = i
            prev_trend = t
    trend_segments.append((seg_start, len(df) - 1, prev_trend))

    structure = {
        "swing_highs": labeled_highs,
        "swing_lows": labeled_lows,
        "bos_events": bos_events,
        "trend_segments": trend_segments,
        "liquidity_levels": liq_levels,
    }

    # -- Zone detection loop --
    zones = []
    used = set()

    for i in range(3, len(df)):
        if pd.isna(atr.iloc[i]) or atr.iloc[i] == 0:
            continue

        rng = candle_range.iloc[i]
        if rng == 0:
            continue
        body_ratio = body.iloc[i] / rng
        if body_ratio < MIN_BODY_RATIO:
            continue

        is_single = body.iloc[i] >= mult * atr.iloc[i]

        is_double = False
        if i >= 3 and not is_single:
            same_dir = (
                (df["close"].iloc[i] > df["open"].iloc[i] and
                 df["close"].iloc[i-1] > df["open"].iloc[i-1]) or
                (df["close"].iloc[i] < df["open"].iloc[i] and
                 df["close"].iloc[i-1] < df["open"].iloc[i-1])
            )
            if same_dir:
                combined_body = abs(df["close"].iloc[i] - df["open"].iloc[i-1])
                pair_range = max(df["high"].iloc[i], df["high"].iloc[i-1]) - \
                             min(df["low"].iloc[i], df["low"].iloc[i-1])
                pair_ratio = combined_body / pair_range if pair_range > 0 else 0
                if combined_body >= mult * 1.3 * atr.iloc[i] and pair_ratio >= 0.50:
                    is_double = True

        if not is_single and not is_double:
            continue

        if is_double:
            bullish = df["close"].iloc[i] > df["open"].iloc[i-1]
            impulse_start = i - 1
        else:
            bullish = df["close"].iloc[i] > df["open"].iloc[i]
            impulse_start = i

        zone_type = "demand" if bullish else "supply"

        origin_end = impulse_start
        origin_start = impulse_start - 1
        if origin_start < 0 or origin_start in used:
            continue

        j = origin_start
        while j > max(0, origin_start - BASE_MAX_CANDLES + 1):
            prev = j - 1
            if prev < 0 or pd.isna(atr.iloc[prev]):
                break
            if body.iloc[prev] >= BASE_BODY_THRESH * atr.iloc[prev]:
                break
            j -= 1

        origin_start = j
        origin_candles = list(range(origin_start, origin_end))
        if not origin_candles:
            origin_candles = [impulse_start - 1]

        origin_slice = df.iloc[origin_candles]
        zone_top = float(origin_slice["high"].max())
        zone_bottom = float(origin_slice["low"].min())

        mid = (zone_top + zone_bottom) / 2
        max_width = mid * max_width_pct
        if zone_top - zone_bottom > max_width:
            if zone_type == "demand":
                zone_top = zone_bottom + max_width
            else:
                zone_bottom = zone_top - max_width

        for c in origin_candles:
            used.add(c)

        # Freshness (wick-tolerant) -- skip lookahead when in backtest mode
        if no_lookahead:
            fresh = True
            mitigated = False
        else:
            future = df.iloc[i + 1:] if i + 1 < len(df) else pd.DataFrame()
            fresh = True
            mitigated = False
            tolerance = mid * WICK_TOLERANCE_PCT

            if not future.empty:
                if zone_type == "demand":
                    if (future["close"] < zone_bottom - tolerance).any():
                        fresh = False
                    elif (future["low"] <= zone_top).any():
                        mitigated = True
                else:
                    if (future["close"] > zone_top + tolerance).any():
                        fresh = False
                    elif (future["high"] >= zone_bottom).any():
                        mitigated = True

        # Trend
        trend = detect_trend_at(df, i, swing_highs, swing_lows)
        with_trend = (
            (zone_type == "demand" and trend == "bullish") or
            (zone_type == "supply" and trend == "bearish")
        )
        counter_trend = (
            (zone_type == "demand" and trend == "bearish") or
            (zone_type == "supply" and trend == "bullish")
        )

        has_bos = check_bos(df, i,
                            "bullish" if zone_type == "demand" else "bearish",
                            swing_highs, swing_lows)

        # Liquidity swept
        zone_dict_temp = {"type": zone_type, "top": zone_top, "bottom": zone_bottom,
                          "impulse_idx": i}
        liq_swept = check_liquidity_swept(df, liq_levels, zone_dict_temp)

        # Strength scoring
        impulse_strength = float(body.iloc[i] / atr.iloc[i])
        vz = vol_z.iloc[i] if not pd.isna(vol_z.iloc[i]) else 0
        vol_bonus = max(0, float(vz)) * 0.3 if vz >= VOL_ZSCORE_THRESH else 0
        trend_bonus = 0.5 if with_trend else (-0.3 if counter_trend else 0)
        bos_bonus = 0.4 if has_bos else 0
        fresh_bonus = 0.5 if (fresh and not mitigated) else 0
        liq_bonus = 0.5 if liq_swept else 0

        strength = impulse_strength + vol_bonus + trend_bonus + bos_bonus + fresh_bonus + liq_bonus

        zones.append({
            "type": zone_type,
            "top": zone_top,
            "bottom": zone_bottom,
            "mid": (zone_top + zone_bottom) / 2,
            "width": zone_top - zone_bottom,
            "time": df.index[origin_start],
            "timeframe": timeframe,
            "fresh": fresh,
            "mitigated": mitigated,
            "strength": round(strength, 2),
            "trend": trend,
            "with_trend": with_trend,
            "has_bos": has_bos,
            "liq_swept": liq_swept,
            "vol_zscore": round(float(vz), 2),
            "body_ratio": round(float(body_ratio), 2),
            "base_candles": len(origin_candles),
            "impulse_idx": i,
        })

    zones.sort(key=lambda z: -z["strength"])
    return zones, structure
