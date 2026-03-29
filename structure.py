"""Indicators & market structure: ATR, volume, regime, swings, trend, BOS, liquidity."""

from __future__ import annotations

import pandas as pd
import numpy as np

from config import EQL_TOLERANCE_PCT, EQL_MIN_TOUCHES


# ── Indicators ───────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_vol_zscore(df: pd.DataFrame, period: int = 20) -> pd.Series:
    vol = df["volume"]
    mean = vol.rolling(period).mean()
    std = vol.rolling(period).std().replace(0, np.nan)
    return (vol - mean) / std


def tag_market_regime(df: pd.DataFrame, ema_period: int = 200,
                      slope_window: int = 20) -> pd.Series:
    """Classify each bar as 'bull', 'bear', or 'sideways' based on 200-EMA slope."""
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    slope = (ema - ema.shift(slope_window)) / ema.shift(slope_window)
    regime = pd.Series("sideways", index=df.index)
    regime[slope > 0.02] = "bull"
    regime[slope < -0.02] = "bear"
    return regime


# ── Structure ────────────────────────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple[list, list]:
    """
    Swing detection using body (max/min of open,close), not wicks.
    Returns: swing_highs = [(idx, body_price, wick_price), ...]
    """
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df) - lookback):
        window_highs = df["high"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window_highs.max():
            body_top = max(df["open"].iloc[i], df["close"].iloc[i])
            neighbor_tops = [
                max(df["open"].iloc[j], df["close"].iloc[j])
                for j in range(max(0, i - lookback), min(len(df), i + lookback + 1))
                if j != i
            ]
            if not neighbor_tops or body_top >= max(neighbor_tops):
                swing_highs.append((i, float(body_top), float(df["high"].iloc[i])))

        window_lows = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["low"].iloc[i] == window_lows.min():
            body_bottom = min(df["open"].iloc[i], df["close"].iloc[i])
            neighbor_bottoms = [
                min(df["open"].iloc[j], df["close"].iloc[j])
                for j in range(max(0, i - lookback), min(len(df), i + lookback + 1))
                if j != i
            ]
            if not neighbor_bottoms or body_bottom <= min(neighbor_bottoms):
                swing_lows.append((i, float(body_bottom), float(df["low"].iloc[i])))

    return swing_highs, swing_lows


def detect_trend_at(df: pd.DataFrame, idx: int, swing_highs: list, swing_lows: list) -> str:
    recent_highs = [(i, bp) for i, bp, wp in swing_highs if i < idx][-3:]
    recent_lows = [(i, bp) for i, bp, wp in swing_lows if i < idx][-3:]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return "ranging"

    hh = all(recent_highs[j][1] <= recent_highs[j + 1][1] for j in range(len(recent_highs) - 1))
    hl = all(recent_lows[j][1] <= recent_lows[j + 1][1] for j in range(len(recent_lows) - 1))
    lh = all(recent_highs[j][1] >= recent_highs[j + 1][1] for j in range(len(recent_highs) - 1))
    ll = all(recent_lows[j][1] >= recent_lows[j + 1][1] for j in range(len(recent_lows) - 1))

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"

    if idx >= 20:
        ema = df["close"].iloc[idx - 20:idx].ewm(span=20).mean()
        slope = (ema.iloc[-1] - ema.iloc[-5]) / ema.iloc[-5] if len(ema) >= 5 else 0
        if slope > 0.002:
            return "bullish"
        elif slope < -0.002:
            return "bearish"

    return "ranging"


def check_bos(df: pd.DataFrame, idx: int, direction: str,
              swing_highs: list, swing_lows: list) -> bool:
    if direction == "bullish":
        prev_highs = [bp for i, bp, wp in swing_highs if i < idx]
        if prev_highs:
            window = df.iloc[max(0, idx - 3):idx + 1]
            return (window["close"] > prev_highs[-1]).any()
    else:
        prev_lows = [bp for i, bp, wp in swing_lows if i < idx]
        if prev_lows:
            window = df.iloc[max(0, idx - 3):idx + 1]
            return (window["close"] < prev_lows[-1]).any()
    return False


def detect_liquidity_levels(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> list[dict]:
    """
    Detect equal highs / equal lows (liquidity pools).
    When multiple swing points cluster at the same price -> stops are sitting there.
    """
    levels = []

    # Equal highs
    for i, (idx_a, bp_a, wp_a) in enumerate(swing_highs):
        cluster = [(idx_a, wp_a)]
        for j in range(i + 1, len(swing_highs)):
            idx_b, bp_b, wp_b = swing_highs[j]
            if abs(wp_a - wp_b) / wp_a <= EQL_TOLERANCE_PCT:
                cluster.append((idx_b, wp_b))
        if len(cluster) >= EQL_MIN_TOUCHES:
            avg_price = np.mean([p for _, p in cluster])
            levels.append({
                "type": "equal_highs",
                "price": float(avg_price),
                "touches": len(cluster),
                "first_idx": cluster[0][0],
                "last_idx": cluster[-1][0],
            })

    # Equal lows
    for i, (idx_a, bp_a, wp_a) in enumerate(swing_lows):
        cluster = [(idx_a, wp_a)]
        for j in range(i + 1, len(swing_lows)):
            idx_b, bp_b, wp_b = swing_lows[j]
            if abs(wp_a - wp_b) / wp_a <= EQL_TOLERANCE_PCT:
                cluster.append((idx_b, wp_b))
        if len(cluster) >= EQL_MIN_TOUCHES:
            avg_price = np.mean([p for _, p in cluster])
            levels.append({
                "type": "equal_lows",
                "price": float(avg_price),
                "touches": len(cluster),
                "first_idx": cluster[0][0],
                "last_idx": cluster[-1][0],
            })

    # Deduplicate close levels
    levels.sort(key=lambda l: l["price"])
    deduped = []
    for lv in levels:
        if deduped and abs(lv["price"] - deduped[-1]["price"]) / lv["price"] < EQL_TOLERANCE_PCT * 2:
            if lv["touches"] > deduped[-1]["touches"]:
                deduped[-1] = lv
        else:
            deduped.append(lv)

    return deduped


def check_liquidity_swept(df: pd.DataFrame, liq_levels: list[dict], zone: dict) -> bool:
    """
    Check if liquidity near a zone was swept before the zone became active.
    A demand zone is stronger if equal lows below it were swept (stops taken).
    A supply zone is stronger if equal highs above it were swept.
    """
    for lv in liq_levels:
        if zone["type"] == "demand" and lv["type"] == "equal_lows":
            if lv["price"] <= zone["top"] * 1.005 and lv["price"] >= zone["bottom"] * 0.99:
                sweep_window = df.iloc[lv["last_idx"]:]
                if not sweep_window.empty and (sweep_window["low"] < lv["price"]).any():
                    return True

        elif zone["type"] == "supply" and lv["type"] == "equal_highs":
            if lv["price"] >= zone["bottom"] * 0.995 and lv["price"] <= zone["top"] * 1.01:
                sweep_window = df.iloc[lv["last_idx"]:]
                if not sweep_window.empty and (sweep_window["high"] > lv["price"]).any():
                    return True

    return False


