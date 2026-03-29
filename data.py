"""Data fetching from Kraken Futures via ccxt."""

from __future__ import annotations

import logging
import time as _time

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]  # seconds between retries


def _fetch_with_retry(exchange, symbol, timeframe, **kwargs) -> list:
    """Fetch OHLCV with retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            log.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            _time.sleep(wait)
        except ccxt.ExchangeError as e:
            log.error(f"Exchange error (non-retryable): {e}")
            raise
    raise RuntimeError(f"Failed to fetch {symbol} {timeframe} after {MAX_RETRIES} retries")


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch recent OHLCV bars (single API call)."""
    exchange = ccxt.krakenfutures({"enableRateLimit": True})
    raw = _fetch_with_retry(exchange, symbol, timeframe, limit=limit)
    if not raw:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


def fetch_ohlcv_full(symbol: str, timeframe: str, days: int = 365,
                     batch: int = 1000) -> pd.DataFrame:
    """Fetch extended history via pagination (since parameter)."""
    exchange = ccxt.krakenfutures({"enableRateLimit": True})
    since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_rows = []
    calls = 0
    while since_ms < now_ms:
        raw = _fetch_with_retry(exchange, symbol, timeframe, since=since_ms, limit=batch)
        if not raw:
            break
        all_rows.extend(raw)
        last_ts = raw[-1][0]
        since_ms = last_ts + 1
        calls += 1
        if len(raw) < batch * 0.5:
            break
        _time.sleep(0.3)
    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol} {timeframe}")
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    log.debug(f"Fetched {len(df)} bars ({calls} API calls) | "
              f"{df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
    return df
