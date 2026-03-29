"""Kraken Futures REST API client."""
from __future__ import annotations

import hashlib
import hmac
import base64
import time
import logging
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://futures.kraken.com"


class KrakenFuturesClient:
    """Thin wrapper around Kraken Futures REST API."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "sd-farm/1.0"})

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _sign(self, endpoint: str, post_data: str = "", nonce: str = "") -> str:
        # Kraken signs without the /derivatives prefix
        sign_endpoint = endpoint
        if sign_endpoint.startswith("/derivatives"):
            sign_endpoint = sign_endpoint[len("/derivatives"):]
        message = post_data + nonce + sign_endpoint
        sha256_hash = hashlib.sha256(message.encode("utf-8")).digest()
        secret_decoded = base64.b64decode(self.api_secret)
        signature = hmac.new(secret_decoded, sha256_hash, hashlib.sha512).digest()
        return base64.b64encode(signature).decode("utf-8")

    def _private_request(self, method: str, endpoint: str,
                         data: dict | None = None) -> dict:
        nonce = str(int(time.time() * 100_000_000))
        # GET: signature uses empty postData; params go in query string
        # POST: signature uses url-encoded body
        if method in ("POST", "PUT"):
            sign_data = urlencode(data) if data else ""
        else:
            sign_data = ""

        headers = {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "Authent": self._sign(endpoint, sign_data, nonce),
        }

        url = self.base_url + endpoint
        for attempt in range(3):
            try:
                if method == "GET":
                    resp = self.session.get(url, headers=headers, params=data, timeout=10)
                elif method == "PUT":
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = self.session.put(url, headers=headers, data=sign_data, timeout=10)
                else:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = self.session.post(url, headers=headers, data=sign_data, timeout=10)

                resp.raise_for_status()
                result = resp.json()

                if result.get("result") == "success":
                    return result
                error = result.get("error", result.get("errors", "unknown"))
                raise RuntimeError(f"Kraken API error: {error}")

            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    wait = [1, 3, 10][attempt]
                    log.warning(f"Request failed (attempt {attempt+1}): {e}. Retry in {wait}s")
                    time.sleep(wait)
                else:
                    raise

    def _public_request(self, endpoint: str, params: dict | None = None) -> dict:
        url = self.base_url + endpoint
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Account ───────────────────────────────────────────────────────────────

    def get_accounts(self) -> dict:
        return self._private_request("GET", "/derivatives/api/v3/accounts")

    def get_balance(self) -> float:
        """Get available USD balance for futures trading."""
        result = self.get_accounts()
        accounts = result.get("accounts", {})
        # Flex futures use the 'flex' account
        flex = accounts.get("flex", accounts.get("fi_xbtusd", {}))
        return float(flex.get("availableMargin", 0))

    # ── Positions ─────────────────────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        result = self._private_request("GET", "/derivatives/api/v3/openpositions")
        return result.get("openPositions", [])

    # ── Orders ────────────────────────────────────────────────────────────────

    def get_open_orders(self) -> list[dict]:
        result = self._private_request("GET", "/derivatives/api/v3/openorders")
        return result.get("openOrders", [])

    def send_order(self, symbol: str, side: str, size: float,
                   order_type: str = "lmt", limit_price: float | None = None,
                   stop_price: float | None = None, reduce_only: bool = False,
                   post_only: bool = False, cli_ord_id: str | None = None) -> dict:
        """Place an order on Kraken Futures.

        Args:
            symbol: e.g. "PF_SOLUSD"
            side: "buy" or "sell"
            size: contract size (USD notional for linear perps)
            order_type: "lmt", "mkt", "stp", "take_profit"
            limit_price: required for limit orders
            stop_price: required for stop orders
            reduce_only: only reduce existing position
            post_only: reject if would take liquidity
            cli_ord_id: optional client order ID for tracking
        """
        data = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": size,
        }
        if limit_price is not None:
            data["limitPrice"] = limit_price
        if stop_price is not None:
            data["stopPrice"] = stop_price
        if reduce_only:
            data["reduceOnly"] = "true"
        if post_only:
            data["postOnly"] = "true"
        if cli_ord_id:
            data["cliOrdId"] = cli_ord_id

        log.info(f"SEND ORDER: {side} {size} {symbol} @ {limit_price or stop_price} "
                 f"({order_type}) reduce={reduce_only} post={post_only}")
        return self._private_request("POST", "/derivatives/api/v3/sendorder", data)

    def cancel_order(self, order_id: str) -> dict:
        return self._private_request("POST", "/derivatives/api/v3/cancelorder",
                                     {"order_id": order_id})

    def cancel_all_orders(self, symbol: str | None = None) -> dict:
        data = {"symbol": symbol} if symbol else {}
        return self._private_request("POST", "/derivatives/api/v3/cancelallorders", data)

    # ── Market data ───────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        result = self._public_request("/derivatives/api/v3/tickers")
        for t in result.get("tickers", []):
            if t.get("symbol") == symbol:
                return t
        raise ValueError(f"Ticker not found: {symbol}")

    def get_instruments(self) -> list[dict]:
        result = self._public_request("/derivatives/api/v3/instruments")
        return result.get("instruments", [])

    def get_orderbook(self, symbol: str) -> dict:
        return self._public_request("/derivatives/api/v3/orderbook",
                                    {"symbol": symbol})

    # ── Leverage ─────────────────────────────────────────────────────────────

    def set_leverage(self, symbol: str, max_leverage: int) -> dict:
        """Set max leverage for a symbol (switches to isolated margin)."""
        return self._private_request(
            "PUT", "/derivatives/api/v3/leveragepreferences",
            {"symbol": symbol, "maxLeverage": max_leverage},
        )

    def get_leverage(self) -> dict:
        return self._private_request("GET", "/derivatives/api/v3/leveragepreferences")

    # ── Instrument specs ─────────────────────────────────────────────────────

    def get_instrument_specs(self, symbol: str) -> dict:
        """Get tick size and min order size for a symbol."""
        instruments = self.get_instruments()
        for inst in instruments:
            if inst.get("symbol") == symbol:
                return {
                    "tick_size": float(inst.get("tickSize", 0.01)),
                    "min_order_size": float(inst.get("minimumOrderSize", 1)),
                    "contract_value_trade_precision":
                        int(inst.get("contractValueTradePrecision", 0)),
                }
        raise ValueError(f"Instrument not found: {symbol}")

    # ── Fills ─────────────────────────────────────────────────────────────────

    def get_fills(self, last_fill_time: str | None = None) -> list[dict]:
        """Get recent fills. lastFillTime format: '2024-01-01T00:00:00.000Z'."""
        data = {}
        if last_fill_time:
            data["lastFillTime"] = last_fill_time
        result = self._private_request("GET", "/derivatives/api/v3/fills", data)
        return result.get("fills", [])
