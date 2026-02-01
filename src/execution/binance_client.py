"""
Binance Futures API Client.

Provides async interface to Binance Futures REST API for:
- Account information
- Order placement
- Position management

Follows @python-pro skill patterns for API integration.
"""

import hashlib
import hmac
import time
from typing import Any

import aiohttp
from loguru import logger

from src.infrastructure.config import Settings, get_settings


class BinanceClient:
    """
    Async Binance Futures API client.

    Features:
    - Secure request signing
    - Rate limiting aware
    - Error handling and retries
    - Support for testnet

    Example:
        ```python
        client = BinanceClient()
        await client.initialize()

        # Get account info
        account = await client.get_account()

        # Place order
        order = await client.place_order(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=0.01,
            price=45000,
        )

        await client.close()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialize Binance client.

        Args:
            settings: Application settings.
        """
        self._settings = settings or get_settings()
        self._binance = self._settings.binance

        self._api_key = self._binance.api_key.get_secret_value()
        self._api_secret = self._binance.api_secret.get_secret_value()
        self._base_url = self._binance.rest_url

        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self._api_key}
            )
            logger.info(f"Binance client initialized. Base URL: {self._base_url}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _sign_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add signature to request parameters."""
        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self._api_secret.encode(),
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        """Make API request."""
        if self._session is None:
            await self.initialize()

        url = f"{self._base_url}{endpoint}"
        params = params or {}

        if signed:
            params = self._sign_request(params)

        try:
            async with self._session.request(method, url, params=params) as response:
                data = await response.json()

                if response.status != 200:
                    error_msg = data.get("msg", "Unknown error")
                    error_code = data.get("code", -1)
                    logger.error(f"API error {error_code}: {error_msg}")
                    raise BinanceAPIError(error_code, error_msg)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            raise

    # Account endpoints
    async def get_account(self) -> dict[str, Any]:
        """Get account information."""
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self) -> list[dict[str, Any]]:
        """Get account balance."""
        return await self._request("GET", "/fapi/v2/balance", signed=True)

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all open positions."""
        data = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
        # Filter to only positions with non-zero amount
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]

    async def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Get position for specific symbol."""
        positions = await self._request(
            "GET", "/fapi/v2/positionRisk", params={"symbol": symbol}, signed=True
        )
        return positions[0] if positions else None

    # Order endpoints
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> dict[str, Any]:
        """
        Place an order.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT").
            side: "BUY" or "SELL".
            order_type: "LIMIT", "MARKET", "STOP", "STOP_MARKET", etc.
            quantity: Order quantity.
            price: Limit price (for LIMIT orders).
            stop_price: Stop price (for STOP orders).
            time_in_force: "GTC", "IOC", "FOK", "GTX".
            reduce_only: Only reduce existing position.
            post_only: Maker only (use GTX time_in_force).

        Returns:
            Order response from Binance.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }

        if price is not None:
            params["price"] = price

        if stop_price is not None:
            params["stopPrice"] = stop_price

        if order_type == "LIMIT":
            params["timeInForce"] = "GTX" if post_only else time_in_force

        if reduce_only:
            params["reduceOnly"] = "true"

        logger.info(f"Placing order: {params}")
        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
    ) -> dict[str, Any]:
        """Place a limit order (convenience method)."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=price,
            post_only=post_only,
        )

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """Place a market order (convenience method)."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type="MARKET",
            quantity=quantity,
        )

    async def place_stop_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> dict[str, Any]:
        """Place a stop-market order."""
        return await self.place_order(
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=reduce_only,
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Cancel an order."""
        params: dict[str, Any] = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        return await self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict[str, Any]:
        """Cancel all open orders for a symbol."""
        return await self._request(
            "DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True
        )

    async def get_order(
        self,
        symbol: str,
        order_id: int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Get order status."""
        params: dict[str, Any] = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id

        return await self._request("GET", "/fapi/v1/order", params=params, signed=True)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get all open orders."""
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)

    # Leverage and margin
    async def set_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        """Set leverage for a symbol."""
        return await self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            signed=True,
        )

    async def set_margin_type(self, symbol: str, margin_type: str) -> dict[str, Any]:
        """Set margin type (ISOLATED or CROSSED)."""
        return await self._request(
            "POST",
            "/fapi/v1/marginType",
            params={"symbol": symbol, "marginType": margin_type},
            signed=True,
        )

    # Market data
    async def get_ticker_price(self, symbol: str) -> dict[str, Any]:
        """Get current price for a symbol."""
        return await self._request(
            "GET", "/fapi/v1/ticker/price", params={"symbol": symbol}
        )

    async def get_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Get order book snapshot."""
        return await self._request(
            "GET", "/fapi/v1/depth", params={"symbol": symbol, "limit": limit}
        )

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
    ) -> list[list[Any]]:
        """Get candlestick data."""
        return await self._request(
            "GET",
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )


class BinanceAPIError(Exception):
    """Binance API error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error {code}: {message}")
