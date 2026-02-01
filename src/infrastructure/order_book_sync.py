"""
Local Order Book Synchronization for Binance Futures.

Implements the 5-step synchronization process:
1. Start WebSocket and buffer events
2. Get REST API snapshot
3. Drop stale events from buffer
4. Find connecting event and apply
5. Apply subsequent events in sequence

Follows @quant-analyst skill patterns for market microstructure.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import aiohttp
from loguru import logger

from .config import Settings, get_settings


@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book."""

    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        """Ensure Decimal types."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


@dataclass
class OrderBook:
    """Local order book state."""

    symbol: str
    last_update_id: int = 0
    bids: dict[Decimal, Decimal] = field(default_factory=dict)  # price -> quantity
    asks: dict[Decimal, Decimal] = field(default_factory=dict)  # price -> quantity

    def get_best_bid(self) -> OrderBookLevel | None:
        """Get the best (highest) bid."""
        if not self.bids:
            return None
        best_price = max(self.bids.keys())
        return OrderBookLevel(best_price, self.bids[best_price])

    def get_best_ask(self) -> OrderBookLevel | None:
        """Get the best (lowest) ask."""
        if not self.asks:
            return None
        best_price = min(self.asks.keys())
        return OrderBookLevel(best_price, self.asks[best_price])

    def get_mid_price(self) -> Decimal | None:
        """Calculate mid-price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

    def get_spread(self) -> Decimal | None:
        """Calculate bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    def get_bids(self, depth: int = 20) -> list[OrderBookLevel]:
        """Get top N bid levels sorted by price (descending)."""
        sorted_prices = sorted(self.bids.keys(), reverse=True)[:depth]
        return [OrderBookLevel(p, self.bids[p]) for p in sorted_prices]

    def get_asks(self, depth: int = 20) -> list[OrderBookLevel]:
        """Get top N ask levels sorted by price (ascending)."""
        sorted_prices = sorted(self.asks.keys())[:depth]
        return [OrderBookLevel(p, self.asks[p]) for p in sorted_prices]


@dataclass
class DepthUpdate:
    """Represents a depth update event from WebSocket."""

    symbol: str
    first_update_id: int  # U
    final_update_id: int  # u
    bids: list[tuple[str, str]]
    asks: list[tuple[str, str]]
    event_time: int


class OrderBookSynchronizer:
    """
    Synchronizes local order book with Binance Futures.

    Implements the official Binance synchronization algorithm:
    1. Buffer WebSocket depth events
    2. Fetch REST API snapshot
    3. Discard stale buffered events
    4. Find and apply the connecting event
    5. Continue processing sequential updates

    Example:
        ```python
        sync = OrderBookSynchronizer("BTCUSDT")
        await sync.initialize()

        # Process incoming WebSocket messages
        async def on_depth_update(data):
            await sync.process_depth_update(data)
        ```
    """

    def __init__(
        self,
        symbol: str,
        depth: int = 20,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize order book synchronizer.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT").
            depth: Order book depth to maintain.
            settings: Application settings.
        """
        self._symbol = symbol.upper()
        self._depth = depth
        self._settings = settings or get_settings()

        # Order book state
        self._order_book = OrderBook(symbol=self._symbol)
        self._synchronized = False

        # Event buffer for synchronization
        self._event_buffer: deque[DepthUpdate] = deque(
            maxlen=self._settings.websocket.snapshot_buffer_size
        )

        # Synchronization lock
        self._sync_lock = asyncio.Lock()

        # HTTP session for REST API
        self._session: aiohttp.ClientSession | None = None

    @property
    def order_book(self) -> OrderBook:
        """Get the current order book state."""
        return self._order_book

    @property
    def is_synchronized(self) -> bool:
        """Check if order book is synchronized."""
        return self._synchronized

    async def initialize(self) -> None:
        """Initialize the synchronizer and fetch initial snapshot."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        await self._fetch_snapshot()
        logger.info(f"Order book initialized for {self._symbol}")

    async def close(self) -> None:
        """Close the synchronizer and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

    async def process_depth_update(self, data: dict[str, Any]) -> None:
        """
        Process a depth update event from WebSocket.

        Args:
            data: Raw depth update data from WebSocket.
        """
        # Parse the update
        update = self._parse_depth_update(data)
        if update is None:
            return

        async with self._sync_lock:
            if not self._synchronized:
                # Buffer events while waiting for synchronization
                self._event_buffer.append(update)
                logger.debug(f"Buffered update: U={update.first_update_id}, u={update.final_update_id}")
                return

            # Validate sequence
            if update.first_update_id != self._order_book.last_update_id + 1:
                # Sequence gap detected - need to re-sync
                logger.warning(
                    f"Sequence gap detected: expected {self._order_book.last_update_id + 1}, "
                    f"got {update.first_update_id}"
                )
                self._synchronized = False
                self._event_buffer.clear()
                await self._fetch_snapshot()
                return

            # Apply the update
            self._apply_update(update)

    async def _fetch_snapshot(self) -> None:
        """Fetch order book snapshot from REST API."""
        if self._session is None:
            raise RuntimeError("Session not initialized")

        url = f"{self._settings.binance.rest_url}/fapi/v1/depth"
        params = {"symbol": self._symbol, "limit": 1000}

        try:
            async with self._session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

            # Parse snapshot
            last_update_id = data["lastUpdateId"]
            bids = {Decimal(p): Decimal(q) for p, q in data["bids"]}
            asks = {Decimal(p): Decimal(q) for p, q in data["asks"]}

            # Update order book
            self._order_book.bids = bids
            self._order_book.asks = asks
            self._order_book.last_update_id = last_update_id

            logger.info(f"Fetched snapshot: lastUpdateId={last_update_id}")

            # Process buffered events
            await self._process_buffer()

        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch snapshot: {e}")
            raise

    async def _process_buffer(self) -> None:
        """Process buffered events after getting snapshot."""
        snapshot_id = self._order_book.last_update_id

        # Find the first valid event
        # Drop events where u < lastUpdateId
        # First event should satisfy U <= lastUpdateId+1 <= u
        valid_events: list[DepthUpdate] = []
        found_connecting = False

        for event in self._event_buffer:
            if event.final_update_id < snapshot_id:
                # Stale event - discard
                continue

            if not found_connecting:
                # Look for connecting event: U <= lastUpdateId + 1 <= u
                if (
                    event.first_update_id <= snapshot_id + 1
                    and event.final_update_id >= snapshot_id + 1
                ):
                    found_connecting = True
                    valid_events.append(event)
                    logger.debug(
                        f"Found connecting event: U={event.first_update_id}, u={event.final_update_id}"
                    )
            else:
                # After connecting event, add all subsequent events
                valid_events.append(event)

        if not found_connecting:
            logger.warning("No connecting event found in buffer")
            self._event_buffer.clear()
            self._synchronized = False
            return

        # Apply valid events
        for event in valid_events:
            self._apply_update(event)

        self._event_buffer.clear()
        self._synchronized = True
        logger.info(f"Order book synchronized: lastUpdateId={self._order_book.last_update_id}")

    def _apply_update(self, update: DepthUpdate) -> None:
        """Apply a depth update to the order book."""
        # Process bids
        for price_str, qty_str in update.bids:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            if qty == 0:
                self._order_book.bids.pop(price, None)
            else:
                self._order_book.bids[price] = qty

        # Process asks
        for price_str, qty_str in update.asks:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            if qty == 0:
                self._order_book.asks.pop(price, None)
            else:
                self._order_book.asks[price] = qty

        self._order_book.last_update_id = update.final_update_id

    def _parse_depth_update(self, data: dict[str, Any]) -> DepthUpdate | None:
        """Parse raw WebSocket data into DepthUpdate."""
        try:
            # Handle combined stream format
            if "stream" in data:
                data = data["data"]

            # Check if this is a depth update
            if data.get("e") != "depthUpdate":
                return None

            return DepthUpdate(
                symbol=data["s"],
                first_update_id=data["U"],
                final_update_id=data["u"],
                bids=data["b"],
                asks=data["a"],
                event_time=data["E"],
            )
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to parse depth update: {e}")
            return None

    def calculate_obi(self, levels: int = 10) -> float:
        """
        Calculate Order Book Imbalance (OBI).

        OBI = (sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty))

        Args:
            levels: Number of price levels to consider.

        Returns:
            OBI value between -1 and 1.
        """
        bids = self.order_book.get_bids(levels)
        asks = self.order_book.get_asks(levels)

        bid_volume = sum(level.quantity for level in bids)
        ask_volume = sum(level.quantity for level in asks)

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        return float((bid_volume - ask_volume) / total_volume)

    def calculate_vamp(self) -> Decimal | None:
        """
        Calculate Volume Adjusted Mid-Price (VAMP).

        VAMP = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)

        Returns:
            VAMP value or None if order book is empty.
        """
        best_bid = self.order_book.get_best_bid()
        best_ask = self.order_book.get_best_ask()

        if not best_bid or not best_ask:
            return None

        total_qty = best_bid.quantity + best_ask.quantity
        if total_qty == 0:
            return None

        return (
            best_bid.price * best_ask.quantity + best_ask.price * best_bid.quantity
        ) / total_qty
