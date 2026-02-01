"""
Market Microstructure Features.

Computes advanced order book features including:
- Order Book Imbalance (OBI) at multiple levels
- Volume Adjusted Mid-Price (VAMP)
- Trade Flow Imbalance
- Liquidity metrics

Follows @quant-analyst skill patterns for market microstructure.
"""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class MicrostructureSnapshot:
    """Container for microstructure metrics at a point in time."""

    timestamp: pd.Timestamp
    symbol: str

    # Order Book Imbalance at different depths
    obi_1: float  # Level 1
    obi_5: float  # Level 5
    obi_10: float  # Level 10
    obi_20: float  # Level 20

    # Price metrics
    mid_price: float
    vamp: float
    spread: float
    spread_bps: float  # Spread in basis points

    # Volume metrics
    bid_volume_total: float
    ask_volume_total: float
    volume_imbalance: float


class MicrostructureFeatures:
    """
    Calculator for market microstructure features.

    Computes real-time and historical microstructure metrics from order book data.

    Example:
        ```python
        micro = MicrostructureFeatures()

        # Compute from raw order book
        snapshot = micro.compute_snapshot(
            symbol="BTCUSDT",
            bid_prices=[45000, 44999, 44998],
            bid_quantities=[1.0, 2.0, 3.0],
            ask_prices=[45001, 45002, 45003],
            ask_quantities=[1.5, 2.5, 3.5],
        )

        # Get rolling features
        features = micro.compute_rolling_features(obi_series, window=100)
        ```
    """

    def __init__(self) -> None:
        """Initialize microstructure feature calculator."""
        self._history: dict[str, list[MicrostructureSnapshot]] = {}

    def compute_obi(
        self,
        bid_quantities: list[float] | np.ndarray,
        ask_quantities: list[float] | np.ndarray,
        levels: int | None = None,
    ) -> float:
        """
        Calculate Order Book Imbalance (OBI).

        OBI = (Σ bid_qty - Σ ask_qty) / (Σ bid_qty + Σ ask_qty)

        Args:
            bid_quantities: Bid quantities at each level.
            ask_quantities: Ask quantities at each level.
            levels: Number of levels to include (None = all).

        Returns:
            OBI value between -1 and 1.
        """
        bids = np.array(bid_quantities[:levels] if levels else bid_quantities)
        asks = np.array(ask_quantities[:levels] if levels else ask_quantities)

        bid_volume = np.sum(bids)
        ask_volume = np.sum(asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return float((bid_volume - ask_volume) / total)

    def compute_vamp(
        self,
        best_bid_price: float,
        best_bid_qty: float,
        best_ask_price: float,
        best_ask_qty: float,
    ) -> float:
        """
        Calculate Volume Adjusted Mid-Price (VAMP).

        VAMP = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)

        This weights the mid-price towards the side with less liquidity,
        reflecting where the next trade is more likely to push the price.

        Args:
            best_bid_price: Best bid price.
            best_bid_qty: Best bid quantity.
            best_ask_price: Best ask price.
            best_ask_qty: Best ask quantity.

        Returns:
            VAMP value.
        """
        total_qty = best_bid_qty + best_ask_qty
        if total_qty == 0:
            return (best_bid_price + best_ask_price) / 2

        return (
            best_bid_price * best_ask_qty + best_ask_price * best_bid_qty
        ) / total_qty

    def compute_weighted_obi(
        self,
        bid_prices: list[float] | np.ndarray,
        bid_quantities: list[float] | np.ndarray,
        ask_prices: list[float] | np.ndarray,
        ask_quantities: list[float] | np.ndarray,
        mid_price: float,
        decay: float = 0.5,
    ) -> float:
        """
        Calculate distance-weighted Order Book Imbalance.

        Weights each level by distance from mid-price, giving more
        importance to levels closer to the market.

        Args:
            bid_prices: Bid prices at each level.
            bid_quantities: Bid quantities at each level.
            ask_prices: Ask prices at each level.
            ask_quantities: Ask quantities at each level.
            mid_price: Current mid-price.
            decay: Exponential decay factor for distance weighting.

        Returns:
            Weighted OBI value between -1 and 1.
        """
        bid_prices = np.array(bid_prices)
        bid_quantities = np.array(bid_quantities)
        ask_prices = np.array(ask_prices)
        ask_quantities = np.array(ask_quantities)

        # Distance weights (closer = higher weight)
        bid_distances = np.abs(bid_prices - mid_price)
        ask_distances = np.abs(ask_prices - mid_price)

        bid_weights = np.exp(-decay * bid_distances / mid_price * 10000)
        ask_weights = np.exp(-decay * ask_distances / mid_price * 10000)

        weighted_bid_vol = np.sum(bid_quantities * bid_weights)
        weighted_ask_vol = np.sum(ask_quantities * ask_weights)

        total = weighted_bid_vol + weighted_ask_vol
        if total == 0:
            return 0.0

        return float((weighted_bid_vol - weighted_ask_vol) / total)

    def compute_snapshot(
        self,
        symbol: str,
        bid_prices: list[float],
        bid_quantities: list[float],
        ask_prices: list[float],
        ask_quantities: list[float],
        timestamp: pd.Timestamp | None = None,
    ) -> MicrostructureSnapshot:
        """
        Compute all microstructure metrics for a single snapshot.

        Args:
            symbol: Trading pair symbol.
            bid_prices: Bid prices at each level.
            bid_quantities: Bid quantities at each level.
            ask_prices: Ask prices at each level.
            ask_quantities: Ask quantities at each level.
            timestamp: Snapshot timestamp.

        Returns:
            MicrostructureSnapshot with all metrics.
        """
        timestamp = timestamp or pd.Timestamp.utcnow()

        # Basic price metrics
        best_bid = bid_prices[0] if bid_prices else 0
        best_ask = ask_prices[0] if ask_prices else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0

        # Volume metrics
        bid_volume_total = sum(bid_quantities)
        ask_volume_total = sum(ask_quantities)
        total_volume = bid_volume_total + ask_volume_total
        volume_imbalance = (
            (bid_volume_total - ask_volume_total) / total_volume
            if total_volume > 0
            else 0
        )

        # OBI at different levels
        obi_1 = self.compute_obi(bid_quantities, ask_quantities, levels=1)
        obi_5 = self.compute_obi(bid_quantities, ask_quantities, levels=5)
        obi_10 = self.compute_obi(bid_quantities, ask_quantities, levels=10)
        obi_20 = self.compute_obi(bid_quantities, ask_quantities, levels=20)

        # VAMP
        best_bid_qty = bid_quantities[0] if bid_quantities else 0
        best_ask_qty = ask_quantities[0] if ask_quantities else 0
        vamp = self.compute_vamp(best_bid, best_bid_qty, best_ask, best_ask_qty)

        snapshot = MicrostructureSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            obi_1=obi_1,
            obi_5=obi_5,
            obi_10=obi_10,
            obi_20=obi_20,
            mid_price=mid_price,
            vamp=vamp,
            spread=spread,
            spread_bps=spread_bps,
            bid_volume_total=bid_volume_total,
            ask_volume_total=ask_volume_total,
            volume_imbalance=volume_imbalance,
        )

        # Store in history
        if symbol not in self._history:
            self._history[symbol] = []
        self._history[symbol].append(snapshot)

        # Keep only recent history (limit memory)
        if len(self._history[symbol]) > 10000:
            self._history[symbol] = self._history[symbol][-5000:]

        return snapshot

    def compute_rolling_features(
        self,
        obi_series: pd.Series,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Compute rolling statistics on OBI series.

        Args:
            obi_series: Time series of OBI values.
            windows: Rolling window sizes.

        Returns:
            DataFrame with rolling features.
        """
        windows = windows or [10, 50, 100, 200]
        features = pd.DataFrame(index=obi_series.index)

        for w in windows:
            features[f"obi_mean_{w}"] = obi_series.rolling(w).mean()
            features[f"obi_std_{w}"] = obi_series.rolling(w).std()
            features[f"obi_zscore_{w}"] = (
                obi_series - features[f"obi_mean_{w}"]
            ) / features[f"obi_std_{w}"]
            features[f"obi_skew_{w}"] = obi_series.rolling(w).skew()

        return features

    def get_history_df(self, symbol: str) -> pd.DataFrame:
        """
        Get historical snapshots as DataFrame.

        Args:
            symbol: Trading pair symbol.

        Returns:
            DataFrame with snapshot history.
        """
        if symbol not in self._history or not self._history[symbol]:
            return pd.DataFrame()

        data = [
            {
                "timestamp": s.timestamp,
                "obi_1": s.obi_1,
                "obi_5": s.obi_5,
                "obi_10": s.obi_10,
                "obi_20": s.obi_20,
                "mid_price": s.mid_price,
                "vamp": s.vamp,
                "spread": s.spread,
                "spread_bps": s.spread_bps,
                "volume_imbalance": s.volume_imbalance,
            }
            for s in self._history[symbol]
        ]

        return pd.DataFrame(data).set_index("timestamp")
