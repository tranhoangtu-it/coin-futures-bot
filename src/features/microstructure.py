"""
Microstructure market features for order book and trade data analysis.
Includes Order Book Imbalance (OBI), Cumulative Volume Delta (CVD), and liquidity features.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class OrderBookSnapshot:
    """Order book snapshot data."""
    timestamp: float
    bids: List[List[float]]  # [[price, quantity], ...]
    asks: List[List[float]]  # [[price, quantity], ...]
    last_update_id: int


@dataclass
class TradeSnapshot:
    """Trade snapshot data."""
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int


class MicrostructureFeatures:
    """Microstructure features calculator."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        """Initialize the microstructure features."""
        self.initialized = True
    
    async def calculate_order_book_features(self, order_book_data: List[Any]) -> Dict[str, float]:
        """Calculate order book microstructure features."""
        if not self.initialized:
            await self.initialize()
        
        if not order_book_data:
            return {}
        
        features = {}
        
        # Convert to OrderBookSnapshot objects
        snapshots = []
        for data in order_book_data:
            if hasattr(data, 'data'):
                snapshot = OrderBookSnapshot(
                    timestamp=data.timestamp,
                    bids=data.data.get('bids', []),
                    asks=data.data.get('asks', []),
                    last_update_id=data.data.get('last_update_id', 0)
                )
                snapshots.append(snapshot)
        
        if not snapshots:
            return {}
        
        # Calculate features for the latest snapshot
        latest = snapshots[-1]
        
        # Order Book Imbalance (OBI) at multiple depth levels
        features.update(self._calculate_obi_features(latest))
        
        # Bid-Ask Spread
        features.update(self._calculate_spread_features(latest))
        
        # Market Depth
        features.update(self._calculate_depth_features(latest))
        
        # Order Book Pressure
        features.update(self._calculate_pressure_features(latest))
        
        # Price Impact
        features.update(self._calculate_price_impact_features(latest))
        
        return features
    
    async def calculate_trade_features(self, trade_data: List[Any]) -> Dict[str, float]:
        """Calculate trade microstructure features."""
        if not self.initialized:
            await self.initialize()
        
        if not trade_data:
            return {}
        
        features = {}
        
        # Convert to TradeSnapshot objects
        trades = []
        for data in trade_data:
            if hasattr(data, 'data'):
                trade = TradeSnapshot(
                    timestamp=data.timestamp,
                    price=data.data.get('price', 0),
                    quantity=data.data.get('quantity', 0),
                    is_buyer_maker=data.data.get('is_buyer_maker', False),
                    trade_id=data.data.get('trade_id', 0)
                )
                trades.append(trade)
        
        if not trades:
            return {}
        
        # Sort by timestamp
        trades.sort(key=lambda x: x.timestamp)
        
        # Cumulative Volume Delta (CVD)
        features.update(self._calculate_cvd_features(trades))
        
        # Trade Size Analysis
        features.update(self._calculate_trade_size_features(trades))
        
        # Trade Intensity
        features.update(self._calculate_trade_intensity_features(trades))
        
        # Price Impact from Trades
        features.update(self._calculate_trade_price_impact_features(trades))
        
        return features
    
    def _calculate_obi_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate Order Book Imbalance features."""
        features = {}
        
        if not snapshot.bids or not snapshot.asks:
            return features
        
        # OBI at different depth levels
        for depth in [5, 10, 20]:
            if len(snapshot.bids) >= depth and len(snapshot.asks) >= depth:
                bid_volume = sum(bid[1] for bid in snapshot.bids[:depth])
                ask_volume = sum(ask[1] for ask in snapshot.asks[:depth])
                
                if bid_volume + ask_volume > 0:
                    obi = (bid_volume - ask_volume) / (bid_volume + ask_volume)
                    features[f'obi_{depth}'] = float(obi)
                else:
                    features[f'obi_{depth}'] = 0.0
        
        # Weighted OBI (closer to mid-price has more weight)
        if snapshot.bids and snapshot.asks:
            mid_price = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
            
            weighted_bid_volume = 0
            weighted_ask_volume = 0
            
            for i, (price, quantity) in enumerate(snapshot.bids[:10]):
                weight = 1.0 / (i + 1)  # Closer to mid-price has higher weight
                weighted_bid_volume += quantity * weight
            
            for i, (price, quantity) in enumerate(snapshot.asks[:10]):
                weight = 1.0 / (i + 1)
                weighted_ask_volume += quantity * weight
            
            if weighted_bid_volume + weighted_ask_volume > 0:
                weighted_obi = (weighted_bid_volume - weighted_ask_volume) / (weighted_bid_volume + weighted_ask_volume)
                features['weighted_obi'] = float(weighted_obi)
            else:
                features['weighted_obi'] = 0.0
        
        return features
    
    def _calculate_spread_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate bid-ask spread features."""
        features = {}
        
        if not snapshot.bids or not snapshot.asks:
            return features
        
        best_bid = snapshot.bids[0][0]
        best_ask = snapshot.asks[0][0]
        
        # Absolute spread
        spread = best_ask - best_bid
        features['spread_absolute'] = float(spread)
        
        # Relative spread (as percentage of mid-price)
        mid_price = (best_bid + best_ask) / 2
        if mid_price > 0:
            features['spread_relative'] = float(spread / mid_price)
        else:
            features['spread_relative'] = 0.0
        
        # Spread volatility (if we have historical data)
        # This would require multiple snapshots, so we'll calculate it separately
        
        return features
    
    def _calculate_depth_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate market depth features."""
        features = {}
        
        if not snapshot.bids or not snapshot.asks:
            return features
        
        # Total bid and ask volume
        total_bid_volume = sum(bid[1] for bid in snapshot.bids)
        total_ask_volume = sum(ask[1] for ask in snapshot.asks)
        
        features['total_bid_volume'] = float(total_bid_volume)
        features['total_ask_volume'] = float(total_ask_volume)
        features['total_volume'] = float(total_bid_volume + total_ask_volume)
        
        # Volume imbalance
        if total_bid_volume + total_ask_volume > 0:
            features['volume_imbalance'] = float((total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume))
        else:
            features['volume_imbalance'] = 0.0
        
        # Depth at different price levels
        mid_price = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
        
        for level in [0.001, 0.005, 0.01]:  # 0.1%, 0.5%, 1% from mid-price
            bid_depth = sum(bid[1] for bid in snapshot.bids if bid[0] >= mid_price * (1 - level))
            ask_depth = sum(ask[1] for ask in snapshot.asks if ask[0] <= mid_price * (1 + level))
            
            features[f'bid_depth_{int(level*1000)}bps'] = float(bid_depth)
            features[f'ask_depth_{int(level*1000)}bps'] = float(ask_depth)
        
        return features
    
    def _calculate_pressure_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate order book pressure features."""
        features = {}
        
        if not snapshot.bids or not snapshot.asks:
            return features
        
        # Price levels
        bid_prices = [bid[0] for bid in snapshot.bids]
        ask_prices = [ask[0] for ask in snapshot.asks]
        
        # Price pressure (how close prices are to mid-price)
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Bid pressure (how close the best bid is to mid-price)
        if mid_price > 0:
            features['bid_pressure'] = float((mid_price - bid_prices[0]) / mid_price)
            features['ask_pressure'] = float((ask_prices[0] - mid_price) / mid_price)
        
        # Order book slope (how quickly prices change with depth)
        if len(bid_prices) >= 5:
            bid_slope = np.polyfit(range(5), bid_prices[:5], 1)[0]
            features['bid_slope'] = float(bid_slope)
        
        if len(ask_prices) >= 5:
            ask_slope = np.polyfit(range(5), ask_prices[:5], 1)[0]
            features['ask_slope'] = float(ask_slope)
        
        return features
    
    def _calculate_price_impact_features(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate price impact features."""
        features = {}
        
        if not snapshot.bids or not snapshot.asks:
            return features
        
        # Calculate price impact for different trade sizes
        mid_price = (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2
        
        for size in [1000, 5000, 10000]:  # Different trade sizes
            # Simulate market buy order
            remaining_size = size
            total_cost = 0
            
            for price, quantity in snapshot.asks:
                if remaining_size <= 0:
                    break
                
                trade_quantity = min(remaining_size, quantity)
                total_cost += trade_quantity * price
                remaining_size -= trade_quantity
            
            if remaining_size > 0:
                # Not enough liquidity
                features[f'price_impact_buy_{size}'] = float('inf')
            else:
                avg_price = total_cost / size
                price_impact = (avg_price - mid_price) / mid_price
                features[f'price_impact_buy_{size}'] = float(price_impact)
            
            # Simulate market sell order
            remaining_size = size
            total_revenue = 0
            
            for price, quantity in snapshot.bids:
                if remaining_size <= 0:
                    break
                
                trade_quantity = min(remaining_size, quantity)
                total_revenue += trade_quantity * price
                remaining_size -= trade_quantity
            
            if remaining_size > 0:
                features[f'price_impact_sell_{size}'] = float('inf')
            else:
                avg_price = total_revenue / size
                price_impact = (mid_price - avg_price) / mid_price
                features[f'price_impact_sell_{size}'] = float(price_impact)
        
        return features
    
    def _calculate_cvd_features(self, trades: List[TradeSnapshot]) -> Dict[str, float]:
        """Calculate Cumulative Volume Delta features."""
        features = {}
        
        if not trades:
            return features
        
        # Calculate CVD
        cvd = 0
        cvd_values = []
        
        for trade in trades:
            if trade.is_buyer_maker:  # Sell order
                cvd -= trade.quantity
            else:  # Buy order
                cvd += trade.quantity
            cvd_values.append(cvd)
        
        features['cvd_current'] = float(cvd)
        features['cvd_max'] = float(max(cvd_values)) if cvd_values else 0.0
        features['cvd_min'] = float(min(cvd_values)) if cvd_values else 0.0
        features['cvd_range'] = float(features['cvd_max'] - features['cvd_min'])
        
        # CVD trend
        if len(cvd_values) >= 10:
            recent_cvd = cvd_values[-10:]
            cvd_trend = np.polyfit(range(len(recent_cvd)), recent_cvd, 1)[0]
            features['cvd_trend'] = float(cvd_trend)
        
        return features
    
    def _calculate_trade_size_features(self, trades: List[TradeSnapshot]) -> Dict[str, float]:
        """Calculate trade size analysis features."""
        features = {}
        
        if not trades:
            return features
        
        quantities = [trade.quantity for trade in trades]
        prices = [trade.price for trade in trades]
        
        # Basic statistics
        features['avg_trade_size'] = float(np.mean(quantities))
        features['median_trade_size'] = float(np.median(quantities))
        features['std_trade_size'] = float(np.std(quantities))
        features['max_trade_size'] = float(np.max(quantities))
        features['min_trade_size'] = float(np.min(quantities))
        
        # Large trade detection
        large_trade_threshold = np.percentile(quantities, 95)
        large_trades = [q for q in quantities if q >= large_trade_threshold]
        features['large_trade_count'] = float(len(large_trades))
        features['large_trade_ratio'] = float(len(large_trades) / len(quantities))
        
        # Trade size distribution
        features['trade_size_skewness'] = float(self._calculate_skewness(quantities))
        features['trade_size_kurtosis'] = float(self._calculate_kurtosis(quantities))
        
        return features
    
    def _calculate_trade_intensity_features(self, trades: List[TradeSnapshot]) -> Dict[str, float]:
        """Calculate trade intensity features."""
        features = {}
        
        if not trades:
            return features
        
        # Trade frequency (trades per minute)
        time_span = trades[-1].timestamp - trades[0].timestamp
        if time_span > 0:
            features['trades_per_minute'] = float(len(trades) / (time_span / 60))
        else:
            features['trades_per_minute'] = 0.0
        
        # Volume per minute
        total_volume = sum(trade.quantity for trade in trades)
        if time_span > 0:
            features['volume_per_minute'] = float(total_volume / (time_span / 60))
        else:
            features['volume_per_minute'] = 0.0
        
        # Trade clustering (bursts of activity)
        if len(trades) >= 10:
            intervals = [trades[i].timestamp - trades[i-1].timestamp for i in range(1, len(trades))]
            avg_interval = np.mean(intervals)
            interval_std = np.std(intervals)
            
            if avg_interval > 0:
                features['trade_clustering'] = float(interval_std / avg_interval)
            else:
                features['trade_clustering'] = 0.0
        
        return features
    
    def _calculate_trade_price_impact_features(self, trades: List[TradeSnapshot]) -> Dict[str, float]:
        """Calculate price impact from trades."""
        features = {}
        
        if len(trades) < 2:
            return features
        
        # Price changes
        price_changes = []
        for i in range(1, len(trades)):
            price_change = (trades[i].price - trades[i-1].price) / trades[i-1].price
            price_changes.append(price_change)
        
        if price_changes:
            features['avg_price_change'] = float(np.mean(price_changes))
            features['std_price_change'] = float(np.std(price_changes))
            features['max_price_change'] = float(np.max(price_changes))
            features['min_price_change'] = float(np.min(price_changes))
        
        # Price impact by trade size
        if len(trades) >= 10:
            # Group trades by size
            quantities = [trade.quantity for trade in trades]
            median_size = np.median(quantities)
            
            large_trades = [trades[i] for i, q in enumerate(quantities) if q >= median_size]
            small_trades = [trades[i] for i, q in enumerate(quantities) if q < median_size]
            
            if len(large_trades) >= 2:
                large_price_changes = []
                for i in range(1, len(large_trades)):
                    price_change = (large_trades[i].price - large_trades[i-1].price) / large_trades[i-1].price
                    large_price_changes.append(price_change)
                
                if large_price_changes:
                    features['large_trade_price_impact'] = float(np.mean(large_price_changes))
            
            if len(small_trades) >= 2:
                small_price_changes = []
                for i in range(1, len(small_trades)):
                    price_change = (small_trades[i].price - small_trades[i-1].price) / small_trades[i-1].price
                    small_price_changes.append(price_change)
                
                if small_price_changes:
                    features['small_trade_price_impact'] = float(np.mean(small_price_changes))
        
        return features
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3
        return kurtosis
