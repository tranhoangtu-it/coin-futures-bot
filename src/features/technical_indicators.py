"""
Advanced technical indicators for feature engineering.
Includes Ichimoku Cloud, SuperTrend, and normalized oscillator variants.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler


class TechnicalIndicators:
    """Advanced technical indicators calculator."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the technical indicators."""
        self.initialized = True
    
    async def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators."""
        if not self.initialized:
            await self.initialize()
        
        features = {}
        
        # Basic OHLCV data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Price-based indicators
        features.update(self._calculate_price_indicators(high, low, close))
        
        # Volume indicators
        features.update(self._calculate_volume_indicators(close, volume))
        
        # Momentum indicators
        features.update(self._calculate_momentum_indicators(high, low, close))
        
        # Trend indicators
        features.update(self._calculate_trend_indicators(high, low, close))
        
        # Volatility indicators
        features.update(self._calculate_volatility_indicators(high, low, close))
        
        # Ichimoku Cloud
        features.update(self._calculate_ichimoku(high, low, close))
        
        # SuperTrend
        features.update(self._calculate_supertrend(high, low, close))
        
        # Statistical features
        features.update(self._calculate_statistical_features(close))
        
        return features
    
    def _calculate_price_indicators(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray) -> Dict[str, float]:
        """Calculate price-based indicators."""
        features = {}
        
        # ATR (Average True Range)
        atr = talib.ATR(high, low, close, timeperiod=14)
        features['atr'] = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
        features['atr_normalized'] = float(atr[-1] / close[-1]) if close[-1] != 0 else 0.0
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        features['rsi'] = float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        features['stoch_k'] = float(slowk[-1]) if not np.isnan(slowk[-1]) else 50.0
        features['stoch_d'] = float(slowd[-1]) if not np.isnan(slowd[-1]) else 50.0
        
        # Williams %R
        willr = talib.WILLR(high, low, close, timeperiod=14)
        features['willr'] = float(willr[-1]) if not np.isnan(willr[-1]) else -50.0
        
        # CCI (Commodity Channel Index)
        cci = talib.CCI(high, low, close, timeperiod=14)
        features['cci'] = float(cci[-1]) if not np.isnan(cci[-1]) else 0.0
        
        # ADX (Average Directional Index)
        adx = talib.ADX(high, low, close, timeperiod=14)
        features['adx'] = float(adx[-1]) if not np.isnan(adx[-1]) else 0.0
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = float(macd[-1]) if not np.isnan(macd[-1]) else 0.0
        features['macd_signal'] = float(macdsignal[-1]) if not np.isnan(macdsignal[-1]) else 0.0
        features['macd_histogram'] = float(macdhist[-1]) if not np.isnan(macdhist[-1]) else 0.0
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        features['bb_upper'] = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else close[-1]
        features['bb_middle'] = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else close[-1]
        features['bb_lower'] = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else close[-1]
        features['bb_width'] = float((bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]) if bb_middle[-1] != 0 else 0.0
        features['bb_position'] = float((close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])) if bb_upper[-1] != bb_lower[-1] else 0.5
        
        return features
    
    def _calculate_volume_indicators(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based indicators."""
        features = {}
        
        # OBV (On-Balance Volume)
        obv = talib.OBV(close, volume)
        features['obv'] = float(obv[-1]) if not np.isnan(obv[-1]) else 0.0
        
        # Volume SMA
        volume_sma = talib.SMA(volume, timeperiod=20)
        features['volume_sma'] = float(volume_sma[-1]) if not np.isnan(volume_sma[-1]) else 0.0
        features['volume_ratio'] = float(volume[-1] / volume_sma[-1]) if volume_sma[-1] != 0 else 1.0
        
        # AD (Accumulation/Distribution)
        ad = talib.AD(high, low, close, volume)
        features['ad'] = float(ad[-1]) if not np.isnan(ad[-1]) else 0.0
        
        # MFI (Money Flow Index)
        mfi = talib.MFI(high, low, close, volume, timeperiod=14)
        features['mfi'] = float(mfi[-1]) if not np.isnan(mfi[-1]) else 50.0
        
        return features
    
    def _calculate_momentum_indicators(self, high: np.ndarray, low: np.ndarray, 
                                     close: np.ndarray) -> Dict[str, float]:
        """Calculate momentum indicators."""
        features = {}
        
        # ROC (Rate of Change)
        roc = talib.ROC(close, timeperiod=10)
        features['roc'] = float(roc[-1]) if not np.isnan(roc[-1]) else 0.0
        
        # MOM (Momentum)
        mom = talib.MOM(close, timeperiod=10)
        features['momentum'] = float(mom[-1]) if not np.isnan(mom[-1]) else 0.0
        
        # CMO (Chande Momentum Oscillator)
        cmo = talib.CMO(close, timeperiod=14)
        features['cmo'] = float(cmo[-1]) if not np.isnan(cmo[-1]) else 0.0
        
        return features
    
    def _calculate_trend_indicators(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray) -> Dict[str, float]:
        """Calculate trend indicators."""
        features = {}
        
        # SMA
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        sma_200 = talib.SMA(close, timeperiod=200)
        
        features['sma_20'] = float(sma_20[-1]) if not np.isnan(sma_20[-1]) else close[-1]
        features['sma_50'] = float(sma_50[-1]) if not np.isnan(sma_50[-1]) else close[-1]
        features['sma_200'] = float(sma_200[-1]) if not np.isnan(sma_200[-1]) else close[-1]
        
        # EMA
        ema_20 = talib.EMA(close, timeperiod=20)
        ema_50 = talib.EMA(close, timeperiod=50)
        
        features['ema_20'] = float(ema_20[-1]) if not np.isnan(ema_20[-1]) else close[-1]
        features['ema_50'] = float(ema_50[-1]) if not np.isnan(ema_50[-1]) else close[-1]
        
        # Trend strength
        features['price_above_sma20'] = float(close[-1] > sma_20[-1]) if not np.isnan(sma_20[-1]) else 0.0
        features['price_above_sma50'] = float(close[-1] > sma_50[-1]) if not np.isnan(sma_50[-1]) else 0.0
        features['sma20_above_sma50'] = float(sma_20[-1] > sma_50[-1]) if not np.isnan(sma_20[-1]) and not np.isnan(sma_50[-1]) else 0.0
        
        return features
    
    def _calculate_volatility_indicators(self, high: np.ndarray, low: np.ndarray, 
                                       close: np.ndarray) -> Dict[str, float]:
        """Calculate volatility indicators."""
        features = {}
        
        # NATR (Normalized Average True Range)
        natr = talib.NATR(high, low, close, timeperiod=14)
        features['natr'] = float(natr[-1]) if not np.isnan(natr[-1]) else 0.0
        
        # TRANGE (True Range)
        trange = talib.TRANGE(high, low, close)
        features['trange'] = float(trange[-1]) if not np.isnan(trange[-1]) else 0.0
        
        # Volatility of volatility
        if len(close) >= 20:
            returns = np.diff(np.log(close))
            vol_of_vol = np.std(returns[-20:])
            features['vol_of_vol'] = float(vol_of_vol) if not np.isnan(vol_of_vol) else 0.0
        
        return features
    
    def _calculate_ichimoku(self, high: np.ndarray, low: np.ndarray, 
                          close: np.ndarray) -> Dict[str, float]:
        """Calculate Ichimoku Cloud indicators."""
        features = {}
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = np.max(high[-9:]) if len(high) >= 9 else high[-1]
        tenkan_low = np.min(low[-9:]) if len(low) >= 9 else low[-1]
        tenkan = (tenkan_high + tenkan_low) / 2
        features['ichimoku_tenkan'] = float(tenkan)
        
        # Kijun-sen (Base Line)
        kijun_high = np.max(high[-26:]) if len(high) >= 26 else high[-1]
        kijun_low = np.min(low[-26:]) if len(low) >= 26 else low[-1]
        kijun = (kijun_high + kijun_low) / 2
        features['ichimoku_kijun'] = float(kijun)
        
        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan + kijun) / 2
        features['ichimoku_senkou_a'] = float(senkou_a)
        
        # Senkou Span B (Leading Span B)
        senkou_high = np.max(high[-52:]) if len(high) >= 52 else high[-1]
        senkou_low = np.min(low[-52:]) if len(low) >= 52 else low[-1]
        senkou_b = (senkou_high + senkou_low) / 2
        features['ichimoku_senkou_b'] = float(senkou_b)
        
        # Cloud position
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        features['price_above_cloud'] = float(close[-1] > cloud_top)
        features['price_below_cloud'] = float(close[-1] < cloud_bottom)
        features['price_in_cloud'] = float(cloud_bottom <= close[-1] <= cloud_top)
        
        return features
    
    def _calculate_supertrend(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, period: int = 10, multiplier: float = 3.0) -> Dict[str, float]:
        """Calculate SuperTrend indicator."""
        features = {}
        
        if len(close) < period:
            return features
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        # Calculate basic upper and lower bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Calculate SuperTrend
        supertrend = np.zeros_like(close)
        direction = np.ones_like(close)
        
        for i in range(1, len(close)):
            if close[i] <= lower_band[i-1]:
                direction[i] = -1
            elif close[i] >= upper_band[i-1]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
        
        features['supertrend'] = float(supertrend[-1]) if not np.isnan(supertrend[-1]) else close[-1]
        features['supertrend_direction'] = float(direction[-1])
        features['price_above_supertrend'] = float(close[-1] > supertrend[-1])
        
        return features
    
    def _calculate_statistical_features(self, close: np.ndarray) -> Dict[str, float]:
        """Calculate statistical features."""
        features = {}
        
        if len(close) < 20:
            return features
        
        # Hurst exponent (trend vs mean-reversion measure)
        if len(close) >= 50:
            returns = np.diff(np.log(close))
            hurst = self._calculate_hurst_exponent(returns)
            features['hurst_exponent'] = float(hurst) if not np.isnan(hurst) else 0.5
        
        # Entropy (market disorder)
        if len(close) >= 20:
            returns = np.diff(np.log(close))
            entropy = self._calculate_entropy(returns[-20:])
            features['entropy'] = float(entropy) if not np.isnan(entropy) else 0.0
        
        # Skewness and Kurtosis
        returns = np.diff(np.log(close))
        features['skewness'] = float(stats.skew(returns)) if not np.isnan(stats.skew(returns)) else 0.0
        features['kurtosis'] = float(stats.kurtosis(returns)) if not np.isnan(stats.kurtosis(returns)) else 0.0
        
        # Z-score
        if len(close) >= 20:
            z_score = (close[-1] - np.mean(close[-20:])) / np.std(close[-20:])
            features['z_score'] = float(z_score) if not np.isnan(z_score) else 0.0
        
        return features
    
    def _calculate_hurst_exponent(self, returns: np.ndarray) -> float:
        """Calculate Hurst exponent."""
        try:
            lags = range(2, min(20, len(returns) // 4))
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy."""
        try:
            hist, _ = np.histogram(data, bins=bins)
            hist = hist[hist > 0]  # Remove zero bins
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
        except:
            return 0.0
