"""
Tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.02, 100)
    prices = [100.0]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 100)
    })
    
    data.set_index('timestamp', inplace=True)
    return data


@pytest.fixture
async def technical_indicators():
    """Create technical indicators instance."""
    indicators = TechnicalIndicators()
    await indicators.initialize()
    return indicators


@pytest.mark.asyncio
async def test_technical_indicators_calculation(technical_indicators, sample_data):
    """Test technical indicators calculation."""
    features = await technical_indicators.calculate_all(sample_data)
    
    # Check that features are calculated
    assert len(features) > 0
    
    # Check specific indicators
    assert 'rsi' in features
    assert 'atr' in features
    assert 'macd' in features
    assert 'bb_upper' in features
    assert 'bb_lower' in features
    
    # Check value ranges
    assert 0 <= features['rsi'] <= 100
    assert features['atr'] >= 0
    assert features['bb_upper'] > features['bb_lower']


@pytest.mark.asyncio
async def test_ichimoku_calculation(technical_indicators, sample_data):
    """Test Ichimoku Cloud calculation."""
    features = await technical_indicators.calculate_all(sample_data)
    
    # Check Ichimoku features
    ichimoku_features = [f for f in features.keys() if 'ichimoku' in f]
    assert len(ichimoku_features) > 0
    
    # Check specific Ichimoku components
    assert 'ichimoku_tenkan' in features
    assert 'ichimoku_kijun' in features
    assert 'ichimoku_senkou_a' in features
    assert 'ichimoku_senkou_b' in features


@pytest.mark.asyncio
async def test_supertrend_calculation(technical_indicators, sample_data):
    """Test SuperTrend calculation."""
    features = await technical_indicators.calculate_all(sample_data)
    
    # Check SuperTrend features
    assert 'supertrend' in features
    assert 'supertrend_direction' in features
    assert 'price_above_supertrend' in features
    
    # Check value ranges
    assert features['supertrend'] > 0
    assert features['supertrend_direction'] in [-1, 1]
    assert features['price_above_supertrend'] in [0, 1]


@pytest.mark.asyncio
async def test_statistical_features(technical_indicators, sample_data):
    """Test statistical features calculation."""
    features = await technical_indicators.calculate_all(sample_data)
    
    # Check statistical features
    assert 'skewness' in features
    assert 'kurtosis' in features
    assert 'z_score' in features
    
    # Check that values are reasonable
    assert not np.isnan(features['skewness'])
    assert not np.isnan(features['kurtosis'])
    assert not np.isnan(features['z_score'])


@pytest.mark.asyncio
async def test_empty_data_handling(technical_indicators):
    """Test handling of empty data."""
    empty_data = pd.DataFrame()
    features = await technical_indicators.calculate_all(empty_data)
    
    # Should return empty dict for empty data
    assert features == {}


@pytest.mark.asyncio
async def test_insufficient_data_handling(technical_indicators):
    """Test handling of insufficient data."""
    # Create data with only 5 rows (insufficient for most indicators)
    dates = pd.date_range('2023-01-01', periods=5, freq='1min')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    data.set_index('timestamp', inplace=True)
    
    features = await technical_indicators.calculate_all(data)
    
    # Should handle insufficient data gracefully
    assert isinstance(features, dict)
