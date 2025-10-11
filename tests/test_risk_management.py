"""
Tests for risk management module.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.modules.risk_management import RiskManagementModule, RiskLevel, RiskCheckResult
from src.config import Config


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=Config)
    config.MAX_POSITION_SIZE = 0.1
    config.MAX_DAILY_DRAWDOWN = 0.05
    config.MAX_CORRELATION = 0.7
    config.RISK_PERCENTAGE = 0.02
    config.VAR_CONFIDENCE_LEVEL = 0.95
    config.DEFAULT_SYMBOL = "BTCUSDT"
    config.is_live_trading.return_value = False
    config.is_paper_trading.return_value = True
    return config


@pytest.fixture
def mock_message_queue():
    """Create mock message queue."""
    return Mock()


@pytest.fixture
async def risk_management_module(mock_config, mock_message_queue):
    """Create risk management module instance."""
    module = RiskManagementModule(mock_config, mock_message_queue)
    
    # Mock database connections
    module.timescale_db = Mock()
    module.redis_cache = Mock()
    
    # Mock Binance client
    module.client = Mock()
    
    await module.initialize()
    return module


@pytest.mark.asyncio
async def test_risk_check_approved(risk_management_module):
    """Test risk check approval for valid order."""
    # Set up portfolio value
    risk_management_module.portfolio_value = 100000.0
    
    # Test valid order
    result = await risk_management_module._check_risk(
        symbol="BTCUSDT",
        action="BUY",
        position_size=0.05,  # 5% position size
        confidence=0.8
    )
    
    assert result.approved is True
    assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    assert result.suggested_position_size > 0


@pytest.mark.asyncio
async def test_risk_check_rejected_oversized(risk_management_module):
    """Test risk check rejection for oversized position."""
    # Set up portfolio value
    risk_management_module.portfolio_value = 100000.0
    
    # Test oversized order
    result = await risk_management_module._check_risk(
        symbol="BTCUSDT",
        action="BUY",
        position_size=0.2,  # 20% position size (exceeds 10% limit)
        confidence=0.8
    )
    
    assert result.approved is False
    assert result.risk_level == RiskLevel.CRITICAL
    assert "exceeds maximum" in result.reason


@pytest.mark.asyncio
async def test_risk_check_rejected_zero_portfolio(risk_management_module):
    """Test risk check rejection for zero portfolio value."""
    # Set zero portfolio value
    risk_management_module.portfolio_value = 0.0
    
    # Test order with zero portfolio
    result = await risk_management_module._check_risk(
        symbol="BTCUSDT",
        action="BUY",
        position_size=0.05,
        confidence=0.8
    )
    
    assert result.approved is False
    assert result.risk_level == RiskLevel.CRITICAL
    assert "zero or negative" in result.reason


@pytest.mark.asyncio
async def test_volatility_adjusted_sizing(risk_management_module):
    """Test volatility-adjusted position sizing."""
    # Mock recent kline data
    mock_kline_data = [
        {'close': 100, 'high': 101, 'low': 99},
        {'close': 101, 'high': 102, 'low': 100},
        {'close': 102, 'high': 103, 'low': 101},
        {'close': 103, 'high': 104, 'low': 102},
        {'close': 104, 'high': 105, 'low': 103}
    ] * 20  # 100 data points
    
    risk_management_module.timescale_db.get_recent_klines = AsyncMock(
        return_value=mock_kline_data
    )
    
    # Test volatility-adjusted sizing
    suggested_size = await risk_management_module._calculate_volatility_adjusted_size(
        symbol="BTCUSDT",
        base_size=0.1,
        confidence=0.8
    )
    
    assert 0 <= suggested_size <= 0.1
    assert isinstance(suggested_size, float)


@pytest.mark.asyncio
async def test_correlation_risk_calculation(risk_management_module):
    """Test correlation risk calculation."""
    # Mock existing positions
    risk_management_module.positions = {
        "ETHUSDT": Mock(),
        "ADAUSDT": Mock()
    }
    
    # Mock price data for correlation calculation
    mock_kline_data = [
        {'close': 100 + i, 'high': 101 + i, 'low': 99 + i}
        for i in range(100)
    ]
    
    risk_management_module.timescale_db.get_recent_klines = AsyncMock(
        return_value=mock_kline_data
    )
    
    # Test correlation risk calculation
    correlation = await risk_management_module._calculate_correlation_risk("BTCUSDT")
    
    assert 0 <= correlation <= 1
    assert isinstance(correlation, float)


@pytest.mark.asyncio
async def test_risk_level_determination(risk_management_module):
    """Test risk level determination."""
    # Test different risk levels
    test_cases = [
        (0.05, [], RiskLevel.LOW),      # Small position, no warnings
        (0.08, ["High correlation"], RiskLevel.MEDIUM),  # Medium position, 1 warning
        (0.09, ["High correlation", "High VaR"], RiskLevel.HIGH),  # Large position, 2 warnings
        (0.0, [], RiskLevel.CRITICAL),  # No position
        (0.1, ["High correlation", "High VaR", "High volatility"], RiskLevel.CRITICAL)  # 3+ warnings
    ]
    
    for position_size, warnings, expected_level in test_cases:
        risk_level = risk_management_module._determine_risk_level(position_size, warnings)
        assert risk_level == expected_level


@pytest.mark.asyncio
async def test_position_update_from_order(risk_management_module):
    """Test position update from order execution."""
    # Mock order data
    order_data = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'position_size': 0.1,
        'price': 50000.0,
        'order_id': 'test_order_1'
    }
    
    # Mock database update
    risk_management_module.timescale_db.update_position = AsyncMock()
    
    # Test position update
    await risk_management_module._update_position_from_order(order_data)
    
    # Check that position was created
    assert 'BTCUSDT' in risk_management_module.positions
    position = risk_management_module.positions['BTCUSDT']
    assert position.symbol == 'BTCUSDT'
    assert position.side == 'BUY'
    assert position.size == 0.1
    assert position.entry_price == 50000.0
    
    # Check that database was updated
    risk_management_module.timescale_db.update_position.assert_called_once()


@pytest.mark.asyncio
async def test_emergency_close_all_positions(risk_management_module):
    """Test emergency close all positions."""
    # Set up some positions
    risk_management_module.positions = {
        'BTCUSDT': Mock(symbol='BTCUSDT', side='LONG', size=0.1),
        'ETHUSDT': Mock(symbol='ETHUSDT', side='SHORT', size=0.05)
    }
    
    # Mock message publishing
    risk_management_module.message_queue.publish = AsyncMock()
    
    # Test emergency close
    await risk_management_module._close_all_positions("Test emergency")
    
    # Check that close orders were published for each position
    assert risk_management_module.message_queue.publish.call_count == 2
    
    # Check the published messages
    calls = risk_management_module.message_queue.publish.call_args_list
    published_symbols = [call[0][2] for call in calls]  # Extract symbol from publish calls
    assert 'BTCUSDT' in published_symbols
    assert 'ETHUSDT' in published_symbols
