-- Initialize TimescaleDB for trading bot
-- This script sets up the required tables and hypertables

-- Create database if it doesn't exist
CREATE DATABASE trading_bot;

-- Connect to the trading_bot database
\c trading_bot;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create kline_data table
CREATE TABLE IF NOT EXISTS kline_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(20, 8) NOT NULL,
    high_price DECIMAL(20, 8) NOT NULL,
    low_price DECIMAL(20, 8) NOT NULL,
    close_price DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for kline_data
SELECT create_hypertable('kline_data', 'timestamp', if_not_exists => TRUE);

-- Create features table
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for features
SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Create trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) NOT NULL,
    commission_asset VARCHAR(10) NOT NULL,
    pnl DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create hypertable for trades
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Create positions table
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_kline_symbol_timestamp 
ON kline_data (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
ON features (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
ON trades (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_positions_symbol 
ON positions (symbol);

-- Create continuous aggregates for performance optimization
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_ohlc
WITH (timescaledb.continuous) AS
SELECT 
    symbol,
    time_bucket('1 day', timestamp) AS day,
    first(open_price, timestamp) AS open,
    max(high_price) AS high,
    min(low_price) AS low,
    last(close_price, timestamp) AS close,
    sum(volume) AS volume
FROM kline_data
GROUP BY symbol, day;

-- Create retention policy (keep data for 1 year)
SELECT add_retention_policy('kline_data', INTERVAL '1 year');
SELECT add_retention_policy('features', INTERVAL '1 year');
SELECT add_retention_policy('trades', INTERVAL '2 years');

-- Create compression policy
SELECT add_compression_policy('kline_data', INTERVAL '7 days');
SELECT add_compression_policy('features', INTERVAL '7 days');
SELECT add_compression_policy('trades', INTERVAL '30 days');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO tradingbot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO tradingbot;
