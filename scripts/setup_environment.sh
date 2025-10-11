#!/bin/bash

# Setup script for the Coin Futures Trading Bot
# This script sets up the development environment and starts the system

set -e

echo "🚀 Setting up Coin Futures Trading Bot..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p notebooks

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating environment file..."
    cp config.env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
fi

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check TimescaleDB
if docker-compose exec timescaledb pg_isready -U tradingbot; then
    echo "✅ TimescaleDB is ready"
else
    echo "❌ TimescaleDB is not ready"
fi

# Check Redis
if docker-compose exec redis redis-cli ping | grep -q PONG; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check Kafka
if docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list &> /dev/null; then
    echo "✅ Kafka is ready"
else
    echo "❌ Kafka is not ready"
fi

# Run database migrations
echo "🗄️  Running database migrations..."
docker-compose exec trading-bot python -c "
import asyncio
from src.database.timescale import TimescaleDB
from src.config import Config

async def migrate():
    config = Config()
    db = TimescaleDB(config)
    await db.initialize()
    print('Database migration completed')

asyncio.run(migrate())
"

echo "🎉 Setup completed successfully!"
echo ""
echo "📊 Access the services:"
echo "  - Trading Dashboard: http://localhost:8050"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jupyter: http://localhost:8888"
echo ""
echo "📝 Next steps:"
echo "  1. Edit .env file with your Binance API keys"
echo "  2. Configure trading parameters in .env"
echo "  3. Start trading with: docker-compose up trading-bot"
echo ""
echo "🔧 Useful commands:"
echo "  - View logs: docker-compose logs -f trading-bot"
echo "  - Stop all: docker-compose down"
echo "  - Restart: docker-compose restart trading-bot"
