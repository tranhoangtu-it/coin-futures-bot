@echo off
REM Setup script for the Coin Futures Trading Bot (Windows)
REM This script sets up the development environment and starts the system

echo 🚀 Setting up Coin Futures Trading Bot...

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist monitoring\grafana\dashboards mkdir monitoring\grafana\dashboards
if not exist monitoring\grafana\datasources mkdir monitoring\grafana\datasources
if not exist notebooks mkdir notebooks

REM Copy environment file if it doesn't exist
if not exist .env (
    echo 📋 Creating environment file...
    copy config.env.example .env
    echo ⚠️  Please edit .env file with your API keys and configuration
)

REM Build and start services
echo 🐳 Building and starting Docker services...
docker-compose up -d --build

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...

REM Check TimescaleDB
docker-compose exec timescaledb pg_isready -U tradingbot >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ TimescaleDB is ready
) else (
    echo ❌ TimescaleDB is not ready
)

REM Check Redis
docker-compose exec redis redis-cli ping | findstr PONG >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Redis is ready
) else (
    echo ❌ Redis is not ready
)

REM Check Kafka
docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Kafka is ready
) else (
    echo ❌ Kafka is not ready
)

REM Run database migrations
echo 🗄️  Running database migrations...
docker-compose exec trading-bot python -c "import asyncio; from src.database.timescale import TimescaleDB; from src.config import Config; asyncio.run(TimescaleDB(Config()).initialize()); print('Database migration completed')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📊 Access the services:
echo   - Trading Dashboard: http://localhost:8050
echo   - Grafana: http://localhost:3000 (admin/admin)
echo   - Prometheus: http://localhost:9090
echo   - Jupyter: http://localhost:8888
echo.
echo 📝 Next steps:
echo   1. Edit .env file with your Binance API keys
echo   2. Configure trading parameters in .env
echo   3. Start trading with: docker-compose up trading-bot
echo.
echo 🔧 Useful commands:
echo   - View logs: docker-compose logs -f trading-bot
echo   - Stop all: docker-compose down
echo   - Restart: docker-compose restart trading-bot

pause
