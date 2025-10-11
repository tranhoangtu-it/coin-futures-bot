"""
Module 5: Dashboard & Monitoring
Command center for end-to-end visibility of system performance and health.
Provides real-time KPIs, system health monitoring, and alerting.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.database.timescale import TimescaleDB
from src.database.redis_cache import RedisCache


@dataclass
class PerformanceMetrics:
    """Performance metrics for the trading system."""
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_holding_time: float


@dataclass
class SystemHealth:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    database_connections: int
    kafka_lag: float
    api_errors: int
    last_data_update: float
    active_orders: int
    positions_count: int


class MonitoringModule:
    """Monitoring and dashboard module."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        self.redis_cache = RedisCache(config)
        
        # Dashboard
        self.app = None
        self.dashboard_data = {}
        
        # Performance tracking
        self.performance_metrics = None
        self.system_health = None
        self.equity_curve = []
        self.trade_history = []
        
        # State
        self.running = False
    
    async def initialize(self):
        """Initialize the monitoring module."""
        self.logger.info("Initializing monitoring module...")
        
        # Initialize database connections
        await self.timescale_db.initialize()
        await self.redis_cache.initialize()
        
        # Initialize dashboard
        await self._initialize_dashboard()
        
        # Subscribe to system events
        await self.message_queue.subscribe(
            "orders",
            self._handle_order_event
        )
        await self.message_queue.subscribe(
            "alerts",
            self._handle_alert_event
        )
        
        self.logger.info("Monitoring module initialized")
    
    async def start(self):
        """Start the monitoring module."""
        self.logger.info("Starting monitoring module...")
        self.running = True
        
        # Start data collection loop
        asyncio.create_task(self._data_collection_loop())
        
        # Start dashboard update loop
        asyncio.create_task(self._dashboard_update_loop())
        
        # Start system health monitoring
        asyncio.create_task(self._system_health_loop())
        
        # Start dashboard server
        asyncio.create_task(self._run_dashboard())
        
        self.logger.info("Monitoring module started")
    
    async def stop(self):
        """Stop the monitoring module."""
        self.logger.info("Stopping monitoring module...")
        self.running = False
        
        # Close database connections
        await self.timescale_db.close()
        await self.redis_cache.close()
        
        self.logger.info("Monitoring module stopped")
    
    async def _initialize_dashboard(self):
        """Initialize the Dash dashboard."""
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Define dashboard layout
        self.app.layout = self._create_dashboard_layout()
        
        # Register callbacks
        self._register_callbacks()
    
    def _create_dashboard_layout(self):
        """Create the dashboard layout."""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Trading Bot Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # System Status Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.Div(id="system-status"),
                            html.Div(id="last-update")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Performance Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance Metrics"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4(id="total-pnl", className="text-center"),
                                    html.P("Total P&L", className="text-center text-muted")
                                ], width=2),
                                dbc.Col([
                                    html.H4(id="daily-pnl", className="text-center"),
                                    html.P("Daily P&L", className="text-center text-muted")
                                ], width=2),
                                dbc.Col([
                                    html.H4(id="sharpe-ratio", className="text-center"),
                                    html.P("Sharpe Ratio", className="text-center text-muted")
                                ], width=2),
                                dbc.Col([
                                    html.H4(id="max-drawdown", className="text-center"),
                                    html.P("Max Drawdown", className="text-center text-muted")
                                ], width=2),
                                dbc.Col([
                                    html.H4(id="win-rate", className="text-center"),
                                    html.P("Win Rate", className="text-center text-muted")
                                ], width=2),
                                dbc.Col([
                                    html.H4(id="total-trades", className="text-center"),
                                    html.P("Total Trades", className="text-center text-muted")
                                ], width=2)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Equity Curve"),
                        dbc.CardBody([
                            dcc.Graph(id="equity-curve-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trade Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="trade-distribution-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # System Health Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Health"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("CPU Usage"),
                                    html.Div(id="cpu-usage")
                                ], width=3),
                                dbc.Col([
                                    html.H5("Memory Usage"),
                                    html.Div(id="memory-usage")
                                ], width=3),
                                dbc.Col([
                                    html.H5("Database Connections"),
                                    html.Div(id="db-connections")
                                ], width=3),
                                dbc.Col([
                                    html.H5("API Errors"),
                                    html.Div(id="api-errors")
                                ], width=3)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Recent Trades Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody([
                            html.Div(id="recent-trades-table")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def _register_callbacks(self):
        """Register dashboard callbacks."""
        
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('last-update', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_system_status(n):
            status = "🟢 Running" if self.running else "🔴 Stopped"
            last_update = f"Last update: {datetime.now().strftime('%H:%M:%S')}"
            return status, last_update
        
        @self.app.callback(
            [Output('total-pnl', 'children'),
             Output('daily-pnl', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('max-drawdown', 'children'),
             Output('win-rate', 'children'),
             Output('total-trades', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_metrics(n):
            if not self.performance_metrics:
                return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
            
            total_pnl = f"${self.performance_metrics.total_pnl:,.2f}"
            daily_pnl = f"${self.performance_metrics.daily_pnl:,.2f}"
            sharpe = f"{self.performance_metrics.sharpe_ratio:.2f}"
            drawdown = f"{self.performance_metrics.max_drawdown:.2%}"
            win_rate = f"{self.performance_metrics.win_rate:.1%}"
            trades = f"{self.performance_metrics.total_trades}"
            
            return total_pnl, daily_pnl, sharpe, drawdown, win_rate, trades
        
        @self.app.callback(
            Output('equity-curve-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_equity_curve(n):
            if not self.equity_curve:
                return go.Figure()
            
            df = pd.DataFrame(self.equity_curve)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['equity'],
                mode='lines',
                name='Equity Curve',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('trade-distribution-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trade_distribution(n):
            if not self.trade_history:
                return go.Figure()
            
            df = pd.DataFrame(self.trade_history)
            
            # Create P&L distribution histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['pnl'],
                nbinsx=20,
                name='P&L Distribution',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Trade P&L Distribution",
                xaxis_title="P&L ($)",
                yaxis_title="Frequency"
            )
            
            return fig
        
        @self.app.callback(
            [Output('cpu-usage', 'children'),
             Output('memory-usage', 'children'),
             Output('db-connections', 'children'),
             Output('api-errors', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_system_health(n):
            if not self.system_health:
                return "N/A", "N/A", "N/A", "N/A"
            
            cpu = f"{self.system_health.cpu_usage:.1f}%"
            memory = f"{self.system_health.memory_usage:.1f}%"
            db_conn = f"{self.system_health.database_connections}"
            api_errors = f"{self.system_health.api_errors}"
            
            return cpu, memory, db_conn, api_errors
        
        @self.app.callback(
            Output('recent-trades-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_recent_trades(n):
            if not self.trade_history:
                return "No trades available"
            
            # Get last 10 trades
            recent_trades = self.trade_history[-10:]
            
            table_rows = []
            for trade in recent_trades:
                pnl_color = "text-success" if trade['pnl'] > 0 else "text-danger"
                table_rows.append(
                    dbc.Row([
                        dbc.Col(trade['symbol'], width=2),
                        dbc.Col(trade['side'], width=2),
                        dbc.Col(f"${trade['price']:.2f}", width=2),
                        dbc.Col(f"{trade['quantity']:.4f}", width=2),
                        dbc.Col([
                            html.Span(f"${trade['pnl']:.2f}", className=pnl_color)
                        ], width=2),
                        dbc.Col(datetime.fromtimestamp(trade['timestamp']).strftime('%H:%M:%S'), width=2)
                    ], className="mb-1")
                )
            
            return table_rows
    
    async def _data_collection_loop(self):
        """Collect performance data."""
        while self.running:
            try:
                # Calculate performance metrics
                await self._calculate_performance_metrics()
                
                # Update equity curve
                await self._update_equity_curve()
                
                # Update trade history
                await self._update_trade_history()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _dashboard_update_loop(self):
        """Update dashboard data."""
        while self.running:
            try:
                # Cache dashboard data
                await self.redis_cache.set_system_state({
                    'performance_metrics': self.performance_metrics.__dict__ if self.performance_metrics else {},
                    'system_health': self.system_health.__dict__ if self.system_health else {},
                    'equity_curve_length': len(self.equity_curve),
                    'trade_history_length': len(self.trade_history)
                })
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(30)
    
    async def _system_health_loop(self):
        """Monitor system health."""
        while self.running:
            try:
                await self._calculate_system_health()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in system health loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_dashboard(self):
        """Run the dashboard server."""
        try:
            self.app.run_server(
                host='0.0.0.0',
                port=8050,
                debug=False
            )
        except Exception as e:
            self.logger.error(f"Error running dashboard: {e}")
    
    async def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        try:
            # Get trade history
            trades = await self.timescale_db.get_trade_history(
                self.config.DEFAULT_SYMBOL,
                start_time=time.time() - 30 * 24 * 3600,  # Last 30 days
                limit=10000
            )
            
            if not trades:
                return
            
            # Calculate metrics
            pnl_values = [trade['pnl'] for trade in trades if trade['pnl'] is not None]
            if not pnl_values:
                return
            
            total_pnl = sum(pnl_values)
            daily_pnl = sum([pnl for trade in trades if trade['pnl'] is not None 
                           and trade['timestamp'] > time.time() - 24 * 3600])
            
            # Calculate drawdown
            cumulative_pnl = np.cumsum(pnl_values)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = cumulative_pnl - running_max
            max_drawdown = abs(np.min(drawdowns))
            current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
            
            # Calculate ratios
            returns = np.array(pnl_values)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            downside_returns = returns[returns < 0]
            sortino_ratio = (np.mean(returns) / np.std(downside_returns) * np.sqrt(252) 
                           if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0)
            
            calmar_ratio = (np.mean(returns) * 252 / max_drawdown 
                          if max_drawdown > 0 else 0)
            
            # Trade statistics
            winning_trades = [pnl for pnl in pnl_values if pnl > 0]
            losing_trades = [pnl for pnl in pnl_values if pnl < 0]
            
            win_rate = len(winning_trades) / len(pnl_values) if pnl_values else 0
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades)) 
                           if losing_trades and sum(losing_trades) != 0 else 0)
            
            average_win = np.mean(winning_trades) if winning_trades else 0
            average_loss = np.mean(losing_trades) if losing_trades else 0
            
            # Calculate average holding time
            holding_times = []
            for i in range(1, len(trades)):
                if trades[i]['side'] != trades[i-1]['side']:  # Position change
                    holding_time = trades[i]['timestamp'] - trades[i-1]['timestamp']
                    holding_times.append(holding_time)
            
            average_holding_time = np.mean(holding_times) if holding_times else 0
            
            self.performance_metrics = PerformanceMetrics(
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                average_holding_time=average_holding_time
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
    
    async def _update_equity_curve(self):
        """Update equity curve data."""
        try:
            # Get portfolio value over time
            trades = await self.timescale_db.get_trade_history(
                self.config.DEFAULT_SYMBOL,
                start_time=time.time() - 7 * 24 * 3600,  # Last 7 days
                limit=1000
            )
            
            if not trades:
                return
            
            # Calculate cumulative PnL
            initial_capital = 100000.0  # Starting capital
            cumulative_pnl = initial_capital
            equity_points = []
            
            for trade in trades:
                if trade['pnl'] is not None:
                    cumulative_pnl += trade['pnl']
                    equity_points.append({
                        'timestamp': datetime.fromtimestamp(trade['timestamp']),
                        'equity': cumulative_pnl
                    })
            
            self.equity_curve = equity_points
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")
    
    async def _update_trade_history(self):
        """Update trade history data."""
        try:
            trades = await self.timescale_db.get_trade_history(
                self.config.DEFAULT_SYMBOL,
                start_time=time.time() - 24 * 3600,  # Last 24 hours
                limit=1000
            )
            
            self.trade_history = trades
            
        except Exception as e:
            self.logger.error(f"Error updating trade history: {e}")
    
    async def _calculate_system_health(self):
        """Calculate system health metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Database connections (simplified)
            database_connections = 10  # Placeholder
            
            # API errors (from cache)
            api_errors = await self.redis_cache.increment_counter("api_errors", 0)
            
            # Active orders
            active_orders = len(await self.redis_cache.get_expiring_key("active_orders") or [])
            
            # Positions count
            positions = await self.timescale_db.get_positions()
            positions_count = len(positions)
            
            # Last data update
            last_data_update = time.time()  # Placeholder
            
            # Network latency (simplified)
            network_latency = 50  # Placeholder - 50ms
            
            # Kafka lag (simplified)
            kafka_lag = 0  # Placeholder
            
            self.system_health = SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                database_connections=database_connections,
                kafka_lag=kafka_lag,
                api_errors=api_errors,
                last_data_update=last_data_update,
                active_orders=active_orders,
                positions_count=positions_count
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating system health: {e}")
    
    async def _handle_order_event(self, message):
        """Handle order events for monitoring."""
        try:
            order_data = message.data
            order_id = order_data.get('order_id')
            status = order_data.get('status')
            
            # Update active orders count
            if status == "FILLED":
                await self.redis_cache.increment_counter("filled_orders", 1)
            elif status == "CANCELED":
                await self.redis_cache.increment_counter("canceled_orders", 1)
            elif status == "REJECTED":
                await self.redis_cache.increment_counter("rejected_orders", 1)
                await self.redis_cache.increment_counter("api_errors", 1)
            
        except Exception as e:
            self.logger.error(f"Error handling order event: {e}")
    
    async def _handle_alert_event(self, message):
        """Handle alert events."""
        try:
            alert_data = message.data
            alert_type = alert_data.get('alert_type')
            message_text = alert_data.get('message')
            severity = alert_data.get('severity', 'INFO')
            
            # Log alert
            if severity == 'CRITICAL':
                self.logger.critical(f"CRITICAL ALERT: {alert_type} - {message_text}")
            elif severity == 'WARNING':
                self.logger.warning(f"WARNING: {alert_type} - {message_text}")
            else:
                self.logger.info(f"INFO: {alert_type} - {message_text}")
            
            # Store alert in cache
            await self.redis_cache.set_alert(alert_type, message_text, severity)
            
        except Exception as e:
            self.logger.error(f"Error handling alert event: {e}")
