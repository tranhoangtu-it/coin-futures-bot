"""
Module 2: AI Core - Multi-model Architecture
3-layer architecture: Market Regime Filter, Alpha Models, Reinforcement Learning Agent
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

from src.config import Config
from src.core.message_queue import MessageQueue, MessageType
from src.database.timescale import TimescaleDB
from src.database.redis_cache import RedisCache


class MarketRegime(Enum):
    """Market regime types."""
    BULL_VOLATILE = "bull_volatile"
    BULL_STABLE = "bull_stable"
    BEAR_VOLATILE = "bear_volatile"
    BEAR_STABLE = "bear_stable"
    SIDEWAYS_COMPRESSION = "sideways_compression"
    SIDEWAYS_EXPANSION = "sideways_expansion"


@dataclass
class AlphaPrediction:
    """Alpha model prediction."""
    model_name: str
    prediction: float
    confidence: float
    features_used: List[str]


@dataclass
class TradingSignal:
    """Trading signal from AI Core."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: str
    market_regime: MarketRegime
    alpha_predictions: List[AlphaPrediction]


class MarketRegimeDetector:
    """Layer 1: Market Regime Filter using GMM."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gmm = None
        self.scaler = StandardScaler()
        self.regime_labels = [
            MarketRegime.BULL_VOLATILE,
            MarketRegime.BULL_STABLE,
            MarketRegime.BEAR_VOLATILE,
            MarketRegime.BEAR_STABLE,
            MarketRegime.SIDEWAYS_COMPRESSION,
            MarketRegime.SIDEWAYS_EXPANSION
        ]
        self.trained = False
    
    async def train(self, features_df: pd.DataFrame):
        """Train the market regime detector."""
        self.logger.info("Training market regime detector...")
        
        # Select features for regime detection
        regime_features = [
            'atr_normalized', 'adx', 'vol_of_vol', 'spread_relative',
            'rsi', 'bb_width', 'volume_ratio', 'cvd_trend'
        ]
        
        # Filter available features
        available_features = [f for f in regime_features if f in features_df.columns]
        
        if len(available_features) < 3:
            self.logger.warning("Not enough features for regime detection")
            return
        
        # Prepare data
        X = features_df[available_features].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train GMM
        self.gmm = GaussianMixture(n_components=len(self.regime_labels), random_state=42)
        self.gmm.fit(X_scaled)
        
        self.trained = True
        self.logger.info("Market regime detector trained successfully")
    
    async def detect_regime(self, features: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect current market regime."""
        if not self.trained or self.gmm is None:
            return MarketRegime.SIDEWAYS_COMPRESSION, 0.5
        
        try:
            # Prepare features
            regime_features = [
                'atr_normalized', 'adx', 'vol_of_vol', 'spread_relative',
                'rsi', 'bb_width', 'volume_ratio', 'cvd_trend'
            ]
            
            feature_vector = []
            for feature in regime_features:
                feature_vector.append(features.get(feature, 0.0))
            
            # Scale features
            X_scaled = self.scaler.transform([feature_vector])
            
            # Predict regime
            regime_probs = self.gmm.predict_proba(X_scaled)[0]
            regime_idx = np.argmax(regime_probs)
            confidence = regime_probs[regime_idx]
            
            return self.regime_labels[regime_idx], float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS_COMPRESSION, 0.5


class AlphaModel:
    """Base class for alpha models."""
    
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.trained = False
    
    async def train(self, features_df: pd.DataFrame, target: pd.Series):
        """Train the alpha model."""
        raise NotImplementedError
    
    async def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction and return (prediction, confidence)."""
        raise NotImplementedError


class TransformerAlphaModel(AlphaModel):
    """Transformer-based alpha model for breakout patterns."""
    
    def __init__(self, config: Config):
        super().__init__("transformer_breakout", config)
        self.sequence_length = 60
        self.feature_dim = 20
    
    async def train(self, features_df: pd.DataFrame, target: pd.Series):
        """Train the transformer model."""
        self.logger.info(f"Training {self.name} model...")
        
        # Prepare sequences
        sequences, targets = self._prepare_sequences(features_df, target)
        
        if len(sequences) < 100:
            self.logger.warning(f"Not enough data for {self.name}")
            return
        
        # Create model
        self.model = TransformerPredictor(
            input_dim=self.feature_dim,
            d_model=64,
            nhead=4,
            num_layers=3,
            sequence_length=self.sequence_length
        )
        
        # Train model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(sequences), 32):
                batch_sequences = torch.FloatTensor(sequences[i:i+32])
                batch_targets = torch.FloatTensor(targets[i:i+32])
                
                optimizer.zero_grad()
                predictions = self.model(batch_sequences)
                loss = criterion(predictions.squeeze(), batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss/len(sequences)*32:.6f}")
        
        self.trained = True
        self.logger.info(f"{self.name} model trained successfully")
    
    async def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction using transformer."""
        if not self.trained or self.model is None:
            return 0.0, 0.0
        
        try:
            # This is a simplified version - in practice, you'd need historical sequences
            # For now, return a dummy prediction
            return 0.0, 0.0
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            return 0.0, 0.0
    
    def _prepare_sequences(self, features_df: pd.DataFrame, target: pd.Series):
        """Prepare sequences for transformer training."""
        # Select features
        feature_cols = [col for col in features_df.columns if col != 'timestamp'][:self.feature_dim]
        
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features_df)):
            sequence = features_df[feature_cols].iloc[i-self.sequence_length:i].values
            target_val = target.iloc[i]
            
            sequences.append(sequence)
            targets.append(target_val)
        
        return np.array(sequences), np.array(targets)


class LightGBMAlphaModel(AlphaModel):
    """LightGBM alpha model for microstructure features."""
    
    def __init__(self, config: Config):
        super().__init__("lightgbm_microstructure", config)
        self.feature_importance = {}
    
    async def train(self, features_df: pd.DataFrame, target: pd.Series):
        """Train the LightGBM model."""
        self.logger.info(f"Training {self.name} model...")
        
        # Select features
        feature_cols = [col for col in features_df.columns if col != 'timestamp']
        X = features_df[feature_cols].fillna(0)
        y = target.fillna(0)
        
        if len(X) < 100:
            self.logger.warning(f"Not enough data for {self.name}")
            return
        
        # Train model
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        )
        
        self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        self.trained = True
        self.logger.info(f"{self.name} model trained successfully")
    
    async def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction using LightGBM."""
        if not self.trained or self.model is None:
            return 0.0, 0.0
        
        try:
            # Prepare features
            feature_vector = []
            feature_names = list(self.feature_importance.keys())
            
            for feature_name in feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            
            # Calculate confidence based on feature importance and prediction magnitude
            confidence = min(abs(prediction) * 0.1, 1.0)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} prediction: {e}")
            return 0.0, 0.0


class TradingEnvironment(gym.Env):
    """Trading environment for reinforcement learning."""
    
    def __init__(self, features_df: pd.DataFrame, config: Config):
        super().__init__()
        self.config = config
        self.features_df = features_df
        self.current_step = 0
        self.max_steps = len(features_df) - 1
        
        # Action space: [-2, -1, 0, 1, 2] (Sell Strong, Sell, Hold, Buy, Buy Strong)
        self.action_space = gym.spaces.Discrete(5)
        
        # State space: normalized features + position info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        
        # Position tracking
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        return self._get_observation(), {}
    
    def step(self, action):
        """Take a step in the environment."""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, {}
        
        # Get current market data
        current_data = self.features_df.iloc[self.current_step]
        current_price = current_data['close']
        
        # Execute action
        action_intensity = action - 2  # Convert to [-2, -1, 0, 1, 2]
        position_change = action_intensity * 0.1  # 10% position change per action
        
        # Update position
        old_position = self.position
        self.position = max(-1.0, min(1.0, self.position + position_change))
        
        # Calculate PnL
        if old_position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            self.unrealized_pnl = old_position * price_change
        
        # Update entry price if position changed
        if self.position != 0 and old_position == 0:
            self.entry_price = current_price
        elif self.position == 0 and old_position != 0:
            self.total_pnl += self.unrealized_pnl
            self.unrealized_pnl = 0.0
            self.trade_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(current_data)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        """Get current observation."""
        if self.current_step >= len(self.features_df):
            return np.zeros(30, dtype=np.float32)
        
        current_data = self.features_df.iloc[self.current_step]
        
        # Normalize features
        features = []
        feature_cols = [col for col in current_data.index if col not in ['timestamp', 'close']]
        
        for col in feature_cols[:25]:  # Use top 25 features
            value = current_data[col] if not pd.isna(current_data[col]) else 0.0
            features.append(float(value))
        
        # Pad with zeros if needed
        while len(features) < 25:
            features.append(0.0)
        
        # Add position info
        features.extend([
            self.position,
            self.unrealized_pnl,
            self.total_pnl,
            self.trade_count / 100.0,  # Normalize trade count
            current_data['close'] / 100000.0  # Normalize price
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_reward(self, current_data):
        """Calculate reward for current step."""
        # Base reward from PnL
        reward = self.unrealized_pnl * 100
        
        # Transaction cost penalty
        if self.position != 0:
            reward -= 0.001  # 0.1% transaction cost
        
        # Holding time penalty for losing positions
        if self.position != 0 and self.unrealized_pnl < 0:
            reward -= 0.0001  # Small penalty for holding losing positions
        
        return reward


class TransformerPredictor(nn.Module):
    """Transformer model for price prediction."""
    
    def __init__(self, input_dim, d_model, nhead, num_layers, sequence_length):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(sequence_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.dropout(x)
        x = self.output_projection(x)
        
        return x


class AICoreModule:
    """AI Core module with 3-layer architecture."""
    
    def __init__(self, config: Config, message_queue: MessageQueue):
        self.config = config
        self.message_queue = message_queue
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.regime_detector = MarketRegimeDetector(config)
        self.alpha_models = {}
        self.rl_agent = None
        
        # Database connections
        self.timescale_db = TimescaleDB(config)
        self.redis_cache = RedisCache(config)
        
        # State
        self.running = False
        self.last_signal_time = {}
    
    async def initialize(self):
        """Initialize the AI Core module."""
        self.logger.info("Initializing AI Core module...")
        
        # Initialize database connections
        await self.timescale_db.initialize()
        await self.redis_cache.initialize()
        
        # Initialize alpha models
        self.alpha_models = {
            'transformer_breakout': TransformerAlphaModel(config),
            'lightgbm_microstructure': LightGBMAlphaModel(config)
        }
        
        # Subscribe to market data
        await self.message_queue.subscribe(
            f"market_data.{config.DEFAULT_SYMBOL}",
            self._handle_market_data
        )
        
        self.logger.info("AI Core module initialized")
    
    async def start(self):
        """Start the AI Core module."""
        self.logger.info("Starting AI Core module...")
        self.running = True
        
        # Start training loop
        asyncio.create_task(self._training_loop())
        
        # Start prediction loop
        asyncio.create_task(self._prediction_loop())
        
        self.logger.info("AI Core module started")
    
    async def stop(self):
        """Stop the AI Core module."""
        self.logger.info("Stopping AI Core module...")
        self.running = False
        
        # Close database connections
        await self.timescale_db.close()
        await self.redis_cache.close()
        
        self.logger.info("AI Core module stopped")
    
    async def _handle_market_data(self, message):
        """Handle incoming market data."""
        try:
            symbol = message.data.get('symbol', self.config.DEFAULT_SYMBOL)
            
            # Trigger prediction if we have new data
            await self._generate_signal(symbol)
            
        except Exception as e:
            self.logger.error(f"Error handling market data: {e}")
    
    async def _training_loop(self):
        """Periodic training loop."""
        while self.running:
            try:
                # Train models every hour
                await asyncio.sleep(self.config.MODEL_UPDATE_FREQUENCY)
                
                for symbol in self.config.get_symbols():
                    await self._train_models(symbol)
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(60)
    
    async def _prediction_loop(self):
        """Periodic prediction loop."""
        while self.running:
            try:
                # Generate signals every minute
                await asyncio.sleep(60)
                
                for symbol in self.config.get_symbols():
                    await self._generate_signal(symbol)
                
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)
    
    async def _train_models(self, symbol: str):
        """Train all models for a symbol."""
        try:
            # Get historical data
            kline_data = await self.timescale_db.get_recent_klines(symbol, limit=5000)
            
            if len(kline_data) < 1000:
                self.logger.warning(f"Not enough data for training {symbol}")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(kline_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Calculate features
            from src.features.technical_indicators import TechnicalIndicators
            from src.features.microstructure import MicrostructureFeatures
            
            tech_indicators = TechnicalIndicators()
            await tech_indicators.initialize()
            
            features = await tech_indicators.calculate_all(df)
            
            # Create features DataFrame
            features_df = pd.DataFrame([features])
            
            # Calculate target (future price change)
            df['future_return'] = df['close'].pct_change(5).shift(-5)  # 5-period forward return
            target = df['future_return'].dropna()
            
            # Align features and target
            min_len = min(len(features_df), len(target))
            if min_len < 100:
                return
            
            features_df = features_df.iloc[:min_len]
            target = target.iloc[:min_len]
            
            # Train regime detector
            await self.regime_detector.train(features_df)
            
            # Train alpha models
            for model in self.alpha_models.values():
                await model.train(features_df, target)
            
            self.logger.info(f"Models trained for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}")
    
    async def _generate_signal(self, symbol: str):
        """Generate trading signal for a symbol."""
        try:
            # Check if we should generate a new signal
            current_time = time.time()
            if symbol in self.last_signal_time:
                if current_time - self.last_signal_time[symbol] < 60:  # 1 minute cooldown
                    return
            
            # Get latest features
            features = await self.redis_cache.get_latest_features(symbol)
            if not features:
                return
            
            # Detect market regime
            regime, regime_confidence = await self.regime_detector.detect_regime(features)
            
            # Get alpha predictions
            alpha_predictions = []
            for model_name, model in self.alpha_models.items():
                prediction, confidence = await model.predict(features)
                alpha_predictions.append(AlphaPrediction(
                    model_name=model_name,
                    prediction=prediction,
                    confidence=confidence,
                    features_used=list(features.keys())
                ))
            
            # Generate signal based on regime and alpha predictions
            signal = await self._create_trading_signal(
                symbol, regime, regime_confidence, alpha_predictions, features
            )
            
            # Publish signal
            await self.message_queue.publish_signal(symbol, signal.__dict__)
            
            # Cache signal
            await self.redis_cache.set_trading_signal(symbol, signal.__dict__)
            
            self.last_signal_time[symbol] = current_time
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
    
    async def _create_trading_signal(self, symbol: str, regime: MarketRegime, 
                                   regime_confidence: float, alpha_predictions: List[AlphaPrediction],
                                   features: Dict[str, float]) -> TradingSignal:
        """Create trading signal based on regime and alpha predictions."""
        
        # Calculate weighted prediction
        total_weight = 0
        weighted_prediction = 0
        
        for pred in alpha_predictions:
            weight = pred.confidence
            weighted_prediction += pred.prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
        else:
            weighted_prediction = 0
        
        # Determine action based on regime and prediction
        if regime in [MarketRegime.BULL_VOLATILE, MarketRegime.BULL_STABLE]:
            if weighted_prediction > 0.1:
                action = "BUY"
                confidence = min(regime_confidence * abs(weighted_prediction), 1.0)
            else:
                action = "HOLD"
                confidence = 0.5
        elif regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE]:
            if weighted_prediction < -0.1:
                action = "SELL"
                confidence = min(regime_confidence * abs(weighted_prediction), 1.0)
            else:
                action = "HOLD"
                confidence = 0.5
        else:  # Sideways
            if abs(weighted_prediction) > 0.2:
                action = "BUY" if weighted_prediction > 0 else "SELL"
                confidence = min(regime_confidence * abs(weighted_prediction), 0.8)
            else:
                action = "HOLD"
                confidence = 0.5
        
        # Calculate position size based on confidence and regime
        base_size = 0.1  # 10% base position
        position_size = base_size * confidence
        
        if regime in [MarketRegime.BULL_VOLATILE, MarketRegime.BEAR_VOLATILE]:
            position_size *= 0.5  # Reduce size in volatile regimes
        
        # Calculate stop loss and take profit
        current_price = features.get('close', 0)
        atr = features.get('atr', 0)
        
        stop_loss = None
        take_profit = None
        
        if action in ["BUY", "SELL"] and current_price > 0 and atr > 0:
            if action == "BUY":
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
        
        # Create reasoning
        reasoning = f"Regime: {regime.value} (confidence: {regime_confidence:.2f}), "
        reasoning += f"Prediction: {weighted_prediction:.3f}, "
        reasoning += f"Action: {action} (confidence: {confidence:.2f})"
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            market_regime=regime,
            alpha_predictions=alpha_predictions
        )
