# hedge_fund_ml_backtest.py
"""
Hedge Fund Grade ML Trading Backtest System
Features:
- Walk-forward optimization
- Multiple ML models with GPU acceleration
- Realistic execution simulation
- Risk management and position sizing
- No data leakage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from models.ensemble_gpu_hedge_fund import HedgeFundGPUEnsemble
from models.enhanced_features import EnhancedFeatureEngineer
from execution_simulator import ExecutionSimulator, ExecutionConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HedgeFundBacktestConfig:
    """Configuration for hedge fund backtest"""

    # Capital and position sizing
    initial_capital: float = 1_000_000
    position_size_method: str = "risk_parity"  # "equal", "risk_parity", "kelly"
    base_position_size: float = 0.02  # 2% per position
    max_position_size: float = 0.05  # 5% max
    max_positions: int = 20
    max_sector_exposure: float = 0.30  # 30% in one sector

    # Risk management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0
    max_portfolio_heat: float = 0.06  # 6% total portfolio risk
    correlation_threshold: float = 0.70  # Avoid correlated positions

    # Walk-forward optimization
    train_months: int = 12  # 12 months training
    validation_months: int = 3  # 3 months validation
    test_months: int = 1  # 1 month test
    buffer_days: int = 5  # Days between train/test to avoid leakage
    retrain_frequency_days: int = 21  # Retrain every month

    # ML model thresholds
    min_prediction_confidence: float = 0.65
    ensemble_agreement_threshold: float = 0.60
    feature_importance_threshold: float = 0.02

    # Execution
    execution_delay_minutes: int = 5
    use_vwap: bool = True
    max_spread_bps: float = 20

    # Filters
    min_sharpe_for_trading: float = 1.0
    min_training_samples: int = 100
    min_validation_score: float = 0.55

    # Liquidity
    min_liquidity_usd: float = 1_000_000  # $1M daily volume


class DataManager:
    """Enhanced data manager with proper historical data handling"""

    def __init__(self):
        self.data_cache = {}
        self.data_lookback_days = 500  # Increased for 200-day features + buffer  # IMPORTANT: Extra days for feature calculation

    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data with extra historical buffer for feature engineering"""

        # Convert dates
        end = pd.to_datetime(end_date)
        start = pd.to_datetime(start_date)

        # Fetch extra historical data for feature calculation
        extended_start = start - timedelta(days=self.data_lookback_days)

        cache_key = f"{symbol}_{extended_start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"

        if cache_key in self.data_cache:
            logger.debug(f"Using cached data for {symbol}")
            df = self.data_cache[cache_key].copy()
        else:
            try:
                logger.debug(f"Fetching {symbol} from {extended_start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

                df = yf.download(
                    symbol,
                    start=extended_start.strftime('%Y-%m-%d'),
                    end=end.strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    progress=False
                )

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()

                # Fix column names from yfinance MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    # Extract first level from MultiIndex
                    new_columns = []
                    for col in df.columns:
                        if isinstance(col, tuple):
                            new_columns.append(col[0].lower())
                        else:
                            new_columns.append(str(col).lower())
                    df.columns = new_columns
                else:
                    # Regular columns
                    df.columns = [str(col).lower() for col in df.columns]

                # Verify we have required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    logger.error(f"Missing required columns for {symbol}: {missing}")
                    logger.error(f"Available columns: {df.columns.tolist()}")
                    return pd.DataFrame()

                # Add symbol column
                df['symbol'] = symbol

                # Cache the data
                self.data_cache[cache_key] = df.copy()

                logger.debug(f"Fetched {len(df)} rows for {symbol} (including {self.data_lookback_days} day buffer)")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return pd.DataFrame()

        return df

    def fetch_all_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols with progress tracking"""

        all_data = {}
        failed_symbols = []

        logger.info(f"Fetching data for {len(symbols)} symbols...")

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_data, symbol, start_date, end_date): symbol 
                for symbol in symbols
            }

            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty and len(df) >= 250:  # Increased for 200-day features  # Need minimum data for features
                        all_data[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                except Exception as e:
                    failed_symbols.append(symbol)
                    logger.error(f"Failed to fetch {symbol}: {e}")

                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(symbols)} symbols fetched")

        logger.info(f"Successfully fetched data for {len(all_data)} symbols")
        if failed_symbols:
            logger.warning(f"Failed symbols ({len(failed_symbols)}): {failed_symbols[:10]}...")

        return all_data
class TrainedModel:
    """Container for trained model and metadata"""

    def __init__(self, ensemble: HedgeFundGPUEnsemble, train_date: pd.Timestamp,
                 validation_score: float, feature_importance: Dict[str, float]):
        self.ensemble = ensemble
        self.train_date = train_date
        self.validation_score = validation_score
        self.feature_importance = feature_importance
        self.predictions_cache = {}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Get predictions with caching"""
        cache_key = hash(features.index.tolist()[0]) if len(features) > 0 else 0

        if cache_key not in self.predictions_cache:
            self.predictions_cache[cache_key] = self.ensemble.predict_proba(features)

        return self.predictions_cache[cache_key]


class ModelManager:
    """Manages model training and selection"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.models = {}
        self.model_performance = defaultdict(list)

    def train_models(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame],
                     train_end: pd.Timestamp) -> TrainedModel:
        """Train ensemble models with proper date filtering"""

        # Filter data to actual training period AFTER features are calculated
        filtered_train_data = {}
        for symbol, df in train_data.items():
            if len(df) > 0:
                # Keep only data up to train_end
                mask = df.index <= train_end
                filtered_df = df[mask]

                # Ensure we still have enough data after filtering
                if len(filtered_df) >= 50:  # Minimum for training
                    filtered_train_data[symbol] = filtered_df
                else:
                    logger.debug(f"Skipping {symbol}: insufficient data after date filter")

        if not filtered_train_data:
            logger.error("No valid training data after filtering")
            raise ValueError("No valid training data after date filtering")

        logger.info(f"Training on {len(filtered_train_data)} symbols (from {len(train_data)} provided)")

        # Create ensemble
        gpu_config = self.config.__dict__.copy()
        ensemble = HedgeFundGPUEnsemble()

        # Prepare training data - this is where features are created
        X_train, y_train, train_info = ensemble.prepare_training_data(filtered_train_data)

        # Prepare validation data
        X_val, y_val, _ = ensemble.prepare_training_data(val_data)

        # Train ensemble
        sample_weights = train_info.get('sample_weights')
        ensemble.train_combined(X_train, y_train, X_val, y_val, sample_weights)

        # Validate performance
        val_score = ensemble.validate(X_val, y_val)

        # Get feature importance
        feature_importance = ensemble.get_feature_importance()

        # Create trained model
        model = TrainedModel(ensemble, train_end, val_score, feature_importance)

        # Store model
        self.models[train_end] = model

        logger.info(f"Model trained for {train_end.date()}: Val Score = {val_score:.4f}")

        return model

    def get_best_model(self, date: pd.Timestamp) -> Optional[TrainedModel]:
        """Get the most recent valid model for a given date"""

        valid_models = [
            (train_date, model) for train_date, model in self.models.items()
            if train_date <= date and model.validation_score >= self.config.min_validation_score
        ]

        if not valid_models:
            return None

        # Return most recent valid model
        valid_models.sort(key=lambda x: x[0], reverse=True)
        return valid_models[0][1]


class RiskManager:
    """Manages portfolio risk and position sizing"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.correlation_matrix = None

    def calculate_position_size(self, portfolio_value: float, volatility: float,
                                confidence: float, existing_positions: int) -> float:
        """Calculate position size based on risk parity and constraints"""

        # Base position size
        if self.config.position_size_method == "equal":
            base_size = portfolio_value * self.config.base_position_size
        elif self.config.position_size_method == "risk_parity":
            # Adjust size based on volatility
            target_risk = portfolio_value * self.config.base_position_size * 0.02  # 2% volatility target
            base_size = target_risk / (volatility + 1e-6)
        else:
            base_size = portfolio_value * self.config.base_position_size

        # Adjust for confidence
        size_multiplier = 0.5 + (confidence - self.config.min_prediction_confidence) * 2
        size_multiplier = np.clip(size_multiplier, 0.5, 1.5)

        position_size = base_size * size_multiplier

        # Apply constraints
        max_size = portfolio_value * self.config.max_position_size
        position_size = min(position_size, max_size)

        # Reduce size if too many positions
        if existing_positions >= self.config.max_positions * 0.8:
            position_size *= 0.5

        return position_size

    def check_correlation(self, symbol: str, existing_positions: List[str],
                          returns_data: pd.DataFrame) -> bool:
        """Check if new position is too correlated with existing positions"""

        if not existing_positions or symbol not in returns_data.columns:
            return True

        for existing_symbol in existing_positions:
            if existing_symbol in returns_data.columns:
                correlation = returns_data[symbol].corr(returns_data[existing_symbol])
                if abs(correlation) > self.config.correlation_threshold:
                    logger.debug(f"High correlation between {symbol} and {existing_symbol}: {correlation:.2f}")
                    return False

        return True

    def calculate_stop_loss(self, entry_price: float, atr: float) -> float:
        """Calculate stop loss price"""
        return entry_price - (atr * self.config.stop_loss_atr_multiplier)

    def calculate_take_profit(self, entry_price: float, atr: float) -> float:
        """Calculate take profit price"""
        return entry_price + (atr * self.config.take_profit_atr_multiplier)

    def check_portfolio_heat(self, positions: Dict, portfolio_value: float) -> bool:
        """Check if total portfolio risk is within limits"""

        total_risk = 0
        for position in positions.values():
            position_risk = position['quantity'] * (position['entry_price'] - position['stop_loss'])
            total_risk += position_risk

        portfolio_heat = total_risk / portfolio_value
        return portfolio_heat <= self.config.max_portfolio_heat


class HedgeFundBacktester:
    """Main backtesting engine"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.data_manager = DataManager()
        self.model_manager = ModelManager(config)
        self.risk_manager = RiskManager(config)
        self.execution_sim = ExecutionSimulator()

        # Portfolio state
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run the complete backtest"""

        logger.info(f"Starting hedge fund backtest for {len(symbols)} symbols")
        logger.info(f"Period: {start_date} to {end_date}")

        # Ensure sufficient historical data for all features
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'data_lookback_days'):
                # Increase lookback for advanced features
                self.data_manager.data_lookback_days = max(500, self.data_manager.data_lookback_days)
                logger.info(f"Set data lookback to {self.data_manager.data_lookback_days} days")

        try:
            # Fetch all data with historical buffer
            all_data = self.data_manager.fetch_all_data(symbols, start_date, end_date)

            if not all_data:
                return {'error': 'No data fetched'}

            # Prepare data for backtesting
            prepared_data = self._prepare_backtest_data(all_data, start_date, end_date)

            if not prepared_data:
                return {'error': 'No valid data after preparation'}

            logger.info(
                f"Data prepared for {len(prepared_data)} symbols ({len(prepared_data) / len(symbols) * 100:.1f}% of watchlist)")

            # Get unique dates for iteration
            all_dates = set()
            for symbol_data in prepared_data.values():
                all_dates.update(symbol_data.index)

            trading_dates = sorted(list(all_dates))

            # Filter to backtest period
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)
            trading_dates = [d for d in trading_dates if start_ts <= d <= end_ts]

            logger.info(f"Backtesting over {len(trading_dates)} trading days")

            # Initialize portfolio tracking
            self.portfolio_values = []
            self.daily_returns = []

            # Progress tracking
            from tqdm import tqdm

            # Main backtest loop
            current_model = None
            last_train_date = None

            for i, date in enumerate(tqdm(trading_dates, desc="Backtesting")):
                # Check if we need to retrain
                if (current_model is None or
                        last_train_date is None or
                        (date - last_train_date).days >= self.config.retrain_frequency_days):
                    # Train new model
                    current_model = self._train_model_for_date(all_data, date)
                    last_train_date = date

                if current_model is None:
                    continue

                # Update positions (check stops, take profits)
                self._update_positions(date, prepared_data)

                # Generate signals
                signals = self._generate_signals(date, prepared_data, current_model)

                # Execute trades
                self._execute_trades(date, signals, prepared_data)

                # Track portfolio value
                portfolio_value = self._calculate_portfolio_value(date, prepared_data)
                self.portfolio_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'cash': self.cash,
                    'n_positions': len(self.positions)
                })

                # Calculate daily return
                if i > 0:
                    prev_value = self.portfolio_values[i - 1]['value']
                    daily_return = (portfolio_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)

            # Calculate final metrics
            results = self._calculate_backtest_metrics()

            # Add additional information
            results['symbol_utilization'] = {
                'total_watchlist': len(symbols),
                'symbols_traded': len(set(trade['symbol'] for trade in self.trades)),
                'utilization_rate': len(set(trade['symbol'] for trade in self.trades)) / len(symbols) * 100
            }

            return results

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return {'error': str(e), 'details': repr(e)}

    def _prepare_backtest_data(self, all_data: Dict[str, pd.DataFrame],
                               start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting with proper date filtering"""

        prepared_data = {}
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        for symbol, df in all_data.items():
            # Keep all data for now - will filter after feature creation
            if len(df) >= 200:  # Minimum for feature calculation
                prepared_data[symbol] = df

        return prepared_data

    def _train_model_for_date(self, all_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> Optional[TrainedModel]:
        """Train model for specific date with walk-forward optimization"""

        logger.info(f"Training model for {current_date.date()}")

        # Calculate training periods
        train_end = current_date - timedelta(days=self.config.buffer_days)
        train_start = train_end - timedelta(days=self.config.train_months * 30)

        val_end = train_end
        val_start = val_end - timedelta(days=self.config.validation_months * 30)

        # Prepare training data
        train_data = {}
        val_data = {}

        for symbol, df in all_data.items():
            # Training data
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            train_df = df[train_mask]

            if len(train_df) >= self.config.min_training_samples:
                train_data[symbol] = df  # Pass full data for feature calculation

            # Validation data
            val_mask = (df.index >= val_start) & (df.index <= val_end)
            val_df = df[val_mask]

            if len(val_df) >= 20:
                val_data[symbol] = df  # Pass full data for feature calculation

        if not train_data or not val_data:
            logger.warning(f"Insufficient data for training at {current_date.date()}")
            return None

        logger.info(
            f"Training on {len(train_data)} symbols ({len(train_data) / len(all_data) * 100:.1f}% of available)")

        try:
            # Train models
            model = self.model_manager.train_models(train_data, val_data, train_end)
            return model
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None

    def _generate_signals(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame],
                          model: TrainedModel) -> List[Dict]:
        """Generate trading signals for given date"""

        signals = []

        # Get current positions
        current_symbols = list(self.positions.keys())

        # Feature engineer
        feature_engineer = EnhancedFeatureEngineer()

        for symbol, df in data.items():
            # Skip if already have position
            if symbol in current_symbols:
                continue

            # Check if we have data for this date
            if date not in df.index:
                continue

            # Get recent data for features
            recent_data = df[df.index <= date].tail(250)  # Enough for all features

            if len(recent_data) < 200:
                continue

            try:
                # Create features
                features = feature_engineer.create_all_features(recent_data, symbol)

                if features.empty:
                    continue

                # Get latest features
                if date in features.index:
                    latest_features = features.loc[[date]]
                else:
                    # Use most recent available
                    latest_features = features.iloc[[-1]]

                # Get prediction
                prediction = model.predict(latest_features)

                if len(prediction) > 0:
                    confidence = prediction[0]

                    # Generate signal if confident
                    if confidence >= self.config.min_prediction_confidence:
                        # Calculate volatility for position sizing
                        returns = recent_data['close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)

                        # Check correlation with existing positions
                        if self.risk_manager.check_correlation(symbol, current_symbols,
                                                               pd.DataFrame({s: data[s]['close'].pct_change()
                                                                             for s in current_symbols
                                                                             if s in data})):
                            signals.append({
                                'symbol': symbol,
                                'date': date,
                                'confidence': confidence,
                                'volatility': volatility,
                                'atr': self._calculate_atr(recent_data),
                                'price': recent_data['close'].iloc[-1],
                                'volume': recent_data['volume'].iloc[-1]
                            })

            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")
                continue

        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)

        # Limit signals based on position constraints
        max_new_positions = self.config.max_positions - len(self.positions)
        signals = signals[:max_new_positions]

        return signals

    def _execute_trades(self, date: pd.Timestamp, signals: List[Dict],
                        data: Dict[str, pd.DataFrame]) -> None:
        """Execute trades based on signals"""

        for signal in signals:
            symbol = signal['symbol']

            # Calculate position size
            portfolio_value = self._calculate_portfolio_value(date, data)
            position_size = self.risk_manager.calculate_position_size(
                portfolio_value, signal['volatility'],
                signal['confidence'], len(self.positions)
            )

            # Skip if position size too small
            if position_size < 1000:
                continue

            # Check portfolio heat
            if not self.risk_manager.check_portfolio_heat(self.positions, portfolio_value):
                logger.debug(f"Portfolio heat limit reached, skipping {symbol}")
                continue

            # Calculate shares
            price = signal['price']
            shares = int(position_size / price)

            if shares == 0:
                continue

            # Simulate execution
            exec_price, slippage, commission = self.execution_sim.simulate_entry(
                symbol, price, shares, signal['volume']
            )

            # Calculate cost
            total_cost = (exec_price * shares) + commission

            # Check if we have enough cash
            if total_cost > self.cash:
                continue

            # Calculate stops
            atr = signal['atr']
            stop_loss = self.risk_manager.calculate_stop_loss(exec_price, atr)
            take_profit = self.risk_manager.calculate_take_profit(exec_price, atr)

            # Execute trade
            self.cash -= total_cost

            position = {
                'symbol': symbol,
                'entry_date': date,
                'entry_price': exec_price,
                'quantity': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'commission_paid': commission,
                'slippage_paid': slippage
            }

            self.positions[symbol] = position

            # Record trade
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': shares,
                'price': exec_price,
                'commission': commission,
                'slippage': slippage,
                'confidence': signal['confidence']
            })

            logger.debug(f"Bought {shares} shares of {symbol} at ${exec_price:.2f}")

    def _update_positions(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> None:
        """Update existing positions - check stops and targets"""

        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in data or date not in data[symbol].index:
                continue

            current_price = data[symbol].loc[date, 'close']

            # Check stop loss
            if current_price <= position['stop_loss']:
                positions_to_close.append((symbol, 'STOP_LOSS', current_price))

            # Check take profit
            elif current_price >= position['take_profit']:
                positions_to_close.append((symbol, 'TAKE_PROFIT', current_price))

            # Check time-based exit (optional)
            elif (date - position['entry_date']).days >= 20:  # 20 day max hold
                positions_to_close.append((symbol, 'TIME_EXIT', current_price))

        # Close positions
        for symbol, reason, price in positions_to_close:
            self._close_position(symbol, date, price, reason, data)

    def _close_position(self, symbol: str, date: pd.Timestamp, price: float,
                        reason: str, data: Dict[str, pd.DataFrame]) -> None:
        """Close a position"""

        position = self.positions[symbol]
        quantity = position['quantity']

        # Get volume for execution simulation
        volume = data[symbol].loc[date, 'volume'] if symbol in data and date in data[symbol].index else 1000000

        # Simulate execution
        exec_price, slippage, commission = self.execution_sim.simulate_exit(
            symbol, price, quantity, volume
        )

        # Calculate proceeds
        proceeds = (exec_price * quantity) - commission
        self.cash += proceeds

        # Calculate P&L
        entry_cost = position['entry_price'] * quantity + position['commission_paid']
        exit_proceeds = proceeds
        pnl = exit_proceeds - entry_cost
        pnl_pct = pnl / entry_cost

        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': exec_price,
            'commission': commission,
            'slippage': slippage,
            'reason': reason,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

        # Remove position
        del self.positions[symbol]

        logger.debug(f"Closed {symbol} position: {reason}, P&L: ${pnl:.2f} ({pnl_pct * 100:.1f}%)")

    def _calculate_portfolio_value(self, date: pd.Timestamp,
                                   data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value"""

        positions_value = 0

        for symbol, position in self.positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'close']
                positions_value += current_price * position['quantity']

        return self.cash + positions_value

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # True Range
        tr1 = high - low
        tr2 = abs(high - np.roll(close, 1))
        tr3 = abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # ATR
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]

        return atr

    def _calculate_backtest_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""

        if not self.portfolio_values:
            return {'error': 'No portfolio values recorded'}

        # Convert to DataFrame for easier calculation
        pv_df = pd.DataFrame(self.portfolio_values)

        # Basic metrics
        initial_value = self.config.initial_capital
        final_value = pv_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # Calculate daily returns
        if len(self.daily_returns) > 0:
            daily_returns = np.array(self.daily_returns)

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-6)

            # Max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)

            # Win rate
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]
            win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if self.trades else 0

            # Average win/loss
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

            # Profit factor
            total_wins = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
            total_losses = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Trading frequency
        n_days = len(pv_df)
        n_trades = len([t for t in self.trades if t['action'] == 'BUY'])
        trades_per_day = n_trades / n_days if n_days > 0 else 0

        # Annual metrics
        n_years = n_days / 252
        annual_return = (final_value / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0

        metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': n_trades,
            'trades_per_day': trades_per_day,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_values': pv_df.to_dict('records'),
            'trades': self.trades
        }

        # Symbol-level statistics
        symbol_stats = defaultdict(lambda: {'trades': 0, 'pnl': 0})
        for trade in self.trades:
            if trade['action'] == 'SELL':
                symbol_stats[trade['symbol']]['trades'] += 1
                symbol_stats[trade['symbol']]['pnl'] += trade.get('pnl', 0)

        metrics['symbol_stats'] = dict(symbol_stats)

        return metrics


# Utility functions
def plot_backtest_results(results: Dict) -> None:
    """Plot backtest results"""

    if 'error' in results:
        print(f"Cannot plot results due to error: {results['error']}")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Portfolio value over time
    pv_df = pd.DataFrame(results['portfolio_values'])
    pv_df['date'] = pd.to_datetime(pv_df['date'])
    pv_df.set_index('date', inplace=True)

    axes[0, 0].plot(pv_df.index, pv_df['value'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')

    # Drawdown
    returns = pv_df['value'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max

    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')

    # Number of positions over time
    axes[1, 0].plot(pv_df.index, pv_df['n_positions'])
    axes[1, 0].set_title('Number of Positions')
    axes[1, 0].set_ylabel('Active Positions')

    # P&L distribution
    trades_df = pd.DataFrame([t for t in results['trades'] if 'pnl' in t])
    if not trades_df.empty:
        axes[1, 1].hist(trades_df['pnl'], bins=30, alpha=0.7)
        axes[1, 1].axvline(0, color='black', linestyle='--')
        axes[1, 1].set_title('P&L Distribution')
        axes[1, 1].set_xlabel('P&L ($)')
        axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def save_backtest_results(results: Dict, filename: str) -> None:
    """Save backtest results to file"""

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj

    # Recursively convert all values
    clean_results = json.loads(json.dumps(results, default=convert_types))

    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=2)

    logger.info(f"Results saved to {filename}")


if __name__ == "__main__":
    # Example usage
    from config.watchlist import WATCHLIST

    # Create configuration
    config = HedgeFundBacktestConfig(
        initial_capital=100000,
        max_positions=10,
        train_months=6,
        validation_months=2,
        test_months=1
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Run backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    results = backtester.run_backtest(
        symbols=WATCHLIST[:20],  # Test with 20 symbols
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    # Display results
    if 'error' not in results:
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {results['win_rate'] * 100:.1f}%")

        # Plot results
        plot_backtest_results(results)

        # Save results
        save_backtest_results(results, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        print(f"Backtest failed: {results['error']}")