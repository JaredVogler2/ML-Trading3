# hedge_fund_ml_backtest.py
"""
Institutional-Grade ML Backtesting System - FIXED VERSION
- Proper imports and integrations
- Better threshold management
- Correct yfinance handling
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
import warnings
import json
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import torch
from scipy import stats

# Import your existing modules - FIXED IMPORTS
from models.ensemble_gpu_hedge_fund import HedgeFundGPUEnsemble, GPUConfig
from models.enhanced_features import EnhancedFeatureEngineer
from config.watchlist import WATCHLIST, SECTOR_MAPPING
from execution_simulator import ExecutionSimulator, ExecutionConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HedgeFundBacktestConfig:
    """Professional backtesting configuration - ADJUSTED FOR BETTER SIGNALS"""
    # Capital and position sizing
    initial_capital: float = 100000
    position_size_method: str = "risk_parity"
    base_position_size: float = 0.02
    max_position_size: float = 0.05
    max_positions: int = 20
    max_sector_exposure: float = 0.30

    # Risk parameters
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0  # Reduced from 4.0 for more achievable targets
    max_portfolio_heat: float = 0.06
    correlation_threshold: float = 0.70

    # Walk-forward parameters
    train_months: int = 12
    validation_months: int = 3
    test_months: int = 1
    buffer_days: int = 5
    retrain_frequency_days: int = 21

    # ML parameters - ADJUSTED FOR MORE SIGNALS
    min_prediction_confidence: float = 0.55  # Reduced from 0.65
    ensemble_agreement_threshold: float = 0.50  # Reduced from 0.60
    feature_importance_threshold: float = 0.01  # Reduced from 0.05

    # Target parameters - MORE REALISTIC
    target_return_threshold: float = 0.01  # 1% instead of 2%
    target_lookforward_days: int = 5

    # Execution parameters
    execution_delay_minutes: int = 5
    use_vwap: bool = True
    max_spread_bps: float = 20

    # Performance filters
    min_sharpe_for_trading: float = 0.5  # Reduced from 1.0
    min_training_samples: int = 100
    min_validation_score: float = 0.52  # Reduced from 0.55

    # Data quality
    min_liquidity_usd: float = 1_000_000
    max_missing_data_pct: float = 0.05


class DataPipeline:
    """Efficient data pipeline with caching and quality checks"""

    def __init__(self, cache_dir: str = "cache/hedge_fund"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.feature_engineer = EnhancedFeatureEngineer(use_gpu=torch.cuda.is_available())

    def fetch_and_prepare_data(self, symbols: List[str],
                               start_date: str, end_date: str,
                               force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch data with quality checks and feature engineering"""

        cache_file = os.path.join(self.cache_dir,
                                  f"prepared_data_{len(symbols)}_{start_date}_{end_date}.pkl")

        if os.path.exists(cache_file) and not force_refresh:
            logger.info("Loading prepared data from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Fetching data for {len(symbols)} symbols...")

        # Fetch raw data in parallel
        all_data = {}
        failed_symbols = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_single_symbol, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_symbol),
                               total=len(symbols), desc="Fetching data"):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        # Add features using enhanced feature engineer
                        data_with_features = self.feature_engineer.create_all_features(data, symbol)

                        # Merge original OHLCV data with features
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in data.columns and col not in data_with_features.columns:
                                data_with_features[col] = data[col]

                        # Add symbol column
                        data_with_features['symbol'] = symbol

                        # Quality check
                        if self._pass_quality_check(data_with_features):
                            all_data[symbol] = data_with_features
                        else:
                            failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    failed_symbols.append(symbol)

        logger.info(f"Successfully prepared {len(all_data)} symbols, {len(failed_symbols)} failed")

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(all_data, f)

        return all_data

    def _fetch_single_symbol(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch single symbol with error handling - FIXED FOR MULTIINDEX"""
        try:
            # Download with auto_adjust to avoid MultiIndex
            df = yf.download(symbol, start=start_date, end=end_date,
                             progress=False, auto_adjust=True, actions=False)

            if df.empty or len(df) < 100:
                return None

            # Handle MultiIndex columns if they still exist
            if isinstance(df.columns, pd.MultiIndex):
                # If MultiIndex, get the first level (price data)
                df.columns = df.columns.get_level_values(0)

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"{symbol}: Missing required columns")
                return None

            # Calculate additional base fields
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['dollar_volume'] = df['close'] * df['volume']

            # Basic technical indicators for feature engineering
            df['atr'] = self._calculate_atr(df)
            df['volatility_20d'] = df['returns'].rolling(20).std() * np.sqrt(252)

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))

        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _pass_quality_check(self, df: pd.DataFrame) -> bool:
        """Quality checks for data"""
        # Check minimum length
        if len(df) < 252:  # 1 year minimum
            return False

        # Check liquidity
        if 'dollar_volume' in df.columns:
            avg_dollar_volume = df['dollar_volume'].tail(20).mean()
            if avg_dollar_volume < HedgeFundBacktestConfig().min_liquidity_usd:
                return False

        # Check missing data
        critical_cols = ['close', 'volume', 'returns']
        for col in critical_cols:
            if col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > HedgeFundBacktestConfig().max_missing_data_pct:
                    return False

        return True


class MLModelManager:
    """Manages ML model training, validation, and prediction"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.models = {}
        self.model_performance = defaultdict(list)
        self.feature_importance = {}

    def train_models(self, train_data: Dict[str, pd.DataFrame],
                     val_data: Dict[str, pd.DataFrame],
                     train_end_date: pd.Timestamp) -> HedgeFundGPUEnsemble:
        """Train ensemble model with proper validation"""

        logger.info(f"Training models with {len(train_data)} symbols...")

        # Initialize ensemble with GPU config
        gpu_config = GPUConfig(
            batch_size=512,
            n_epochs=50,  # Reduced for faster iteration
            early_stopping_patience=10,
            learning_rate=0.001
        )
        ensemble = HedgeFundGPUEnsemble(config=gpu_config)

        # Prepare combined training data
        train_combined = {}
        for symbol in train_data:
            if symbol in val_data:
                train_combined[symbol] = train_data[symbol]

        if not train_combined:
            logger.warning("No valid training data")
            return ensemble

        # Use ensemble's prepare_training_data method
        try:
            X_train, y_train, metadata = ensemble.prepare_training_data(train_combined)

            # Prepare validation data similarly
            X_val, y_val, _ = ensemble.prepare_training_data(val_data)

            # Train the ensemble
            ensemble.train_combined(X_train, y_train, X_val, y_val,
                                    sample_weights=metadata.get('sample_weights'))

            # Validate performance
            val_score = ensemble.validate(X_val, y_val)
            logger.info(f"Validation score: {val_score:.3f}")

            # Store performance
            self.model_performance[train_end_date] = {
                'val_score': val_score,
                'n_train_samples': len(X_train),
                'n_val_samples': len(X_val)
            }

            # Get feature importance
            self.feature_importance[train_end_date] = ensemble.get_feature_importance()

        except Exception as e:
            logger.error(f"Error training models: {e}")

        return ensemble

    def predict(self, model: HedgeFundGPUEnsemble,
                current_data: Dict[str, pd.DataFrame],
                date: pd.Timestamp) -> Dict[str, Dict]:
        """Generate predictions for all symbols"""

        predictions = {}

        for symbol, df in current_data.items():
            # Get data up to current date
            hist_data = df[df.index <= date].copy()

            if len(hist_data) < 50:
                continue

            try:
                # Use the feature engineer to create features
                feature_engineer = EnhancedFeatureEngineer(use_gpu=False)
                features = feature_engineer.create_all_features(hist_data, symbol)

                if features.empty:
                    continue

                # Get latest features
                latest_features = features.iloc[-1:].copy()

                # Get prediction from ensemble
                pred = model.predict_proba(latest_features)

                # Get individual model predictions for confidence
                individual_preds = model.get_individual_predictions(latest_features)

                # Calculate agreement
                if len(individual_preds) > 0:
                    agreement = np.mean([p[0] > 0.5 for p in individual_preds if len(p) > 0])
                else:
                    agreement = 0.5

                predictions[symbol] = {
                    'probability': float(pred[0]) if len(pred) > 0 else 0.5,
                    'prediction': int(pred[0] > 0.5) if len(pred) > 0 else 0,
                    'confidence': float(pred[0]) if pred[0] > 0.5 else 1 - float(pred[0]),
                    'model_agreement': agreement,
                    'latest_price': hist_data['close'].iloc[-1],
                    'atr': hist_data['atr'].iloc[-1] if 'atr' in hist_data else 0,
                    'volume': hist_data['volume'].iloc[-1]
                }

            except Exception as e:
                logger.debug(f"Prediction error for {symbol}: {e}")
                continue

        return predictions


class HedgeFundBacktester:
    """Main backtesting engine with ML integration - FIXED VERSION"""

    def __init__(self, config: HedgeFundBacktestConfig = None):
        self.config = config or HedgeFundBacktestConfig()
        self.data_pipeline = DataPipeline()
        self.model_manager = MLModelManager(config)
        self.risk_manager = RiskManager(config)
        self.execution_sim = ExecutionSimulator(ExecutionConfig())

        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        self.model_predictions = defaultdict(list)

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run complete ML-integrated backtest"""

        logger.info(f"Starting hedge fund backtest for {len(symbols)} symbols")
        logger.info(f"Period: {start_date} to {end_date}")

        # Calculate required data range with buffer
        backtest_start = pd.to_datetime(start_date)
        data_start = backtest_start - timedelta(days=365 * 2)  # 2 years of history

        # Fetch and prepare all data
        all_data = self.data_pipeline.fetch_and_prepare_data(
            symbols, data_start.strftime('%Y-%m-%d'), end_date
        )

        if len(all_data) < 10:
            logger.error(f"Insufficient data: only {len(all_data)} symbols available")
            return {'error': 'Insufficient data for backtesting'}

        logger.info(f"Data prepared for {len(all_data)} symbols")

        # Initialize portfolio
        cash = self.config.initial_capital
        portfolio_value = self.config.initial_capital

        # Walk-forward optimization loop
        current_date = backtest_start
        end_date_ts = pd.to_datetime(end_date)

        current_model = None
        last_train_date = None

        # Generate trading dates
        all_trading_dates = pd.bdate_range(start=backtest_start, end=end_date_ts)

        # Track predictions for analysis
        all_predictions = []

        for date in tqdm(all_trading_dates, desc="Backtesting"):
            # Check if we need to retrain
            if (current_model is None or
                    (last_train_date and (date - last_train_date).days >= self.config.retrain_frequency_days)):
                logger.info(f"Training model for {date.date()}")
                current_model = self._train_model_for_date(all_data, date)
                last_train_date = date

                # Skip trading for buffer period
                continue

            # Skip if within buffer period
            if last_train_date and (date - last_train_date).days < self.config.buffer_days:
                continue

            # Update portfolio with current prices
            self._update_portfolio_prices(date, all_data)

            # Check exits
            exits = self._check_exit_conditions(date, all_data)
            for exit in exits:
                cash += self._execute_exit(exit, date, all_data)

            # Generate new signals
            if len(self.positions) < self.config.max_positions:
                predictions = self.model_manager.predict(current_model, all_data, date)

                # Log prediction statistics
                if predictions:
                    pred_probs = [p['probability'] for p in predictions.values()]
                    high_conf = sum(1 for p in predictions.values()
                                    if p['confidence'] > self.config.min_prediction_confidence)
                    logger.debug(f"Date {date.date()}: {len(predictions)} predictions, "
                                 f"{high_conf} high confidence, "
                                 f"avg prob: {np.mean(pred_probs):.3f}")
                    all_predictions.extend(predictions.values())

                signals = self._filter_signals(predictions, date, all_data)

                # Log signal statistics
                if signals:
                    logger.info(f"Date {date.date()}: {len(signals)} valid signals generated")

                # Execute new positions
                for signal in signals[:self.config.max_positions - len(self.positions)]:
                    cost = self._execute_entry(signal, date, all_data, cash, portfolio_value)
                    if cost > 0:
                        cash -= cost

            # Calculate portfolio value
            positions_value = sum(pos['current_value'] for pos in self.positions.values())
            portfolio_value = cash + positions_value

            # Record metrics
            self.equity_curve.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'n_positions': len(self.positions)
            })

        # Close all remaining positions
        for symbol in list(self.positions.keys()):
            exit_info = {
                'symbol': symbol,
                'reason': 'end_of_backtest'
            }
            cash += self._execute_exit(exit_info, all_trading_dates[-1], all_data)

        # Log prediction analysis
        if all_predictions:
            avg_prob = np.mean([p['probability'] for p in all_predictions])
            high_conf_pct = sum(1 for p in all_predictions
                                if p['confidence'] > self.config.min_prediction_confidence) / len(all_predictions)
            logger.info(f"Prediction Statistics:")
            logger.info(f"  Total predictions: {len(all_predictions)}")
            logger.info(f"  Average probability: {avg_prob:.3f}")
            logger.info(f"  High confidence %: {high_conf_pct * 100:.1f}%")

        # Calculate final metrics
        results = self._calculate_performance_metrics()

        # Generate reports
        self._generate_reports(results)

        return results

    def _train_model_for_date(self, all_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> HedgeFundGPUEnsemble:
        """Train model with proper data splitting"""

        # Calculate date ranges
        train_end = current_date - timedelta(days=self.config.buffer_days)
        train_start = train_end - timedelta(days=self.config.train_months * 30)
        val_end = train_end
        val_start = val_end - timedelta(days=self.config.validation_months * 30)

        # Split data
        train_data = {}
        val_data = {}

        for symbol, df in all_data.items():
            # Training data
            train_mask = (df.index >= train_start) & (df.index < val_start)
            train_df = df[train_mask].copy()

            # Validation data
            val_mask = (df.index >= val_start) & (df.index < val_end)
            val_df = df[val_mask].copy()

            if len(train_df) >= self.config.min_training_samples and len(val_df) >= 20:
                train_data[symbol] = train_df
                val_data[symbol] = val_df

        logger.info(f"Training on {len(train_data)} symbols")

        # Train ensemble model
        model = self.model_manager.train_models(train_data, val_data, train_end)

        return model

    def _filter_signals(self, predictions: Dict, date: pd.Timestamp,
                        all_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Filter and rank trading signals - IMPROVED LOGIC"""

        valid_signals = []

        for symbol, pred in predictions.items():
            # More lenient filtering
            if pred['confidence'] < self.config.min_prediction_confidence * 0.9:  # 10% buffer
                continue

            if pred['model_agreement'] < self.config.ensemble_agreement_threshold * 0.9:  # 10% buffer
                continue

            # Check if we have price data
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            recent_data = df[df.index <= date].tail(20)

            if recent_data.empty:
                continue

            # Check liquidity
            avg_dollar_volume = recent_data['dollar_volume'].mean() if 'dollar_volume' in recent_data else 0
            if avg_dollar_volume < self.config.min_liquidity_usd * 0.8:  # 20% buffer
                continue

            # Create signal with all necessary data
            signal = {
                'symbol': symbol,
                'prediction': pred,
                'current_price': pred['latest_price'],
                'atr': pred['atr'],
                'volatility': recent_data['volatility_20d'].iloc[-1] if 'volatility_20d' in recent_data else 0.20,
                'dollar_volume': avg_dollar_volume,
                'score': pred['confidence'] * pred['model_agreement'],
                'date': date
            }

            valid_signals.append(signal)

        # Sort by score
        valid_signals.sort(key=lambda x: x['score'], reverse=True)

        # Log signal generation
        if valid_signals:
            logger.debug(f"Generated {len(valid_signals)} valid signals, top score: {valid_signals[0]['score']:.3f}")

        return valid_signals

    def _execute_entry(self, signal: Dict, date: pd.Timestamp,
                       all_data: Dict[str, pd.DataFrame],
                       cash: float, portfolio_value: float) -> float:
        """Execute entry with realistic simulation"""

        symbol = signal['symbol']

        # Calculate position size
        shares = self.risk_manager.calculate_position_size(
            symbol, signal['score'], portfolio_value,
            all_data[symbol], self.positions
        )

        if shares == 0:
            logger.debug(f"Position size is 0 for {symbol}")
            return 0

        # Check risk limits
        risk_check = self.risk_manager.check_portfolio_risk(
            self.positions, all_data, portfolio_value
        )

        if not risk_check['pass']:
            logger.debug(f"Risk check failed for {symbol}: {risk_check['messages']}")
            return 0

        # Simulate execution
        exec_price, slippage, commission = self.execution_sim.simulate_entry(
            symbol, signal['current_price'], shares, signal['dollar_volume']
        )

        total_cost = (exec_price * shares) + slippage + commission

        if total_cost > cash * 0.95:  # Keep 5% cash buffer
            logger.debug(f"Insufficient cash for {symbol}: need ${total_cost:.2f}, have ${cash:.2f}")
            return 0

        # Calculate risk levels
        stop_loss = self.risk_manager.calculate_stop_loss(
            exec_price, signal['atr']
        )
        take_profit = self.risk_manager.calculate_take_profit(
            exec_price, signal['atr']
        )

        # Create position
        self.positions[symbol] = {
            'entry_date': date,
            'entry_price': exec_price,
            'quantity': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': exec_price,
            'current_value': exec_price * shares,
            'commission_paid': commission,
            'slippage_paid': slippage,
            'prediction': signal['prediction']
        }

        # Record trade
        self.trades.append({
            'symbol': symbol,
            'date': date,
            'action': 'BUY',
            'price': exec_price,
            'quantity': shares,
            'value': total_cost,
            'commission': commission,
            'slippage': slippage,
            'ml_confidence': signal['prediction']['confidence'],
            'ml_agreement': signal['prediction']['model_agreement'],
            'ml_probability': signal['prediction']['probability']
        })

        logger.info(f"ENTRY: {symbol} - {shares} shares @ ${exec_price:.2f} "
                    f"(confidence: {signal['prediction']['confidence']:.3f})")

        return total_cost

    def _check_exit_conditions(self, date: pd.Timestamp,
                               all_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Check for exit conditions"""

        exits = []

        for symbol, position in self.positions.items():
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            if date not in df.index:
                continue

            current_price = df.loc[date, 'close']

            # Update current price
            position['current_price'] = current_price
            position['current_value'] = current_price * position['quantity']

            # Check stop loss
            if current_price <= position['stop_loss']:
                exits.append({
                    'symbol': symbol,
                    'reason': 'stop_loss',
                    'exit_price': current_price
                })

            # Check take profit
            elif current_price >= position['take_profit']:
                exits.append({
                    'symbol': symbol,
                    'reason': 'take_profit',
                    'exit_price': current_price
                })

            # Check time-based exit (optional)
            elif (date - position['entry_date']).days > 20:
                # Re-evaluate position
                current_return = (current_price - position['entry_price']) / position['entry_price']
                if current_return < -0.02:  # Down more than 2%
                    exits.append({
                        'symbol': symbol,
                        'reason': 'time_stop',
                        'exit_price': current_price
                    })

        return exits

    def _execute_exit(self, exit_info: Dict, date: pd.Timestamp,
                      all_data: Dict[str, pd.DataFrame]) -> float:
        """Execute exit and return proceeds"""

        symbol = exit_info['symbol']
        if symbol not in self.positions:
            return 0

        position = self.positions[symbol]

        # Get current market data
        df = all_data[symbol]
        current_data = df[df.index <= date].tail(1)

        if current_data.empty:
            return 0

        exit_price = exit_info.get('exit_price', current_data['close'].iloc[0])
        daily_volume = current_data['volume'].iloc[0] * exit_price

        # Simulate execution
        exec_price, slippage, commission = self.execution_sim.simulate_exit(
            symbol, exit_price, position['quantity'], daily_volume
        )

        # Calculate proceeds
        gross_proceeds = exec_price * position['quantity']
        net_proceeds = gross_proceeds - slippage - commission

        # Calculate P&L
        entry_cost = position['entry_price'] * position['quantity'] + position['commission_paid'] + position[
            'slippage_paid']
        total_pnl = net_proceeds - entry_cost
        pnl_pct = total_pnl / entry_cost

        # Record trade
        self.trades.append({
            'symbol': symbol,
            'date': date,
            'action': 'SELL',
            'price': exec_price,
            'quantity': position['quantity'],
            'value': gross_proceeds,
            'commission': commission,
            'slippage': slippage,
            'pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_info['reason'],
            'holding_days': (date - position['entry_date']).days
        })

        # Remove position
        del self.positions[symbol]

        logger.info(f"EXIT: {symbol} - {exit_info['reason']} - P&L: ${total_pnl:.2f} ({pnl_pct:.2%})")

        return net_proceeds

    def _update_portfolio_prices(self, date: pd.Timestamp,
                                 all_data: Dict[str, pd.DataFrame]):
        """Update current prices for all positions"""

        for symbol, position in self.positions.items():
            if symbol in all_data:
                df = all_data[symbol]
                if date in df.index:
                    current_price = df.loc[date, 'close']
                    position['current_price'] = current_price
                    position['current_value'] = current_price * position['quantity']

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if not self.equity_curve:
            return {'error': 'No data to analyze'}

        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)

        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()

        # Basic metrics
        total_return = (equity_df['portfolio_value'].iloc[
                            -1] - self.config.initial_capital) / self.config.initial_capital

        # Risk metrics
        returns = equity_df['returns'].dropna()
        if len(returns) > 0:
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            # Sortino
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino = annual_return / downside_vol if downside_vol > 0 else 0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calmar
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            annual_return = annual_vol = sharpe = sortino = max_drawdown = calmar = 0

        # Trade analysis
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if len(sell_trades) > 0:
                winning_trades = sell_trades[sell_trades['pnl'] > 0]
                losing_trades = sell_trades[sell_trades['pnl'] <= 0]

                win_rate = len(winning_trades) / len(sell_trades)
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(
                    losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else float('inf')

                # ML performance
                ml_metrics = {
                    'avg_ml_confidence': buy_trades['ml_confidence'].mean() if 'ml_confidence' in buy_trades else 0,
                    'avg_ml_agreement': buy_trades['ml_agreement'].mean() if 'ml_agreement' in buy_trades else 0,
                    'avg_ml_probability': buy_trades['ml_probability'].mean() if 'ml_probability' in buy_trades else 0,
                    'high_confidence_trades': len(
                        buy_trades[buy_trades['ml_confidence'] > 0.6]) if 'ml_confidence' in buy_trades else 0
                }
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
                ml_metrics = {}
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            ml_metrics = {}

        # Compile results
        results = {
            # Returns
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,

            # Risk metrics
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,

            # Trade metrics
            'total_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # ML metrics
            'ml_performance': ml_metrics,

            # Data
            'equity_curve': equity_df,
            'trades': trades_df
        }

        return results

    def _generate_reports(self, results: Dict):
        """Generate comprehensive reports"""

        # Print summary
        print("\n" + "=" * 80)
        print("HEDGE FUND ML BACKTEST RESULTS")
        print("=" * 80)
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        print(f"Annual Return: {results['annual_return'] * 100:.2f}%")
        print(f"Annual Volatility: {results['annual_volatility'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
        print(f"\nTotal Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate'] * 100:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")

        if results['ml_performance']:
            print(f"\nML Performance:")
            print(f"Avg Confidence: {results['ml_performance'].get('avg_ml_confidence', 0) * 100:.1f}%")
            print(f"Avg Model Agreement: {results['ml_performance'].get('avg_ml_agreement', 0) * 100:.1f}%")
            print(f"Avg Probability: {results['ml_performance'].get('avg_ml_probability', 0) * 100:.1f}%")

        print("=" * 80)

        # Generate plots
        self._plot_results(results)

        # Save detailed results
        results_to_save = {k: v for k, v in results.items()
                           if k not in ['equity_curve', 'trades']}

        with open('hedge_fund_backtest_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)

        # Save trades
        if 'trades' in results and isinstance(results['trades'], pd.DataFrame):
            results['trades'].to_csv('hedge_fund_trades.csv', index=False)

    def _plot_results(self, results: Dict):
        """Generate performance plots"""

        if 'equity_curve' not in results:
            return

        equity_df = results['equity_curve']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Equity curve
        ax1 = axes[0, 0]
        ax1.plot(equity_df['date'], equity_df['portfolio_value'], 'b-', linewidth=2)
        ax1.axhline(self.config.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Portfolio Value', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.grid(True, alpha=0.3)

        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # 2. Drawdown
        ax2 = axes[0, 1]
        returns = equity_df['returns'].fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        ax2.fill_between(equity_df['date'], 0, drawdown, color='red', alpha=0.3)
        ax2.plot(equity_df['date'], drawdown, 'r-', linewidth=1)
        ax2.set_title('Drawdown %', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Monthly returns
        ax3 = axes[1, 0]
        monthly_returns = equity_df.set_index('date')['portfolio_value'].resample('M').last().pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        ax3.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.7)
        ax3.set_title('Monthly Returns %', fontsize=14)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)

        # 4. Position count
        ax4 = axes[1, 1]
        ax4.plot(equity_df['date'], equity_df['n_positions'], 'g-', linewidth=2)
        ax4.set_title('Active Positions', fontsize=14)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Number of Positions')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, self.config.max_positions + 2)

        plt.tight_layout()
        plt.savefig('hedge_fund_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("Results plotted: hedge_fund_backtest_results.png")


class RiskManager:
    """Professional risk management system"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.correlation_matrix = None
        self.risk_metrics = defaultdict(list)

    def calculate_position_size(self, symbol: str, signal_strength: float,
                                portfolio_value: float, market_data: pd.DataFrame,
                                current_positions: Dict) -> int:
        """Calculate position size using risk parity approach"""

        if self.config.position_size_method == "fixed":
            position_value = portfolio_value * self.config.base_position_size

        elif self.config.position_size_method == "risk_parity":
            # Get volatility
            volatility = market_data['volatility_20d'].iloc[-1] if 'volatility_20d' in market_data else 0.20

            # Risk budget per position
            risk_budget = portfolio_value * self.config.base_position_size * 0.02  # 2% risk

            # Position size based on volatility
            if volatility > 0:
                position_value = risk_budget / volatility
            else:
                position_value = portfolio_value * self.config.base_position_size

        elif self.config.position_size_method == "kelly":
            # Simplified Kelly criterion
            win_rate = 0.55  # Estimate from historical performance
            avg_win = 0.03
            avg_loss = 0.02

            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_pct = np.clip(kelly_pct * 0.25, 0.01, self.config.max_position_size)  # 1/4 Kelly

            position_value = portfolio_value * kelly_pct

        # Apply signal strength adjustment
        position_value *= signal_strength

        # Apply maximum position size
        max_value = portfolio_value * self.config.max_position_size
        position_value = min(position_value, max_value)

        # Calculate shares
        current_price = market_data['close'].iloc[-1]
        shares = int(position_value / current_price)

        return shares

    def check_portfolio_risk(self, positions: Dict, market_data: Dict[str, pd.DataFrame],
                             portfolio_value: float) -> Dict:
        """Check portfolio-level risk metrics"""

        risk_checks = {
            'pass': True,
            'portfolio_heat': 0,
            'correlation_risk': False,
            'concentration_risk': False,
            'messages': []
        }

        if not positions:
            return risk_checks

        # Calculate portfolio heat (total risk)
        total_risk = 0
        position_values = {}

        for symbol, position in positions.items():
            if symbol in market_data:
                atr = market_data[symbol]['atr'].iloc[-1] if 'atr' in market_data[symbol] else 0
                position_value = position['quantity'] * position['current_price']
                position_risk = (atr * position['quantity']) / portfolio_value
                total_risk += position_risk
                position_values[symbol] = position_value

        risk_checks['portfolio_heat'] = total_risk

        if total_risk > self.config.max_portfolio_heat:
            risk_checks['pass'] = False
            risk_checks['messages'].append(f"Portfolio heat too high: {total_risk:.2%}")

        # Check correlation risk
        if len(positions) > 3:
            corr_risk = self._check_correlation_risk(list(positions.keys()), market_data)
            if corr_risk:
                risk_checks['correlation_risk'] = True
                risk_checks['messages'].append("High correlation between positions")

        # Check concentration
        total_value = sum(position_values.values())
        if total_value > 0:
            for symbol, value in position_values.items():
                if value / total_value > 0.15:  # 15% max per position
                    risk_checks['concentration_risk'] = True
                    risk_checks['messages'].append(f"Position too large: {symbol}")

        return risk_checks

    def _check_correlation_risk(self, symbols: List[str],
                                market_data: Dict[str, pd.DataFrame]) -> bool:
        """Check if positions are too correlated"""

        # Get returns for correlation calculation
        returns_data = {}
        for symbol in symbols:
            if symbol in market_data:
                returns = market_data[symbol]['returns'].tail(60) if 'returns' in market_data[symbol] else None
                if returns is not None and len(returns) >= 60:
                    returns_data[symbol] = returns

        if len(returns_data) < 2:
            return False

        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()

        # Check for high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.config.correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        return len(high_corr_pairs) > 0

    def calculate_stop_loss(self, entry_price: float, atr: float, side: str = 'long') -> float:
        """Calculate dynamic stop loss based on ATR"""
        stop_distance = atr * self.config.stop_loss_atr_multiplier

        if side == 'long':
            stop_loss = entry_price - stop_distance
        else:  # Short positions (if implemented)
            stop_loss = entry_price + stop_distance

        return stop_loss

    def calculate_take_profit(self, entry_price: float, atr: float, side: str = 'long') -> float:
        """Calculate dynamic take profit based on ATR"""
        profit_distance = atr * self.config.take_profit_atr_multiplier

        if side == 'long':
            take_profit = entry_price + profit_distance
        else:  # Short positions (if implemented)
            take_profit = entry_price - profit_distance

        return take_profit


def run_hedge_fund_backtest():
    """Run the hedge fund grade backtest"""

    # Configuration - OPTIMIZED FOR SIGNAL GENERATION
    config = HedgeFundBacktestConfig(
        initial_capital=100000,
        position_size_method="risk_parity",
        max_positions=20,
        min_prediction_confidence=0.55,  # Lowered
        ensemble_agreement_threshold=0.50,  # Lowered
        target_return_threshold=0.01,  # 1% instead of 2%
        min_sharpe_for_trading=0.5,  # Lowered
        min_validation_score=0.52  # Lowered
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Define test period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year backtest

    # Run on full watchlist
    logger.info(f"Running backtest on {len(WATCHLIST)} symbols")
    results = backtester.run_backtest(
        symbols=WATCHLIST[:50],  # Start with first 50 for testing
        start_date=start_date,
        end_date=end_date
    )

    return results


if __name__ == "__main__":
    results = run_hedge_fund_backtest()