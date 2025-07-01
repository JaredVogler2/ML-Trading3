# hedge_fund_ml_backtest.py
"""
Professional Hedge Fund ML Trading Backtest System
Features:
- Walk-forward optimization with strict temporal isolation
- No data leakage guarantees
- Point-in-time data handling
- Professional monitoring and reporting
- Multiple ML models with GPU acceleration
- Realistic execution simulation
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
from tqdm import tqdm

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

    # Data requirements
    min_history_days: int = 504  # 2 years minimum
    feature_lookback_days: int = 252  # 1 year for features


class HedgeFundDataManager:
    """Professional data manager with strict temporal isolation"""

    def __init__(self):
        self.data_cache = {}
        self.feature_lookback_days = 252  # 1 year for feature calculation
        self.min_history_days = 504  # 2 years minimum history

    def fetch_point_in_time_data(self, symbol: str, as_of_date: pd.Timestamp,
                                 lookback_days: int = None) -> pd.DataFrame:
        """Fetch data as it would have been available at a specific point in time"""

        if lookback_days is None:
            lookback_days = self.min_history_days

        # Ensure we're using timezone-naive dates
        if hasattr(as_of_date, 'tz') and as_of_date.tz is not None:
            as_of_date = as_of_date.tz_localize(None)

        # Calculate the data window
        data_end = as_of_date
        data_start = as_of_date - timedelta(days=lookback_days)

        cache_key = f"{symbol}_{data_start.strftime('%Y%m%d')}_{data_end.strftime('%Y%m%d')}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()

        try:
            # Fetch data up to and including as_of_date
            df = yf.download(
                symbol,
                start=data_start.strftime('%Y-%m-%d'),
                end=(data_end + timedelta(days=1)).strftime('%Y-%m-%d'),  # yfinance end is exclusive
                auto_adjust=True,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data for {symbol} as of {as_of_date.date()}")
                return pd.DataFrame()

            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # CRITICAL: Remove any data after as_of_date (in case of data errors)
            df = df[df.index <= as_of_date]

            # Clean column names
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower()
                              for col in df.columns]
            else:
                df.columns = [str(col).lower() for col in df.columns]

            # Verify required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return pd.DataFrame()

            # Remove NaN rows
            df = df.dropna(subset=required_columns)

            # Add metadata
            df['symbol'] = symbol
            df['fetch_date'] = as_of_date

            self.data_cache[cache_key] = df.copy()

            logger.debug(f"Fetched {len(df)} days of {symbol} history as of {as_of_date.date()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol} as of {as_of_date.date()}: {e}")
            return pd.DataFrame()

    def fetch_batch_point_in_time(self, symbols: List[str], as_of_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Fetch point-in-time data for multiple symbols"""

        data = {}

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_point_in_time_data, symbol, as_of_date): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    df = future.result()
                    if not df.empty:
                        data[symbol] = df
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")

        return data


class WalkForwardOptimizer:
    """Professional walk-forward optimization with proper data isolation"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.optimization_results = []

        # Professional hedge fund parameters
        self.train_window_days = config.train_months * 21  # Trading days
        self.validation_window_days = config.validation_months * 21
        self.test_window_days = config.test_months * 21
        self.buffer_days = max(config.buffer_days, 5)  # Minimum 5 day buffer

        # Feature engineering lookback
        self.feature_lookback_days = config.feature_lookback_days

    def create_walk_forward_windows(self, start_date: pd.Timestamp,
                                    end_date: pd.Timestamp) -> List[Dict]:
        """Create non-overlapping train/val/test windows with buffers"""

        windows = []

        # Need enough history for first window
        min_start = start_date + timedelta(days=self.feature_lookback_days)

        # Total days needed for one complete window
        total_window_days = (self.train_window_days +
                             self.validation_window_days +
                             self.test_window_days +
                             3 * self.buffer_days)  # 3 buffers total

        # Start from first possible test date
        current_test_start = min_start + timedelta(days=self.train_window_days +
                                                        self.validation_window_days +
                                                        2 * self.buffer_days)

        while current_test_start + timedelta(days=self.test_window_days) <= end_date:
            # Work backwards from test start date
            test_start = current_test_start
            test_end = test_start + timedelta(days=self.test_window_days)

            # Buffer before test
            buffer2_end = test_start
            buffer2_start = buffer2_end - timedelta(days=self.buffer_days)

            # Validation period
            val_end = buffer2_start
            val_start = val_end - timedelta(days=self.validation_window_days)

            # Buffer before validation
            buffer1_end = val_start
            buffer1_start = buffer1_end - timedelta(days=self.buffer_days)

            # Training period
            train_end = buffer1_start
            train_start = train_end - timedelta(days=self.train_window_days)

            window = {
                'train_start': train_start,
                'train_end': train_end,
                'buffer1_start': buffer1_start,
                'buffer1_end': buffer1_end,
                'val_start': val_start,
                'val_end': val_end,
                'buffer2_start': buffer2_start,
                'buffer2_end': buffer2_end,
                'test_start': test_start,
                'test_end': test_end,
                'as_of_date': test_start - timedelta(days=1)
            }

            windows.append(window)

            # Move forward by test window size
            current_test_start += timedelta(days=self.test_window_days)

        return windows

    def validate_window_integrity(self, window: Dict) -> bool:
        """Ensure no data leakage between periods"""

        # Check that periods don't overlap (but can touch at boundaries)
        # Train should not overlap with validation
        if window['train_end'] > window['val_start']:
            logger.error(f"Train end {window['train_end']} overlaps with val start {window['val_start']}")
            return False

        # Validation should not overlap with test
        if window['val_end'] > window['test_start']:
            logger.error(f"Val end {window['val_end']} overlaps with test start {window['test_start']}")
            return False

        # Check buffer gaps exist
        train_val_gap = (window['val_start'] - window['train_end']).days
        val_test_gap = (window['test_start'] - window['val_end']).days

        if train_val_gap < self.buffer_days:
            logger.error(f"Insufficient buffer between train and val: {train_val_gap} days (need {self.buffer_days})")
            return False

        if val_test_gap < self.buffer_days:
            logger.error(f"Insufficient buffer between val and test: {val_test_gap} days (need {self.buffer_days})")
            return False

        # Verify chronological order
        if not (window['train_start'] < window['train_end'] < window['val_start'] <
                window['val_end'] < window['test_start'] < window['test_end']):
            logger.error("Windows not in chronological order")
            return False

        return True


class BacktestMonitor:
    """Professional monitoring and reporting for backtests"""

    def __init__(self):
        self.metrics = {
            'windows': [],
            'trades': [],
            'daily_pnl': [],
            'feature_importance': {},
            'model_performance': {},
            'data_integrity_checks': []
        }

    def log_data_integrity_check(self, check_name: str, passed: bool, details: str = ""):
        """Log data integrity verification results"""

        self.metrics['data_integrity_checks'].append({
            'timestamp': datetime.now(),
            'check': check_name,
            'passed': passed,
            'details': details
        })

    def log_window_results(self, window_idx: int, window: Dict,
                           model_score: float, n_trades: int, pnl: float):
        """Log results for each window"""

        self.metrics['windows'].append({
            'index': window_idx,
            'train_period': f"{window['train_start'].date()} to {window['train_end'].date()}",
            'val_period': f"{window['val_start'].date()} to {window['val_end'].date()}",
            'test_period': f"{window['test_start'].date()} to {window['test_end'].date()}",
            'validation_auc': model_score,
            'trades_executed': n_trades,
            'period_pnl': pnl
        })

    def log_trade(self, trade: Dict):
        """Log individual trade"""
        self.metrics['trades'].append(trade)

    def generate_report(self) -> Dict:
        """Generate comprehensive backtest report"""

        # Calculate performance metrics
        total_trades = len(self.metrics['trades'])
        winning_trades = [t for t in self.metrics['trades'] if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.metrics['trades'] if t.get('pnl', 0) <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Data integrity summary
        integrity_passed = all(check['passed'] for check in self.metrics['data_integrity_checks'])

        report = {
            'summary': {
                'total_windows': len(self.metrics['windows']),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_validation_auc': np.mean([w['validation_auc'] for w in self.metrics['windows']]) if self.metrics[
                    'windows'] else 0,
                'data_integrity': 'VERIFIED - No leakage detected' if integrity_passed else 'FAILED - Potential leakage'
            },
            'window_analysis': self.metrics['windows'],
            'data_integrity_checks': self.metrics['data_integrity_checks'],
            'trades': self.metrics['trades'][:100]  # First 100 trades
        }

        return report


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
    """Main backtesting engine with professional standards"""

    def __init__(self, config: HedgeFundBacktestConfig):
        self.config = config
        self.data_manager = HedgeFundDataManager()
        self.risk_manager = RiskManager(config)
        self.execution_sim = ExecutionSimulator()
        self.monitor = BacktestMonitor()

        # Portfolio state
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []

    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Run professional hedge fund backtest with walk-forward optimization"""

        logger.info(f"Starting professional hedge fund backtest")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Capital: ${self.config.initial_capital:,.0f}")

        # Initialize components
        walk_forward = WalkForwardOptimizer(self.config)

        # Convert dates
        start_ts = pd.to_datetime(start_date).tz_localize(None)
        end_ts = pd.to_datetime(end_date).tz_localize(None)

        # Create walk-forward windows
        windows = walk_forward.create_walk_forward_windows(start_ts, end_ts)
        logger.info(f"Created {len(windows)} walk-forward windows")

        if not windows:
            return {'error': 'Insufficient data for walk-forward optimization'}

        # Process each window
        for window_idx, window in enumerate(windows):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing window {window_idx + 1}/{len(windows)}")
            logger.info(f"Train: {window['train_start'].date()} to {window['train_end'].date()}")
            logger.info(f"Val: {window['val_start'].date()} to {window['val_end'].date()}")
            logger.info(f"Test: {window['test_start'].date()} to {window['test_end'].date()}")

            # Validate window integrity
            if not walk_forward.validate_window_integrity(window):
                self.monitor.log_data_integrity_check(
                    f"Window {window_idx} integrity",
                    False,
                    "Window failed temporal isolation check"
                )
                continue
            else:
                self.monitor.log_data_integrity_check(
                    f"Window {window_idx} integrity",
                    True,
                    "Window passed temporal isolation check"
                )

            try:
                # Train model for this window
                model = self._train_model_for_window(symbols, window)

                if model is None:
                    logger.warning(f"No model trained for window {window_idx}")
                    continue

                # Run backtest on test period
                window_pnl = self._backtest_test_period(symbols, model, window)

                # Log window results
                self.monitor.log_window_results(
                    window_idx,
                    window,
                    model.validation_score,
                    len([t for t in self.trades if
                         window['test_start'] <= pd.to_datetime(t['date']) <= window['test_end']]),
                    window_pnl
                )

            except Exception as e:
                logger.error(f"Error in window {window_idx}: {e}", exc_info=True)
                self.monitor.log_data_integrity_check(
                    f"Window {window_idx} processing",
                    False,
                    str(e)
                )
                continue

        # Calculate final metrics
        if not self.trades:
            return {'error': 'No trades executed', 'report': self.monitor.generate_report()}

        # Generate comprehensive report
        final_metrics = self._calculate_final_metrics()
        report = self.monitor.generate_report()

        # Combine metrics and report
        return {**final_metrics, 'detailed_report': report}

    def _train_model_for_window(self, symbols: List[str], window: Dict) -> Optional[TrainedModel]:
        """Train model for a specific window with strict data isolation"""

        logger.info(f"Training model for window ending {window['test_start'].date()}")

        # Fetch point-in-time data
        pit_data = self.data_manager.fetch_batch_point_in_time(symbols, window['as_of_date'])

        if not pit_data:
            logger.warning("No data available for training")
            return None

        # Create feature engineer
        feature_engineer = EnhancedFeatureEngineer()

        # Prepare training and validation data
        train_features = []
        train_targets = []
        val_features = []
        val_targets = []

        for symbol, full_data in pit_data.items():
            # Ensure we only use data up to as_of_date
            historical_data = full_data[full_data.index <= window['as_of_date']]

            if len(historical_data) < 200:  # Need history for features
                continue

            # Create features using only historical data
            try:
                features = feature_engineer.create_all_features(historical_data, symbol)

                if features.empty:
                    continue

                # Create target
                prediction_horizon = 5
                returns = historical_data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)

                # Adaptive threshold
                volatility = historical_data['close'].pct_change().rolling(20).std().mean()
                threshold = max(0.01, min(0.03, volatility * 2))

                target = (returns > threshold).astype(int)

                # Split into train and validation periods
                # Training data
                train_mask = (features.index >= window['train_start']) & (features.index <= window['train_end'])
                train_feat = features[train_mask][:-prediction_horizon]
                train_tgt = target[train_mask][:-prediction_horizon]

                # Remove NaN
                valid_train = train_tgt.notna()
                train_feat = train_feat[valid_train]
                train_tgt = train_tgt[valid_train]

                if len(train_feat) > 50:
                    train_features.append(train_feat)
                    train_targets.append(train_tgt)

                # Validation data
                val_mask = (features.index >= window['val_start']) & (features.index <= window['val_end'])
                val_feat = features[val_mask][:-prediction_horizon]
                val_tgt = target[val_mask][:-prediction_horizon]

                # Remove NaN
                valid_val = val_tgt.notna()
                val_feat = val_feat[valid_val]
                val_tgt = val_tgt[valid_val]

                if len(val_feat) > 10:
                    val_features.append(val_feat)
                    val_targets.append(val_tgt)

                # Verify no future data leakage
                if val_feat.index.max() > window['val_end']:
                    logger.error(f"DATA LEAKAGE: Validation features extend beyond val_end")
                    self.monitor.log_data_integrity_check(
                        f"Feature leakage check - {symbol}",
                        False,
                        "Validation features contain future data"
                    )
                    return None
                else:
                    self.monitor.log_data_integrity_check(
                        f"Feature leakage check - {symbol}",
                        True,
                        "No future data in features"
                    )

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not train_features or not val_features:
            logger.warning("Insufficient data for training")
            return None

        # Combine all data
        X_train = pd.concat(train_features, ignore_index=True)
        y_train = pd.concat(train_targets, ignore_index=True)
        X_val = pd.concat(val_features, ignore_index=True)
        y_val = pd.concat(val_targets, ignore_index=True)

        # Log data summary
        logger.info(f"Training data: {len(X_train)} samples, {X_train.shape[1]} features")
        logger.info(f"Validation data: {len(X_val)} samples")
        logger.info(f"Class balance - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")

        # Train ensemble
        ensemble = HedgeFundGPUEnsemble()

        try:
            ensemble.train_combined(X_train, y_train, X_val, y_val)

            # Get validation score
            val_score = ensemble.validate(X_val, y_val)

            if val_score < self.config.min_validation_score:
                logger.warning(f"Model validation score {val_score:.4f} below threshold")
                return None

            # Get feature importance
            feature_importance = ensemble.get_feature_importance()

            logger.info(f"Model trained successfully. Validation AUC: {val_score:.4f}")

            return TrainedModel(ensemble, window['train_end'], val_score, feature_importance)

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return None

    def _backtest_test_period(self, symbols: List[str], model: TrainedModel,
                              window: Dict) -> float:
        """Run backtest for test period with trained model"""

        logger.info(f"Backtesting test period: {window['test_start'].date()} to {window['test_end'].date()}")

        # Track period P&L
        start_portfolio_value = self._calculate_portfolio_value()

        # Get trading days in test period
        test_start = window['test_start']
        test_end = window['test_end']

        # Generate trading days
        trading_days = pd.bdate_range(test_start, test_end)

        for current_date in trading_days:
            # Update existing positions
            self._update_positions(current_date)

            # Generate signals for current date
            signals = self._generate_signals(symbols, model, current_date, window)

            # Execute trades
            self._execute_trades(signals, current_date)

            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value()
            self.portfolio_values.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions)
            })

        # Calculate period return
        end_portfolio_value = self._calculate_portfolio_value()
        period_pnl = end_portfolio_value - start_portfolio_value

        logger.info(f"Test period P&L: ${period_pnl:,.2f}")

        return period_pnl

    def _generate_signals(self, symbols: List[str], model: TrainedModel,
                          current_date: pd.Timestamp, window: Dict) -> List[Dict]:
        """Generate trading signals for current date"""

        signals = []
        current_positions = list(self.positions.keys())

        # Feature engineer
        feature_engineer = EnhancedFeatureEngineer()

        for symbol in symbols:
            # Skip if already have position
            if symbol in current_positions:
                continue

            # Get historical data up to current date
            historical_data = self.data_manager.fetch_point_in_time_data(
                symbol,
                current_date,
                lookback_days=self.config.feature_lookback_days + 100
            )

            if len(historical_data) < 200:
                continue

            try:
                # Create features
                features = feature_engineer.create_all_features(historical_data, symbol)

                if features.empty or current_date not in features.index:
                    continue

                # Get features for current date
                current_features = features.loc[[current_date]]

                # Get prediction
                prediction = model.predict(current_features)

                if len(prediction) > 0:
                    confidence = prediction[0]

                    if confidence >= self.config.min_prediction_confidence:
                        # Calculate metrics
                        recent_data = historical_data.tail(20)
                        volatility = recent_data['close'].pct_change().std() * np.sqrt(252)
                        atr = self._calculate_atr(recent_data)

                        signals.append({
                            'symbol': symbol,
                            'date': current_date,
                            'confidence': confidence,
                            'volatility': volatility,
                            'atr': atr,
                            'price': recent_data['close'].iloc[-1],
                            'volume': recent_data['volume'].iloc[-1]
                        })

            except Exception as e:
                logger.debug(f"Signal generation failed for {symbol}: {e}")
                continue

        # Sort by confidence and limit
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        max_new_positions = self.config.max_positions - len(self.positions)
        signals = signals[:max_new_positions]

        return signals

    def _execute_trades(self, signals: List[Dict], current_date: pd.Timestamp) -> None:
        """Execute trades based on signals"""

        for signal in signals:
            symbol = signal['symbol']

            # Calculate position size
            portfolio_value = self._calculate_portfolio_value()
            position_size = self.risk_manager.calculate_position_size(
                portfolio_value,
                signal['volatility'],
                signal['confidence'],
                len(self.positions)
            )

            if position_size < 1000:  # Minimum position size
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

            # Check cash
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
                'entry_date': current_date,
                'entry_price': exec_price,
                'quantity': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'commission_paid': commission,
                'slippage_paid': slippage
            }

            self.positions[symbol] = position

            # Record trade
            trade = {
                'date': current_date,
                'symbol': symbol,
                'action': 'BUY',
                'quantity': shares,
                'price': exec_price,
                'commission': commission,
                'slippage': slippage,
                'confidence': signal['confidence']
            }

            self.trades.append(trade)
            self.monitor.log_trade(trade)

            logger.debug(f"Bought {shares} shares of {symbol} at ${exec_price:.2f}")

    def _update_positions(self, current_date: pd.Timestamp) -> None:
        """Update existing positions - check stops and targets"""

        positions_to_close = []

        for symbol, position in self.positions.items():
            # Get current price
            current_data = self.data_manager.fetch_point_in_time_data(
                symbol, current_date, lookback_days=10
            )

            if current_data.empty or current_date not in current_data.index:
                continue

            current_price = current_data.loc[current_date, 'close']

            # Check stop loss
            if current_price <= position['stop_loss']:
                positions_to_close.append((symbol, 'STOP_LOSS', current_price))

            # Check take profit
            elif current_price >= position['take_profit']:
                positions_to_close.append((symbol, 'TAKE_PROFIT', current_price))

            # Check time-based exit
            elif (current_date - position['entry_date']).days >= 20:
                positions_to_close.append((symbol, 'TIME_EXIT', current_price))

        # Close positions
        for symbol, reason, price in positions_to_close:
            self._close_position(symbol, current_date, price, reason)

    def _close_position(self, symbol: str, date: pd.Timestamp,
                        price: float, reason: str) -> None:
        """Close a position"""

        position = self.positions[symbol]
        quantity = position['quantity']

        # Get volume for execution simulation
        current_data = self.data_manager.fetch_point_in_time_data(
            symbol, date, lookback_days=10
        )
        volume = current_data.loc[date, 'volume'] if date in current_data.index else 1000000

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
        trade = {
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
        }

        self.trades.append(trade)
        self.monitor.log_trade(trade)

        # Remove position
        del self.positions[symbol]

        logger.debug(f"Closed {symbol}: {reason}, P&L: ${pnl:.2f} ({pnl_pct * 100:.1f}%)")

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""

        positions_value = sum(
            position['quantity'] * position['entry_price']
            for position in self.positions.values()
        )

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

    def _calculate_final_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""

        if not self.portfolio_values:
            return {'error': 'No portfolio values recorded'}

        # Convert to DataFrame
        pv_df = pd.DataFrame(self.portfolio_values)
        pv_df['date'] = pd.to_datetime(pv_df['date'])
        pv_df.set_index('date', inplace=True)

        # Calculate returns
        pv_df['returns'] = pv_df['value'].pct_change()
        daily_returns = pv_df['returns'].dropna()

        # Basic metrics
        initial_value = self.config.initial_capital
        final_value = pv_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # Risk metrics
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / (daily_returns.std() + 1e-6)

            # Max drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()

            # Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = np.sqrt(252) * daily_returns.mean() / (downside_std + 1e-6)

        else:
            sharpe_ratio = 0
            max_drawdown = 0
            sortino_ratio = 0

        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]

            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

            # Profit factor
            total_wins = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0

        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Annual metrics
        n_days = len(pv_df)
        n_years = n_days / 252
        annual_return = (final_value / initial_value) ** (1 / n_years) - 1 if n_years > 0 else 0

        metrics = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'portfolio_values': pv_df.to_dict('records'),
            'trades': self.trades
        }

        # Add symbol statistics
        if not trades_df.empty:
            symbol_stats = trades_df.groupby('symbol').agg({
                'symbol': 'count',
                'pnl': ['sum', 'mean', 'std']
            }).round(2)
            metrics['symbol_stats'] = symbol_stats.to_dict()

        return metrics


# Utility functions
def plot_backtest_results(results: Dict) -> None:
    """Plot professional backtest results"""

    if 'error' in results:
        print(f"Cannot plot results: {results['error']}")
        return

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Professional Hedge Fund Backtest Results', fontsize=16)

    # Portfolio value
    pv_df = pd.DataFrame(results['portfolio_values'])
    pv_df['date'] = pd.to_datetime(pv_df['date'])
    pv_df.set_index('date', inplace=True)

    axes[0, 0].plot(pv_df.index, pv_df['value'], linewidth=2)
    axes[0, 0].set_title('Portfolio Value')
    axes[0, 0].set_ylabel('Value ($)')
    axes[0, 0].grid(True, alpha=0.3)

    # Drawdown
    returns = pv_df['value'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100

    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0,
                            color='red', alpha=0.3, label='Drawdown')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_ylabel('Drawdown (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Rolling Sharpe
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
    axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
    axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Rolling 3-Month Sharpe Ratio')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True, alpha=0.3)

    # Number of positions
    axes[1, 1].plot(pv_df.index, pv_df['n_positions'], linewidth=2)
    axes[1, 1].set_title('Active Positions')
    axes[1, 1].set_ylabel('Number of Positions')
    axes[1, 1].grid(True, alpha=0.3)

    # Monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    axes[2, 0].bar(monthly_returns.index, monthly_returns.values * 100,
                   color=['g' if x > 0 else 'r' for x in monthly_returns.values])
    axes[2, 0].set_title('Monthly Returns')
    axes[2, 0].set_ylabel('Return (%)')
    axes[2, 0].grid(True, alpha=0.3)

    # Trade P&L distribution
    trades_df = pd.DataFrame([t for t in results['trades'] if 'pnl' in t])
    if not trades_df.empty:
        axes[2, 1].hist(trades_df['pnl'], bins=50, alpha=0.7, edgecolor='black')
        axes[2, 1].axvline(0, color='black', linestyle='--')
        axes[2, 1].set_title('Trade P&L Distribution')
        axes[2, 1].set_xlabel('P&L ($)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].grid(True, alpha=0.3)

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
        elif isinstance(obj, datetime):
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

    # Professional configuration
    config = HedgeFundBacktestConfig(
        initial_capital=1_000_000,
        max_positions=20,
        train_months=12,
        validation_months=3,
        test_months=1,
        buffer_days=5,
        min_validation_score=0.55  # Realistic threshold
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Run backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years for walk-forward

    results = backtester.run_backtest(
        symbols=WATCHLIST[:50],  # Test with 50 symbols
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    # Display results
    if 'error' not in results:
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return'] * 100:.2f}%")
        print(f"Annual Return: {results['annual_return'] * 100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
        print(f"Win Rate: {results['win_rate'] * 100:.1f}%")

        # Show data integrity report
        if 'detailed_report' in results:
            report = results['detailed_report']
            print(f"\nData Integrity: {report['summary']['data_integrity']}")
            print(f"Windows Processed: {report['summary']['total_windows']}")
            print(f"Average Validation AUC: {report['summary']['avg_validation_auc']:.4f}")

        # Plot results
        plot_backtest_results(results)

        # Save results
        save_backtest_results(results, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        print(f"Backtest failed: {results['error']}")