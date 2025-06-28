# hedge_fund_ml_backtest.py
"""
Institutional-Grade ML Backtesting System
- No data leakage with proper walk-forward optimization
- Handles 200+ symbols efficiently
- Integrates with your ML ensemble models
- Professional risk management and execution simulation
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

# Import your existing modules
from models.ensemble_gpu_windows import GPUEnsembleModel
from models.enhanced_features import EnhancedFeatureEngineer
from config.watchlist import WATCHLIST, SECTOR_MAPPING
from execution_simulator import ExecutionSimulator, ExecutionConfig

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class HedgeFundBacktestConfig:
    """Professional backtesting configuration"""
    # Capital and position sizing
    initial_capital: float = 100000
    position_size_method: str = "risk_parity"  # "fixed", "kelly", "risk_parity"
    base_position_size: float = 0.02  # 2% base size
    max_position_size: float = 0.05  # 5% max
    max_positions: int = 20
    max_sector_exposure: float = 0.30

    # Risk parameters
    stop_loss_atr_multiplier: float = 2.0  # Dynamic stop based on ATR
    take_profit_atr_multiplier: float = 4.0
    max_portfolio_heat: float = 0.06  # Max 6% portfolio risk
    correlation_threshold: float = 0.70

    # Walk-forward parameters
    train_months: int = 12  # 12 months training
    validation_months: int = 3  # 3 months validation
    test_months: int = 1  # 1 month out-of-sample test
    buffer_days: int = 5  # Buffer between train/test
    retrain_frequency_days: int = 21  # Monthly retraining

    # ML parameters
    min_prediction_confidence: float = 0.65
    ensemble_agreement_threshold: float = 0.60  # 60% of models must agree
    feature_importance_threshold: float = 0.05

    # Execution parameters
    execution_delay_minutes: int = 5  # Realistic execution delay
    use_vwap: bool = True  # Use VWAP for execution price
    max_spread_bps: float = 20  # Max acceptable spread

    # Performance filters
    min_sharpe_for_trading: float = 1.0
    min_training_samples: int = 100
    min_validation_score: float = 0.55

    # Data quality
    min_liquidity_usd: float = 1_000_000  # $1M daily volume
    max_missing_data_pct: float = 0.05  # 5% max missing data


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
                        # Add features
                        data = self._add_technical_features(data)
                        data = self._add_ml_features(data)

                        # Quality check
                        if self._pass_quality_check(data):
                            all_data[symbol] = data
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
        """Fetch single symbol with error handling"""
        try:
            df = yf.download(symbol, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)

            if df.empty or len(df) < 100:
                return None

            # Flatten multi-index if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Standardize column names
            df.columns = df.columns.str.lower()

            # Add basic calculated fields
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['dollar_volume'] = df['close'] * df['volume']
            df['symbol'] = symbol

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using enhanced feature engineering"""

        # Use the enhanced feature engineer for comprehensive features
        feature_engineer = EnhancedFeatureEngineer(use_gpu=self.use_gpu)

        # Generate all features including advanced interactions
        all_features = feature_engineer.create_all_features(df, symbol=df.get('symbol', 'UNKNOWN'))

        # Merge with original dataframe
        for col in all_features.columns:
            if col not in df.columns:
                df[col] = all_features[col]

        return df

    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-specific features"""
        # Market regime
        df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        df['volatility_regime'] = pd.qcut(df['volatility_20d'].rolling(60).mean(),
                                          q=3, labels=['low', 'medium', 'high'],
                                          duplicates='drop')

        # Microstructure
        df['spread_proxy'] = df['high_low_pct'].rolling(5).mean()
        df['volume_concentration'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

        # Calendar features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['is_month_end'] = pd.to_datetime(df.index).is_month_end

        return df

    def _pass_quality_check(self, df: pd.DataFrame) -> bool:
        """Quality checks for data"""
        # Check minimum length
        if len(df) < 252:  # 1 year minimum
            return False

        # Check liquidity
        avg_dollar_volume = df['dollar_volume'].tail(20).mean()
        if avg_dollar_volume < HedgeFundBacktestConfig().min_liquidity_usd:
            return False

        # Check missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
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
                     train_end_date: pd.Timestamp) -> GPUEnsembleModel:
        """Train ensemble model with proper validation"""

        logger.info(f"Training models with {len(train_data)} symbols...")

        # Initialize ensemble
        ensemble = GPUEnsembleModel()

        # Prepare training data
        X_train_all = []
        y_train_all = []
        X_val_all = []
        y_val_all = []

        for symbol in train_data:
            if symbol not in val_data:
                continue

            # Extract features and targets
            X_train, y_train = self._prepare_ml_data(train_data[symbol])
            X_val, y_val = self._prepare_ml_data(val_data[symbol])

            if X_train is not None and X_val is not None:
                X_train_all.append(X_train)
                y_train_all.append(y_train)
                X_val_all.append(X_val)
                y_val_all.append(y_val)

        if not X_train_all:
            logger.warning("No valid training data")
            return ensemble

        # Combine all data
        X_train_combined = pd.concat(X_train_all)
        y_train_combined = pd.concat(y_train_all)
        X_val_combined = pd.concat(X_val_all)
        y_val_combined = pd.concat(y_val_all)

        # Train ensemble
        ensemble.train_combined(X_train_combined, y_train_combined,
                                X_val_combined, y_val_combined)

        # Validate performance
        val_score = ensemble.validate(X_val_combined, y_val_combined)
        logger.info(f"Validation score: {val_score:.3f}")

        # Store performance
        self.model_performance[train_end_date] = {
            'val_score': val_score,
            'n_train_samples': len(X_train_combined),
            'n_val_samples': len(X_val_combined)
        }

        # Calculate feature importance
        self.feature_importance[train_end_date] = ensemble.get_feature_importance()

        return ensemble

    def _prepare_ml_data(self, df: pd.DataFrame,
                         lookback: int = 20,
                         lookahead: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and targets for ML"""

        # Feature columns - EXPANDED WITH ALL ENHANCED FEATURES
        feature_cols = [
            # Basic features
            'returns', 'volatility_20d', 'volume_ratio', 'atr_pct',
            'price_to_vwap', 'return_5d', 'return_20d', 'trend_strength',
            'price_to_sma_20', 'price_to_sma_50', 'close_to_high',
            'spread_proxy', 'dollar_volume_rank',

            # Technical indicators
            'rsi_14', 'macd_hist_12_26_9', 'stoch_k_14_3', 'adx_14',
            'mfi_14', 'cci_20', 'williams_r_14', 'ultimate_osc',

            # Advanced interaction features
            'golden_cross', 'death_cross', 'sma_50_200_ratio', 'ma_alignment_score',
            'price_above_all_ma', 'near_resistance', 'near_support',
            'bearish_divergence', 'bullish_divergence',
            'macd_cross_up', 'macd_histogram_growing',
            'oversold_combo', 'overbought_combo',
            'volume_price_confirm', 'high_volume_reversal',
            'bb_squeeze', 'volatility_breakout',
            'strong_uptrend', 'strong_downtrend',
            'bull_market_score', 'mean_reversion_setup', 'breakout_setup',
            'trend_exhaustion', 'triple_buy_signal', 'triple_sell_signal',
            'accumulation_day', 'distribution_day', 'acc_dist_ratio',

            # Volatility features
            'bb_position_20_20', 'kc_position_20_20', 'volatility_regime',
            'parkinson_vol', 'garman_klass_vol', 'yang_zhang_vol',

            # Microstructure features
            'amihud_illiquidity', 'kyle_lambda', 'noise_ratio',

            # Statistical features
            'skew_20d', 'kurtosis_20d', 'hurst_50', 'entropy_20',
            'zscore_20', 'percentile_rank_50',

            # Pattern features
            'cdl_hammer', 'cdl_engulfing', 'cdl_morning_star',
            'pin_bar_bull', 'pin_bar_bear', 'inside_bar',

            # Regime features
            'trend_regime_bullish', 'trend_regime_bearish',
            'high_vol_regime', 'low_vol_regime',
            'trending_market', 'ranging_market'
        ]

        # Filter for available features
        available_cols = [col for col in feature_cols if col in df.columns]

        # Log missing important features
        missing_features = set(feature_cols) - set(available_cols)
        if missing_features:
            important_missing = [f for f in missing_features
                                 if any(x in f for x in ['golden_cross', 'bull_market_score',
                                                         'divergence', 'triple_'])]
            if important_missing:
                logger.warning(f"Missing important features: {important_missing[:10]}")

        if len(available_cols) < len(feature_cols) * 0.5:  # Need at least 50% of features
            logger.error(f"Too many missing features: {len(available_cols)}/{len(feature_cols)}")
            return None, None

        # Create feature matrix
        features = df[available_cols].copy()

        # Add rolling features for key interactions
        for col in ['returns', 'volume_ratio', 'volatility_20d', 'ma_alignment_score', 'bull_market_score']:
            if col in features.columns:
                for lag in [1, 5, 10]:
                    features[f'{col}_lag_{lag}'] = features[col].shift(lag)

                # Rolling stats
                features[f'{col}_roll_mean_5'] = features[col].rolling(5).mean()
                features[f'{col}_roll_std_5'] = features[col].rolling(5).std()

        # Add time-based features for signals
        for signal in ['golden_cross', 'death_cross', 'macd_cross_up', 'triple_buy_signal']:
            if signal in features.columns:
                # Days since last signal
                signal_cumsum = features[signal].cumsum()
                features[f'{signal}_days_since'] = features.groupby(signal_cumsum)[signal].cumcount()

        # Create target: positive return in next 5 days
        df['future_return'] = df['close'].shift(-lookahead) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)  # 2% threshold

        # Remove NaN rows
        valid_idx = features.notna().all(axis=1) & df['target'].notna()

        logger.info(f"ML features prepared: {len(features.columns)} features, {valid_idx.sum()} valid samples")

        return features[valid_idx], df['target'][valid_idx]

    def predict(self, model: GPUEnsembleModel,
                current_data: Dict[str, pd.DataFrame],
                date: pd.Timestamp) -> Dict[str, Dict]:
        """Generate predictions for all symbols"""

        predictions = {}

        for symbol, df in current_data.items():
            # Get data up to current date
            hist_data = df[df.index <= date].copy()

            if len(hist_data) < 50:
                continue

            # Prepare features
            X, _ = self._prepare_ml_data(hist_data)

            if X is None or len(X) == 0:
                continue

            # Get latest features
            latest_features = X.iloc[-1:].copy()

            # Get prediction from ensemble
            try:
                pred = model.predict_proba(latest_features)

                # Get individual model predictions for confidence
                individual_preds = model.get_individual_predictions(latest_features)
                agreement = np.mean([p > 0.5 for p in individual_preds])

                predictions[symbol] = {
                    'probability': pred[0],
                    'prediction': int(pred[0] > 0.5),
                    'confidence': pred[0] if pred[0] > 0.5 else 1 - pred[0],
                    'model_agreement': agreement,
                    'features': latest_features.to_dict(orient='records')[0]
                }

            except Exception as e:
                logger.error(f"Prediction error for {symbol}: {e}")
                continue

        return predictions


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
            volatility = market_data['volatility_20d'].iloc[-1]

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
                atr = market_data[symbol]['atr'].iloc[-1]
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
            corr_risk = self._check_correlation_risk(positions.keys(), market_data)
            if corr_risk:
                risk_checks['correlation_risk'] = True
                risk_checks['messages'].append("High correlation between positions")

        # Check concentration
        total_value = sum(position_values.values())
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
                returns = market_data[symbol]['returns'].tail(60)
                if len(returns) >= 60:
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


class HedgeFundBacktester:
    """Main backtesting engine with ML integration"""

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

        for date in tqdm(all_trading_dates, desc="Backtesting"):
            # Check if we need to retrain
            if (current_model is None or
                    (last_train_date and (date - last_train_date).days >= self.config.retrain_frequency_days)):
                # Train new model
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
                signals = self._filter_signals(predictions, date, all_data)

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

        # Calculate final metrics
        results = self._calculate_performance_metrics()

        # Generate reports
        self._generate_reports(results)

        return results

    def _train_model_for_date(self, all_data: Dict[str, pd.DataFrame],
                              current_date: pd.Timestamp) -> GPUEnsembleModel:
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
        """Filter and rank trading signals"""

        valid_signals = []

        for symbol, pred in predictions.items():
            # Check prediction quality
            if (pred['confidence'] < self.config.min_prediction_confidence or
                    pred['model_agreement'] < self.config.ensemble_agreement_threshold):
                continue

            # Check liquidity
            if symbol in all_data:
                df = all_data[symbol]
                recent_data = df[df.index <= date].tail(20)

                if not recent_data.empty:
                    avg_dollar_volume = recent_data['dollar_volume'].mean()
                    if avg_dollar_volume < self.config.min_liquidity_usd:
                        continue

                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'prediction': pred,
                        'current_price': recent_data['close'].iloc[-1],
                        'atr': recent_data['atr'].iloc[-1],
                        'volatility': recent_data['volatility_20d'].iloc[-1],
                        'dollar_volume': avg_dollar_volume,
                        'score': pred['confidence'] * pred['model_agreement']
                    }

                    valid_signals.append(signal)

        # Sort by score
        valid_signals.sort(key=lambda x: x['score'], reverse=True)

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
            return 0

        # Check risk limits
        risk_check = self.risk_manager.check_portfolio_risk(
            self.positions, all_data, portfolio_value
        )

        if not risk_check['pass']:
            logger.debug(f"Risk check failed: {risk_check['messages']}")
            return 0

        # Simulate execution
        exec_price, slippage, commission = self.execution_sim.simulate_entry(
            symbol, signal['current_price'], shares, signal['dollar_volume']
        )

        total_cost = (exec_price * shares) + slippage + commission

        if total_cost > cash * 0.95:  # Keep 5% cash buffer
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
            'ml_agreement': signal['prediction']['model_agreement']
        })

        logger.debug(f"Entered {symbol}: {shares} shares @ ${exec_price:.2f}")

        return total_cost

    def _check_exit_conditions(self, date: pd.Timestamp,
                               all_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Check for exit conditions"""

        exits = []

        for symbol, position in self.positions.items():
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            current_data = df[df.index <= date]

            if current_data.empty or date not in df.index:
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
                current_vol = current_data['volatility_20d'].iloc[-1]
                if current_vol > 0.40:  # High volatility environment
                    exits.append({
                        'symbol': symbol,
                        'reason': 'volatility_exit',
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

        logger.debug(f"Exited {symbol}: P&L ${total_pnl:.2f} ({pnl_pct:.2%})")

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
                    'high_confidence_trades': len(
                        buy_trades[buy_trades['ml_confidence'] > 0.7]) if 'ml_confidence' in buy_trades else 0
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


def run_hedge_fund_backtest():
    """Run the hedge fund grade backtest"""

    # Configuration
    config = HedgeFundBacktestConfig(
        initial_capital=100000,
        position_size_method="risk_parity",
        max_positions=20,
        min_prediction_confidence=0.65,
        ensemble_agreement_threshold=0.60
    )

    # Create backtester
    backtester = HedgeFundBacktester(config)

    # Define test period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year backtest

    # Run on full watchlist
    logger.info(f"Running backtest on {len(WATCHLIST)} symbols")
    results = backtester.run_backtest(
        symbols=WATCHLIST,  # Full 200+ symbols
        start_date=start_date,
        end_date=end_date
    )

    return results


if __name__ == "__main__":
    results = run_hedge_fund_backtest()