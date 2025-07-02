# models/enhanced_features.py
"""
Enhanced Feature Engineering with Adaptive Capabilities
Complete version with 30 methods for comprehensive feature generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import talib
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
from numba import cuda, jit, prange
import warnings

# Try to import GPU libraries
try:
    import cupy as cp
    import cudf

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cudf = None

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """GPU-accelerated feature engineering with advanced techniques and fallbacks"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.feature_cache = {}
        self.scaler = RobustScaler()
        self.feature_names = []

        # Feature selection tracking
        self.feature_correlations = {}
        self.feature_importance_history = []

        if self.use_gpu:
            logger.info("GPU-accelerated feature engineering enabled")
        else:
            logger.info("Using CPU for feature engineering")

    def _calculate_warm_up_period(self) -> int:
        """Calculate required warm-up period for all features"""

        # Maximum lookback periods used in features
        warm_up_periods = {
            'sma_200': 200,
            'correlation_100': 100,
            'volatility_60': 60,
            'support_resistance_100': 100,
            'fibonacci_100': 100,
            'hurst_100': 100,
            'statistical_252': 252  # For rolling yearly stats
        }

        # Get maximum required warm-up
        max_warm_up = max(warm_up_periods.values())

        # Add 20% buffer for safety
        return int(max_warm_up * 1.2)

    @staticmethod
    @cuda.jit
    def _gpu_rolling_stats(arr, window, out_mean, out_std, out_skew, out_kurt):
        """GPU kernel for rolling statistics"""
        idx = cuda.grid(1)
        n = arr.shape[0]

        if idx < n - window + 1:
            # Calculate mean
            sum_val = 0.0
            for i in range(window):
                sum_val += arr[idx + i]
            mean = sum_val / window
            out_mean[idx] = mean

            # Calculate variance and higher moments
            sum_sq = 0.0
            sum_cube = 0.0
            sum_quad = 0.0

            for i in range(window):
                diff = arr[idx + i] - mean
                diff_sq = diff * diff
                sum_sq += diff_sq
                sum_cube += diff_sq * diff
                sum_quad += diff_sq * diff_sq

            variance = sum_sq / window
            std = cuda.sqrt(variance)
            out_std[idx] = std

            # Skewness and kurtosis
            if std > 0:
                out_skew[idx] = (sum_cube / window) / (std ** 3)
                out_kurt[idx] = (sum_quad / window) / (variance ** 2) - 3
            else:
                out_skew[idx] = 0
                out_kurt[idx] = 0

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_rolling_stats(arr, window):
        """CPU-optimized rolling statistics using Numba"""
        n = len(arr)
        means = np.empty(n - window + 1)
        stds = np.empty(n - window + 1)

        for i in prange(n - window + 1):
            window_data = arr[i:i + window]
            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)

        return means, stds

    def create_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create comprehensive feature set with GPU acceleration - MORE ROBUST"""

        # REDUCED minimum from 200 to 100
        if len(df) < 100:  # Was 200
            logger.warning(f"Limited data for {symbol}: {len(df)} rows, using reduced features")
            # Don't return empty DataFrame, create minimal features instead
            return self._create_minimal_feature_set(df, symbol)

        try:
            if self.use_gpu and CUPY_AVAILABLE:
                try:
                    # Convert to GPU dataframe
                    df_gpu = cudf.from_pandas(df)
                    features = self._create_features_gpu(df_gpu, symbol)
                    # Convert back to pandas
                    return features.to_pandas()
                except Exception as e:
                    logger.warning(f"GPU feature creation failed, falling back to CPU: {e}")
                    return self._create_features_cpu(df, symbol)
            else:
                return self._create_features_cpu(df, symbol)
        except Exception as e:
            logger.error(f"Feature creation failed for {symbol}: {e}")
            # Return minimal features as fallback
            return self._create_minimal_feature_set(df, symbol)

    def _create_minimal_feature_set(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create minimal but useful feature set for symbols with limited data"""

        features = pd.DataFrame(index=df.index)

        # Ensure float64
        df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype('float64')

        # Basic returns
        features['return_1d'] = df['close'].pct_change()
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)

        # Simple MAs (adaptive to data length)
        for period in [5, 10, 20]:
            if len(df) >= period * 2:  # Need 2x period for stability
                features[f'sma_{period}'] = df['close'].rolling(period, min_periods=period // 2).mean()
                features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']

        # Volume ratio
        features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20, min_periods=5).mean()

        # Price ranges
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_to_high'] = df['close'] / df['high']
        features['close_to_low'] = df['close'] / df['low']

        # Simple volatility
        features['volatility_20d'] = features['return_1d'].rolling(20, min_periods=5).std() * np.sqrt(252)

        # Basic RSI
        features['rsi_14'] = self._calculate_simple_rsi(df['close'], 14)

        # Handle NaN
        features = features.fillna(method='ffill', limit=5).fillna(0)

        self.feature_names = features.columns.tolist()

        return features

    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI without TA-Lib"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period // 2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period // 2).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _add_features_to_dataframe(self, features_df: pd.DataFrame, new_features: Dict,
                                   index: pd.Index) -> None:
        """Helper method to properly add features to DataFrame"""
        for key, value in new_features.items():
            if isinstance(value, pd.Series):
                features_df[key] = value
            elif isinstance(value, pd.DataFrame):
                for col in value.columns:
                    features_df[col] = value[col]
            elif isinstance(value, np.ndarray):
                if len(value) == len(index):
                    features_df[key] = value
                else:
                    # Handle mismatched lengths by creating a Series with NaN
                    series = pd.Series(index=index, dtype=float)
                    series.iloc[:len(value)] = value
                    features_df[key] = series
            else:
                # Scalar value
                features_df[key] = value

    def _create_features_cpu(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """CPU-based feature creation with error handling"""
        try:
            features = pd.DataFrame(index=df.index)

            # FIX: Ensure all price/volume data is float64 for TA-Lib
            df = df.copy()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype('float64')

            # 1. Price-based features
            self._add_features_to_dataframe(features, self._create_price_features(df), df.index)

            # 2. Volume features
            self._add_features_to_dataframe(features, self._create_volume_features(df), df.index)

            # 3. Volatility features
            self._add_features_to_dataframe(features, self._create_volatility_features(df), df.index)

            # 4. Technical indicators
            self._add_features_to_dataframe(features, self._create_technical_indicators(df), df.index)

            # 5. Market microstructure
            self._add_features_to_dataframe(features, self._create_microstructure_features(df), df.index)

            # 6. Statistical features
            self._add_features_to_dataframe(features, self._create_statistical_features(df), df.index)

            # 7. Pattern recognition features
            self._add_features_to_dataframe(features, self._create_pattern_features(df), df.index)

            # 8. Interaction features
            self._add_features_to_dataframe(features, self._create_interaction_features(df, features), df.index)

            # 9. Market regime features
            self._add_features_to_dataframe(features, self._create_regime_features(df), df.index)

            # 10. Advanced ML features
            self._add_features_to_dataframe(features, self._create_ml_features(df, features), df.index)

            # 11. ADVANCED INTERACTION FEATURES
            self._add_features_to_dataframe(features, self._create_advanced_interaction_features(df, features),
                                            df.index)

            # 12. ADDITIONAL: Seasonal features
            self._add_features_to_dataframe(features, self._create_seasonal_features(df), df.index)

            # 13. ADDITIONAL: Momentum features
            self._add_features_to_dataframe(features, self._create_momentum_features(df, features), df.index)

            # Store feature names
            self.feature_names = features.columns.tolist()

            # Handle missing values
            features = self._handle_missing_values(features)

            # Apply feature selection
            features = self._apply_feature_selection(features, symbol)

            return features

        except Exception as e:
            logger.error(f"Full feature creation failed for {symbol}: {e}")
            # Fallback to minimal features
            return self._create_minimal_feature_set(df, symbol)

    def _create_features_gpu(self, df, symbol: str):
        """GPU-accelerated feature creation with fallback"""
        try:
            if CUPY_AVAILABLE:
                features = cudf.DataFrame(index=df.index)
            else:
                # Fallback to pandas if GPU not available
                return self._create_features_cpu(df, symbol)

            # Convert to CuPy arrays for faster computation
            if CUPY_AVAILABLE and cp is not None:
                close = cp.asarray(df['close'].values)
                high = cp.asarray(df['high'].values)
                low = cp.asarray(df['low'].values)
                open_price = cp.asarray(df['open'].values)
                volume = cp.asarray(df['volume'].values)

                # Price features using GPU
                features = self._create_price_features_gpu(features, close, high, low, open_price)

                # Volume features using GPU
                features = self._create_volume_features_gpu(features, close, volume)

                # More GPU-accelerated features...
                # (Implementation continues similar to CPU version but using CuPy)

                return features
            else:
                # Fallback to CPU
                return self._create_features_cpu(df, symbol)
        except Exception as e:
            logger.error(f"GPU feature creation failed: {e}")
            return self._create_features_cpu(df, symbol)

    def _create_price_features(self, df: pd.DataFrame) -> Dict:
        """Create price-based features with error handling"""
        features = {}  # MUST be before try block

        try:
            # Ensure float64 for TA-Lib
            close = df['close'].astype('float64').values
            high = df['high'].astype('float64').values
            low = df['low'].astype('float64').values
            open_price = df['open'].astype('float64').values

            # Returns at multiple timeframes
            for period in [1, 2, 3, 5, 10, 20, 60]:
                if len(df) >= period + 1:
                    features[f'return_{period}d'] = df['close'].pct_change(period)
                    features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))

            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) >= period * 2:  # Need 2x period for stability
                    try:
                        sma = talib.SMA(close, timeperiod=period)
                        features[f'sma_{period}'] = pd.Series(sma, index=df.index)
                        features[f'price_to_sma_{period}'] = pd.Series(close / sma, index=df.index)

                        # MA slopes
                        sma_slope = talib.SMA(sma.astype('float64'), 5)
                        features[f'sma_{period}_slope'] = pd.Series((sma - sma_slope) / 5, index=df.index)
                    except Exception as e:
                        logger.debug(f"Error calculating SMA {period}: {e}")
                        # Fallback to pandas
                        sma_series = df['close'].rolling(period, min_periods=period // 2).mean()
                        features[f'sma_{period}'] = sma_series
                        features[f'price_to_sma_{period}'] = df['close'] / sma_series
                        features[f'sma_{period}_slope'] = sma_series.diff(5) / 5

            # Exponential moving averages
            for period in [8, 12, 21, 26, 50]:
                if len(df) >= period * 2:
                    try:
                        ema = talib.EMA(close, timeperiod=period)
                        features[f'ema_{period}'] = pd.Series(ema, index=df.index)
                        features[f'price_to_ema_{period}'] = pd.Series(close / ema, index=df.index)
                    except Exception as e:
                        logger.debug(f"Error calculating EMA {period}: {e}")
                        ema_series = df['close'].ewm(span=period, min_periods=period // 2).mean()
                        features[f'ema_{period}'] = ema_series
                        features[f'price_to_ema_{period}'] = df['close'] / ema_series

            # VWAP approximation
            typical_price = (high + low + close) / 3
            vwap_numerator = pd.Series(typical_price * df['volume'].astype('float64').values, index=df.index).rolling(
                20, min_periods=5).sum()
            vwap_denominator = df['volume'].astype('float64').rolling(20, min_periods=5).sum()
            features['vwap'] = vwap_numerator / vwap_denominator
            features['price_to_vwap'] = df['close'] / features['vwap']

            # Price positions and ranges
            features['close_to_high'] = pd.Series(close / high, index=df.index)
            features['close_to_low'] = pd.Series(close / low, index=df.index)
            features['high_low_range'] = pd.Series((high - low) / close, index=df.index)
            features['close_to_open'] = pd.Series(close / open_price, index=df.index)
            features['body_size'] = pd.Series(abs(close - open_price) / close, index=df.index)

            # Gaps
            prev_close = df['close'].shift(1).values
            features['gap'] = pd.Series(open_price / prev_close, index=df.index)
            features['gap_size'] = abs(features['gap'] - 1)
            features['gap_up'] = (features['gap'] > 1.01).astype(int)
            features['gap_down'] = (features['gap'] < 0.99).astype(int)

            # Support/Resistance levels (adaptive)
            for period in [10, 20, 50, 100]:
                if len(df) >= period * 2:
                    resistance = df['high'].rolling(period, min_periods=period // 2).max()
                    support = df['low'].rolling(period, min_periods=period // 2).min()

                    features[f'resistance_{period}d'] = resistance
                    features[f'support_{period}d'] = support
                    features[f'dist_from_resistance_{period}d'] = (resistance - df['close']) / df['close']
                    features[f'dist_from_support_{period}d'] = (df['close'] - support) / df['close']
                    features[f'sr_range_{period}d'] = (resistance - support) / df['close']

            # Price channels
            for period in [20, 50]:
                if len(df) >= period * 2:
                    highest = df['high'].rolling(period, min_periods=period // 2).max()
                    lowest = df['low'].rolling(period, min_periods=period // 2).min()
                    features[f'price_channel_pos_{period}'] = (df['close'] - lowest) / (highest - lowest + 1e-10)

            # Fibonacci retracements (only if enough data)
            if len(df) >= 100:
                high_100 = df['high'].rolling(100, min_periods=50).max()
                low_100 = df['low'].rolling(100, min_periods=50).min()
                fib_range = high_100 - low_100

                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                for level in fib_levels:
                    fib_price = low_100 + fib_range * level
                    features[f'dist_from_fib_{int(level * 1000)}'] = (df['close'] - fib_price) / df['close']

        except Exception as e:
            logger.error(f"Error in price features: {e}")
            # Return what we have so far

        return features  # MUST be after except block

    def _create_volume_features(self, df: pd.DataFrame) -> Dict:
        """Create volume-based features with error handling"""
        features = {}

        try:
            # FIX: Ensure float64 for TA-Lib
            volume = df['volume'].astype('float64').values
            close = df['close'].astype('float64').values

            # Volume moving averages and ratios
            for period in [5, 10, 20, 50]:
                if len(df) >= period * 2:
                    try:
                        vol_ma = talib.SMA(volume, timeperiod=period)
                        features[f'volume_ma_{period}'] = vol_ma
                        features[f'volume_ratio_{period}'] = volume / (vol_ma + 1e-10)

                        # Volume trend
                        vol_ma_double = talib.SMA(volume, timeperiod=period * 2)
                        features[f'volume_trend_{period}'] = vol_ma / (vol_ma_double + 1e-10)
                    except Exception as e:
                        logger.debug(f"Error calculating volume MA {period}: {e}")
                        # Fallback
                        features[f'volume_ma_{period}'] = df['volume'].rolling(period, min_periods=period // 2).mean()
                        features[f'volume_ratio_{period}'] = df['volume'] / (features[f'volume_ma_{period}'] + 1e-10)

            # Volume rate of change
            if len(df) >= 10:
                try:
                    features['volume_roc_5'] = talib.ROC(volume, timeperiod=5)
                    features['volume_roc_10'] = talib.ROC(volume, timeperiod=10)
                except Exception as e:
                    logger.debug(f"Error calculating volume ROC: {e}")
                    features['volume_roc_5'] = df['volume'].pct_change(5)
                    features['volume_roc_10'] = df['volume'].pct_change(10)

            # On Balance Volume
            if len(df) >= 20:
                try:
                    obv = talib.OBV(close, volume)
                    features['obv'] = obv
                    obv_series = pd.Series(obv)
                    features['obv_ma'] = talib.SMA(obv.astype('float64'), timeperiod=20)
                    features['obv_signal'] = (obv_series > features['obv_ma']).astype(int)
                    features['obv_divergence'] = self._calculate_divergence(close, obv_series)
                except Exception as e:
                    logger.debug(f"Error calculating OBV: {e}")
                    # Simple OBV fallback
                    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
                    features['obv'] = obv
                    features['obv_ma'] = obv.rolling(20, min_periods=10).mean()
                    features['obv_signal'] = (obv > features['obv_ma']).astype(int)

            # Accumulation/Distribution
            if len(df) >= 20:
                try:
                    ad = talib.AD(df['high'].astype('float64').values,
                                  df['low'].astype('float64').values,
                                  close, volume)
                    features['ad'] = ad
                    features['ad_ma'] = talib.SMA(ad.astype('float64'), timeperiod=20)
                    features['ad_signal'] = (features['ad'] > features['ad_ma']).astype(int)
                except Exception as e:
                    logger.debug(f"Error calculating AD: {e}")
                    # Simple A/D fallback
                    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
                    ad = (clv * df['volume']).cumsum()
                    features['ad'] = ad
                    features['ad_ma'] = ad.rolling(20, min_periods=10).mean()

            # Chaikin Money Flow
            if len(df) >= 20:
                try:
                    features['cmf'] = talib.ADOSC(df['high'].astype('float64').values,
                                                  df['low'].astype('float64').values,
                                                  close, volume,
                                                  fastperiod=3, slowperiod=10)
                except Exception as e:
                    logger.debug(f"Error calculating CMF: {e}")
                    # Simple CMF
                    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * \
                          df['volume']
                    features['cmf'] = mfv.rolling(20, min_periods=10).sum() / df['volume'].rolling(20,
                                                                                                   min_periods=10).sum()

            # Volume Price Trend
            features['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
            features['vpt_ma'] = features['vpt'].rolling(20, min_periods=10).mean()

            # Money Flow
            typical_price = (df['high'].astype('float64') + df['low'].astype('float64') + df['close'].astype(
                'float64')) / 3
            money_flow = typical_price * df['volume'].astype('float64')
            features['money_flow_ratio'] = money_flow / money_flow.rolling(20, min_periods=10).mean()

            # Volume profile
            features['volume_concentration'] = df['volume'].rolling(20, min_periods=10).std() / df['volume'].rolling(20,
                                                                                                                     min_periods=10).mean()

            # Price-Volume correlation
            features['pv_correlation'] = df['close'].rolling(20, min_periods=10).corr(df['volume'])

            # Volume spike detection
            vol_mean = df['volume'].rolling(20, min_periods=10).mean()
            vol_std = df['volume'].rolling(20, min_periods=10).std()
            features['volume_spike'] = ((df['volume'] - vol_mean) / (vol_std + 1e-10) > 2).astype(int)

            # Volume-weighted price momentum
            features['vw_momentum'] = (df['close'] * df['volume']).rolling(10, min_periods=5).sum() / df[
                'volume'].rolling(10, min_periods=5).sum()
            features['vw_momentum_change'] = features['vw_momentum'].pct_change(5)

        except Exception as e:
            logger.error(f"Error in volume features: {e}")

        return features

    def _create_volatility_features(self, df: pd.DataFrame) -> Dict:
        """Create volatility features with error handling"""
        features = {}

        try:
            # Ensure float64 for TA-Lib
            high = df['high'].astype('float64').values
            low = df['low'].astype('float64').values
            close = df['close'].astype('float64').values

            # ATR at multiple timeframes
            for period in [7, 14, 20, 30]:
                if len(df) >= period * 2:
                    try:
                        atr = talib.ATR(high, low, close, timeperiod=period)
                        features[f'atr_{period}'] = atr
                        features[f'atr_pct_{period}'] = atr / close

                        # ATR bands
                        features[f'atr_upper_{period}'] = close + 2 * atr
                        features[f'atr_lower_{period}'] = close - 2 * atr
                    except Exception as e:
                        logger.debug(f"Error calculating ATR {period}: {e}")
                        # Simple ATR fallback
                        hl = df['high'] - df['low']
                        hc = abs(df['high'] - df['close'].shift(1))
                        lc = abs(df['low'] - df['close'].shift(1))
                        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                        features[f'atr_{period}'] = tr.rolling(period, min_periods=period // 2).mean()
                        features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close']

            # Bollinger Bands with multiple parameters
            for period in [10, 20, 30]:
                if len(df) >= period * 2:
                    for std_dev in [1.5, 2, 2.5]:
                        try:
                            upper, middle, lower = talib.BBANDS(close, timeperiod=period,
                                                                nbdevup=std_dev, nbdevdn=std_dev)
                            suffix = f'{period}_{int(std_dev * 10)}'

                            features[f'bb_upper_{suffix}'] = upper
                            features[f'bb_lower_{suffix}'] = lower
                            features[f'bb_middle_{suffix}'] = middle
                            features[f'bb_width_{suffix}'] = (upper - lower) / (middle + 1e-10)
                            features[f'bb_position_{suffix}'] = (close - lower) / (upper - lower + 1e-10)

                            # Bollinger Band squeeze
                            bb_width_series = pd.Series((upper - lower) / (middle + 1e-10))
                            features[f'bb_squeeze_{suffix}'] = bb_width_series / bb_width_series.rolling(50,
                                                                                                         min_periods=20).mean()
                        except Exception as e:
                            logger.debug(f"Error calculating BB {period}_{std_dev}: {e}")
                            # Simple BB fallback
                            sma = df['close'].rolling(period, min_periods=period // 2).mean()
                            std = df['close'].rolling(period, min_periods=period // 2).std()
                            suffix = f'{period}_{int(std_dev * 10)}'
                            features[f'bb_upper_{suffix}'] = sma + std_dev * std
                            features[f'bb_lower_{suffix}'] = sma - std_dev * std
                            features[f'bb_middle_{suffix}'] = sma

            # Keltner Channels
            for period in [10, 20]:
                if len(df) >= period * 2:
                    for mult in [1.5, 2, 2.5]:
                        try:
                            ma = talib.EMA(close, timeperiod=period)
                            atr = talib.ATR(high, low, close, timeperiod=period)

                            features[f'kc_upper_{period}_{int(mult * 10)}'] = ma + (mult * atr)
                            features[f'kc_lower_{period}_{int(mult * 10)}'] = ma - (mult * atr)
                            features[f'kc_position_{period}_{int(mult * 10)}'] = (close - features[
                                f'kc_lower_{period}_{int(mult * 10)}']) / \
                                                                                 (features[
                                                                                      f'kc_upper_{period}_{int(mult * 10)}'] -
                                                                                  features[
                                                                                      f'kc_lower_{period}_{int(mult * 10)}'] + 1e-10)
                        except Exception as e:
                            logger.debug(f"Error calculating KC {period}_{mult}: {e}")

            # Historical volatility
            returns = df['close'].pct_change()
            for period in [5, 10, 20, 30, 60]:
                if len(df) >= period * 2:
                    features[f'volatility_{period}d'] = returns.rolling(period,
                                                                        min_periods=period // 2).std() * np.sqrt(252)

                    # Volatility of volatility
                    if f'volatility_{period}d' in features:
                        features[f'vol_of_vol_{period}d'] = features[f'volatility_{period}d'].rolling(period,
                                                                                                      min_periods=period // 2).std()

            # Parkinson volatility
            if len(df) >= 20:
                features['parkinson_vol'] = np.sqrt(
                    252 * (1 / (4 * np.log(2))) *
                    pd.Series(np.log(high / low) ** 2).rolling(20, min_periods=10).mean()
                )

            # Garman-Klass volatility
            if len(df) >= 20:
                open_prices = df['open'].astype('float64').values
                features['garman_klass_vol'] = np.sqrt(
                    252 * (
                            0.5 * pd.Series(np.log(high / low) ** 2).rolling(20, min_periods=10).mean() -
                            (2 * np.log(2) - 1) * pd.Series(np.log(close / open_prices) ** 2).rolling(20,
                                                                                                      min_periods=10).mean()
                    )
                )

            # Rogers-Satchell volatility
            if len(df) >= 20:
                features['rogers_satchell_vol'] = np.sqrt(
                    252 * pd.Series(
                        np.log(high / close) * np.log(high / open_prices) +
                        np.log(low / close) * np.log(low / open_prices)
                    ).rolling(20, min_periods=10).mean()
                )

            # Yang-Zhang volatility
            if len(df) >= 20:
                overnight_var = pd.Series(np.log(df['open'] / df['close'].shift(1)) ** 2).rolling(20,
                                                                                                  min_periods=10).mean()
                open_close_var = pd.Series(np.log(close / open_prices) ** 2).rolling(20, min_periods=10).mean()
                features['yang_zhang_vol'] = np.sqrt(252 * (overnight_var + 0.383 * open_close_var +
                                                            0.641 * features.get('rogers_satchell_vol', 0) ** 2))

            # Volatility regime
            if 'volatility_20d' in features:
                vol_median = features['volatility_20d'].rolling(100, min_periods=50).median()
                features['vol_regime'] = (features['volatility_20d'] > vol_median * 1.5).astype(int)

        except Exception as e:
            logger.error(f"Error in volatility features: {e}")

        return features

    def _create_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Create technical indicators with extensive error handling"""
        features = {}

        try:
            # Ensure float64 for TA-Lib
            high = df['high'].astype('float64').values
            low = df['low'].astype('float64').values
            close = df['close'].astype('float64').values
            volume = df['volume'].astype('float64').values

            # RSI with multiple periods
            for period in [7, 14, 21, 28]:
                if len(df) >= period * 2:
                    try:
                        rsi = talib.RSI(close, timeperiod=period)
                        features[f'rsi_{period}'] = rsi
                        features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
                        features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)

                        # RSI divergence
                        features[f'rsi_{period}_divergence'] = self._calculate_divergence(close, rsi)
                    except Exception as e:
                        logger.debug(f"Error calculating RSI {period}: {e}")
                        # Fallback RSI
                        features[f'rsi_{period}'] = self._calculate_simple_rsi(df['close'], period)
                        features[f'rsi_{period}_oversold'] = (features[f'rsi_{period}'] < 30).astype(int)
                        features[f'rsi_{period}_overbought'] = (features[f'rsi_{period}'] > 70).astype(int)

            # MACD variations
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
                if len(df) >= slow * 2:
                    try:
                        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=fast,
                                                                  slowperiod=slow, signalperiod=signal)
                        suffix = f'{fast}_{slow}_{signal}'

                        features[f'macd_{suffix}'] = macd
                        features[f'macd_signal_{suffix}'] = macd_signal
                        features[f'macd_hist_{suffix}'] = macd_hist
                        features[f'macd_cross_{suffix}'] = ((macd > macd_signal) &
                                                            (pd.Series(macd).shift(1) <= pd.Series(macd_signal).shift(
                                                                1))).astype(int)

                        # MACD momentum
                        features[f'macd_momentum_{suffix}'] = pd.Series(macd_hist) - pd.Series(macd_hist).shift(1)
                    except Exception as e:
                        logger.debug(f"Error calculating MACD {fast}_{slow}_{signal}: {e}")
                        # Simple MACD fallback
                        ema_fast = df['close'].ewm(span=fast, min_periods=fast // 2).mean()
                        ema_slow = df['close'].ewm(span=slow, min_periods=slow // 2).mean()
                        macd = ema_fast - ema_slow
                        macd_signal = macd.ewm(span=signal, min_periods=signal // 2).mean()
                        suffix = f'{fast}_{slow}_{signal}'
                        features[f'macd_{suffix}'] = macd
                        features[f'macd_signal_{suffix}'] = macd_signal
                        features[f'macd_hist_{suffix}'] = macd - macd_signal

            # Stochastic variations
            for k_period, d_period in [(14, 3), (21, 5), (5, 3)]:
                if len(df) >= k_period * 2:
                    try:
                        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowd_period=d_period)
                        suffix = f'{k_period}_{d_period}'

                        features[f'stoch_k_{suffix}'] = k
                        features[f'stoch_d_{suffix}'] = d
                        features[f'stoch_cross_{suffix}'] = (
                                    (k > d) & (pd.Series(k).shift(1) <= pd.Series(d).shift(1))).astype(int)
                        features[f'stoch_oversold_{suffix}'] = ((k < 20) & (d < 20)).astype(int)
                        features[f'stoch_overbought_{suffix}'] = ((k > 80) & (d > 80)).astype(int)
                    except Exception as e:
                        logger.debug(f"Error calculating Stochastic {k_period}_{d_period}: {e}")
                        # Simple stochastic
                        lowest_low = df['low'].rolling(k_period, min_periods=k_period // 2).min()
                        highest_high = df['high'].rolling(k_period, min_periods=k_period // 2).max()
                        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10))
                        d = k.rolling(d_period, min_periods=1).mean()
                        suffix = f'{k_period}_{d_period}'
                        features[f'stoch_k_{suffix}'] = k
                        features[f'stoch_d_{suffix}'] = d

            # Williams %R
            for period in [10, 14, 20]:
                if len(df) >= period * 2:
                    try:
                        features[f'williams_r_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating Williams %R {period}: {e}")
                        # Simple Williams %R
                        highest_high = df['high'].rolling(period, min_periods=period // 2).max()
                        lowest_low = df['low'].rolling(period, min_periods=period // 2).min()
                        features[f'williams_r_{period}'] = -100 * (
                                    (highest_high - df['close']) / (highest_high - lowest_low + 1e-10))

            # CCI
            for period in [14, 20, 30]:
                if len(df) >= period * 2:
                    try:
                        features[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating CCI {period}: {e}")
                        # Simple CCI
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        sma = typical_price.rolling(period, min_periods=period // 2).mean()
                        mad = (typical_price - sma).abs().rolling(period, min_periods=period // 2).mean()
                        features[f'cci_{period}'] = (typical_price - sma) / (0.015 * mad + 1e-10)

            # MFI
            for period in [14, 20]:
                if len(df) >= period * 2:
                    try:
                        features[f'mfi_{period}'] = talib.MFI(high, low, close, volume, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating MFI {period}: {e}")
                        # Simple MFI
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        money_flow = typical_price * df['volume']
                        positive_flow = (money_flow * (typical_price > typical_price.shift(1))).rolling(period,
                                                                                                        min_periods=period // 2).sum()
                        negative_flow = (money_flow * (typical_price <= typical_price.shift(1))).rolling(period,
                                                                                                         min_periods=period // 2).sum()
                        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
                        features[f'mfi_{period}'] = mfi

            # ADX and directional indicators
            for period in [14, 20]:
                if len(df) >= period * 3:
                    try:
                        features[f'adx_{period}'] = talib.ADX(high, low, close, timeperiod=period)
                        features[f'plus_di_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
                        features[f'minus_di_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
                        features[f'di_diff_{period}'] = features[f'plus_di_{period}'] - features[f'minus_di_{period}']

                        # Trend strength
                        features[f'trend_strength_{period}'] = features[f'adx_{period}'] * np.sign(
                            features[f'di_diff_{period}'])
                    except Exception as e:
                        logger.debug(f"Error calculating ADX {period}: {e}")
                        # Fallback values
                        features[f'adx_{period}'] = 25.0  # Neutral ADX
                        features[f'plus_di_{period}'] = 25.0
                        features[f'minus_di_{period}'] = 25.0
                        features[f'di_diff_{period}'] = 0.0
                        features[f'trend_strength_{period}'] = 0.0

            # Aroon
            for period in [14, 25]:
                if len(df) >= period * 2:
                    try:
                        aroon_up, aroon_down = talib.AROON(high, low, timeperiod=period)
                        features[f'aroon_up_{period}'] = aroon_up
                        features[f'aroon_down_{period}'] = aroon_down
                        features[f'aroon_osc_{period}'] = aroon_up - aroon_down
                    except Exception as e:
                        logger.debug(f"Error calculating Aroon {period}: {e}")
                        # Simple Aroon
                        high_idx = df['high'].rolling(period + 1).apply(lambda x: x.argmax())
                        low_idx = df['low'].rolling(period + 1).apply(lambda x: x.argmin())
                        features[f'aroon_up_{period}'] = ((period - high_idx) / period) * 100
                        features[f'aroon_down_{period}'] = ((period - low_idx) / period) * 100
                        features[f'aroon_osc_{period}'] = features[f'aroon_up_{period}'] - features[
                            f'aroon_down_{period}']

            # Ultimate Oscillator
            if len(df) >= 28:
                try:
                    features['ultimate_osc'] = talib.ULTOSC(high, low, close)
                except Exception as e:
                    logger.debug(f"Error calculating Ultimate Oscillator: {e}")
                    features['ultimate_osc'] = 50.0  # Neutral value

            # ROC
            for period in [5, 10, 20]:
                if len(df) >= period + 1:
                    try:
                        features[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating ROC {period}: {e}")
                        features[f'roc_{period}'] = df['close'].pct_change(period) * 100

            # CMO
            for period in [14, 20]:
                if len(df) >= period * 2:
                    try:
                        features[f'cmo_{period}'] = talib.CMO(close, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating CMO {period}: {e}")
                        # Simple CMO
                        changes = df['close'].diff()
                        gains = changes.where(changes > 0, 0).rolling(period, min_periods=period // 2).sum()
                        losses = -changes.where(changes < 0, 0).rolling(period, min_periods=period // 2).sum()
                        features[f'cmo_{period}'] = 100 * (gains - losses) / (gains + losses + 1e-10)

            # PPO
            if len(df) >= 26:
                try:
                    features['ppo'] = talib.PPO(close)
                except Exception as e:
                    logger.debug(f"Error calculating PPO: {e}")
                    # Simple PPO
                    ema12 = df['close'].ewm(span=12, min_periods=6).mean()
                    ema26 = df['close'].ewm(span=26, min_periods=13).mean()
                    features['ppo'] = ((ema12 - ema26) / ema26) * 100

            # TRIX
            for period in [14, 20]:
                if len(df) >= period * 3:
                    try:
                        features[f'trix_{period}'] = talib.TRIX(close, timeperiod=period)
                    except Exception as e:
                        logger.debug(f"Error calculating TRIX {period}: {e}")
                        # Simple TRIX
                        ema1 = df['close'].ewm(span=period, min_periods=period // 2).mean()
                        ema2 = ema1.ewm(span=period, min_periods=period // 2).mean()
                        ema3 = ema2.ewm(span=period, min_periods=period // 2).mean()
                        features[f'trix_{period}'] = ema3.pct_change() * 10000

        except Exception as e:
            logger.error(f"Error in technical indicators: {e}")

        return features

    def _create_microstructure_features(self, df: pd.DataFrame) -> Dict:
        """Create market microstructure features"""
        features = {}

        try:
            # Spread proxies
            features['hl_spread'] = (df['high'] - df['low']) / df['close']
            features['co_spread'] = abs(df['close'] - df['open']) / df['close']
            features['oc_spread'] = (df['close'] - df['open']) / df['open']

            # Intraday patterns
            features['intraday_momentum'] = (df['close'] - df['open']) / df['open']
            features['intraday_volatility'] = (df['high'] - df['low']) / df['open']

            # Shadows
            body_high = df[['close', 'open']].max(axis=1)
            body_low = df[['close', 'open']].min(axis=1)

            features['upper_shadow'] = (df['high'] - body_high) / df['close']
            features['lower_shadow'] = (body_low - df['low']) / df['close']
            features['shadow_ratio'] = features['upper_shadow'] / (features['lower_shadow'] + 1e-10)

            # Volume at price levels
            features['volume_at_high'] = df['volume'] * (df['close'] == df['high']).astype(int)
            features['volume_at_low'] = df['volume'] * (df['close'] == df['low']).astype(int)
            features['volume_at_close'] = df['volume'] * (
                        abs(df['close'] - df['high']) < abs(df['close'] - df['low'])).astype(int)

            # Order flow imbalance proxy
            features['order_flow_imbalance'] = (df['close'] - df['open']) * df['volume']
            features['ofi_ma'] = features['order_flow_imbalance'].rolling(20, min_periods=10).mean()
            features['ofi_std'] = features['order_flow_imbalance'].rolling(20, min_periods=10).std()
            features['ofi_zscore'] = (features['order_flow_imbalance'] - features['ofi_ma']) / (
                        features['ofi_std'] + 1e-10)

            # Microstructure noise
            if 'volatility_20d' in df.columns:
                volatility_20d = df['volatility_20d']
            else:
                volatility_20d = df['close'].pct_change().rolling(20, min_periods=10).std() * np.sqrt(252)
            features['noise_ratio'] = features['hl_spread'] / (volatility_20d + 1e-10)

            # Time-weighted average price
            features['twap'] = (df['high'] + df['low'] + df['close']) / 3
            features['price_to_twap'] = df['close'] / features['twap']

            # Amihud illiquidity
            features['amihud_illiquidity'] = abs(df['close'].pct_change()) / (df['volume'] + 1)
            features['amihud_ma'] = features['amihud_illiquidity'].rolling(20, min_periods=10).mean()

            # Kyle's lambda (simplified)
            price_change = df['close'].pct_change()
            signed_volume = df['volume'] * np.sign(price_change)
            features['kyle_lambda'] = abs(price_change) / (abs(signed_volume) + 1)

        except Exception as e:
            logger.error(f"Error in microstructure features: {e}")

        return features

    def _create_statistical_features(self, df: pd.DataFrame) -> Dict:
        """Create statistical features"""
        features = {}

        try:
            # Rolling statistics
            for period in [5, 10, 20, 50]:
                if len(df) >= period * 2:
                    rolling_returns = df['close'].pct_change().rolling(period, min_periods=period // 2)

                    # Higher moments
                    features[f'skew_{period}d'] = rolling_returns.skew()
                    features[f'kurtosis_{period}d'] = rolling_returns.kurt()

                    # Jarque-Bera test statistic
                    jb_stat = (period / 6) * (features[f'skew_{period}d'] ** 2 +
                                              0.25 * (features[f'kurtosis_{period}d'] - 3) ** 2)
                    features[f'jarque_bera_{period}d'] = jb_stat

            # Auto-correlation
            returns = df['close'].pct_change()
            for lag in [1, 2, 5, 10, 20]:
                if len(df) >= 50 + lag:
                    features[f'autocorr_lag_{lag}'] = returns.rolling(50, min_periods=25).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                    )

            # Hurst exponent
            for period in [20, 50, 100]:
                if len(df) >= period * 2:
                    features[f'hurst_{period}'] = df['close'].rolling(period, min_periods=period // 2).apply(
                        lambda x: self._calculate_hurst_exponent(x.values) if len(x) == period else 0.5
                    )

            # Entropy
            for period in [20, 50]:
                if len(df) >= period * 2:
                    try:
                        features[f'entropy_{period}'] = returns.rolling(period, min_periods=period // 2).apply(
                            lambda x: stats.entropy(np.histogram(x.dropna(), bins=10)[0] + 1e-10) if len(
                                x.dropna()) > 0 else 0
                        )
                    except:
                        features[f'entropy_{period}'] = 0

            # Price efficiency (Variance Ratio Test)
            for period in [10, 20]:
                if len(df) >= period * 2:
                    features[f'efficiency_ratio_{period}'] = self._calculate_efficiency_ratio(df['close'], period)

            # Z-score
            for period in [20, 50]:
                if len(df) >= period * 2:
                    mean = df['close'].rolling(period, min_periods=period // 2).mean()
                    std = df['close'].rolling(period, min_periods=period // 2).std()
                    features[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)

            # Percentile rank
            for period in [20, 50, 100]:
                if len(df) >= period * 2:
                    features[f'percentile_rank_{period}'] = df['close'].rolling(period, min_periods=period // 2).rank(
                        pct=True)

            # Mean reversion indicators
            for period in [20, 50]:
                if len(df) >= period * 2:
                    ma = df['close'].rolling(period, min_periods=period // 2).mean()
                    features[f'mean_reversion_score_{period}'] = -abs(df['close'] - ma) / ma

            # Trend consistency
            for period in [10, 20]:
                if len(df) >= period * 2:
                    positive_days = (df['close'].pct_change() > 0).rolling(period, min_periods=period // 2).sum()
                    features[f'trend_consistency_{period}'] = positive_days / period

        except Exception as e:
            logger.error(f"Error in statistical features: {e}")

        return features

    def _create_pattern_features(self, df: pd.DataFrame) -> Dict:
        """Create candlestick pattern features"""
        features = {}

        try:
            # Ensure float64 for TA-Lib
            open_price = df['open'].astype('float64').values
            high = df['high'].astype('float64').values
            low = df['low'].astype('float64').values
            close = df['close'].astype('float64').values

            # Basic candle properties
            body = close - open_price
            body_abs = abs(body)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low

            # Candlestick patterns using TA-Lib
            pattern_functions = [
                ('cdl_doji', talib.CDLDOJI),
                ('cdl_hammer', talib.CDLHAMMER),
                ('cdl_hanging_man', talib.CDLHANGINGMAN),
                ('cdl_engulfing', talib.CDLENGULFING),
                ('cdl_harami', talib.CDLHARAMI),
                ('cdl_morning_star', talib.CDLMORNINGSTAR),
                ('cdl_evening_star', talib.CDLEVENINGSTAR),
                ('cdl_3_white_soldiers', talib.CDL3WHITESOLDIERS),
                ('cdl_3_black_crows', talib.CDL3BLACKCROWS),
                ('cdl_spinning_top', talib.CDLSPINNINGTOP),
                ('cdl_shooting_star', talib.CDLSHOOTINGSTAR),
                ('cdl_marubozu', talib.CDLMARUBOZU),
                ('cdl_abandoned_baby', talib.CDLABANDONEDBABY),
                ('cdl_breakaway', talib.CDLBREAKAWAY),
                ('cdl_3_inside', talib.CDL3INSIDE),
                ('cdl_3_outside', talib.CDL3OUTSIDE)
            ]

            for name, func in pattern_functions:
                try:
                    pattern = func(open_price, high, low, close)
                    features[name] = pattern / 100  # Normalize to -1, 0, 1

                    # Pattern strength (how many periods since last occurrence)
                    pattern_series = pd.Series(pattern, index=df.index)
                    last_pattern = pattern_series.replace(0, np.nan).fillna(method='ffill')
                    features[f'{name}_strength'] = (pd.Series(range(len(pattern_series)), index=df.index) -
                                                    pd.Series(range(len(last_pattern)), index=df.index))
                except Exception as e:
                    logger.debug(f"Error calculating pattern {name}: {e}")
                    features[name] = 0
                    features[f'{name}_strength'] = 0

            # Custom patterns

            # Pin bar
            pin_bar_bull = ((lower_shadow > body_abs * 2) &
                            (upper_shadow < body_abs * 0.5) &
                            (close > open_price))
            features['pin_bar_bull'] = pin_bar_bull.astype(int)

            pin_bar_bear = ((upper_shadow > body_abs * 2) &
                            (lower_shadow < body_abs * 0.5) &
                            (close < open_price))
            features['pin_bar_bear'] = pin_bar_bear.astype(int)

            # Inside bar
            inside_bar = ((high < df['high'].shift(1)) &
                          (low > df['low'].shift(1)))
            features['inside_bar'] = inside_bar.astype(int)

            # Outside bar
            outside_bar = ((high > df['high'].shift(1)) &
                           (low < df['low'].shift(1)))
            features['outside_bar'] = outside_bar.astype(int)

            # Consecutive patterns
            for period in [3, 5]:
                if len(df) >= period:
                    features[f'consecutive_up_{period}'] = (df['close'] > df['open']).rolling(period,
                                                                                              min_periods=period // 2).sum() == period
                    features[f'consecutive_down_{period}'] = (df['close'] < df['open']).rolling(period,
                                                                                                min_periods=period // 2).sum() == period

            # Pattern combinations
            features['reversal_pattern'] = (
                                                   features.get('cdl_hammer', 0) +
                                                   features.get('cdl_morning_star', 0) +
                                                   features.get('pin_bar_bull', 0)
                                           ) > 0

            features['continuation_pattern'] = (
                                                       features.get('cdl_3_white_soldiers', 0) +
                                                       features.get('cdl_marubozu', 0)
                                               ) > 0

        except Exception as e:
            logger.error(f"Error in pattern features: {e}")

        return features

    def _create_interaction_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create interaction features between indicators"""
        features = {}

        try:
            # Price-Volume interactions
            if 'volume_ratio_20' in base_features and 'return_5d' in base_features:
                features['volume_momentum'] = base_features['volume_ratio_20'] * base_features['return_5d']
                features['volume_trend_alignment'] = (
                        (base_features['volume_ratio_20'] > 1) &
                        (base_features['return_5d'] > 0)
                ).astype(int)

            # RSI-MACD interaction
            if 'rsi_14' in base_features and 'macd_hist_12_26_9' in base_features:
                features['rsi_macd_confluence'] = (
                        ((base_features['rsi_14'] > 50) & (base_features['macd_hist_12_26_9'] > 0)) |
                        ((base_features['rsi_14'] < 50) & (base_features['macd_hist_12_26_9'] < 0))
                ).astype(int)

                features['rsi_macd_divergence'] = (
                        ((base_features['rsi_14'] > 70) & (base_features['macd_hist_12_26_9'] < 0)) |
                        ((base_features['rsi_14'] < 30) & (base_features['macd_hist_12_26_9'] > 0))
                ).astype(int)

            # Volatility-Volume interaction
            if 'volatility_20d' in base_features and 'volume_ratio_20' in base_features:
                features['high_vol_high_volume'] = (
                        (base_features['volatility_20d'] > base_features['volatility_20d'].rolling(50,
                                                                                                   min_periods=25).median()) &
                        (base_features['volume_ratio_20'] > 1.5)
                ).astype(int)

            # Support/Resistance interaction with indicators
            if 'dist_from_resistance_20d' in base_features and 'rsi_14' in base_features:
                features['resistance_overbought'] = (
                        (base_features['dist_from_resistance_20d'] < 0.02) &
                        (base_features['rsi_14'] > 70)
                ).astype(int)

            if 'dist_from_support_20d' in base_features and 'rsi_14' in base_features:
                features['support_oversold'] = (
                        (base_features['dist_from_support_20d'] < 0.02) &
                        (base_features['rsi_14'] < 30)
                ).astype(int)

            # Moving average interactions
            ma_periods = [col for col in base_features.columns if col.startswith('sma_') and col.count('_') == 1]
            if len(ma_periods) >= 2:
                # MA alignment score
                ma_values = base_features[ma_periods].values
                ma_sorted = np.sort(ma_values, axis=1)
                features['ma_alignment_score'] = np.mean(ma_values == ma_sorted, axis=1)

                # MA compression
                features['ma_compression'] = base_features[ma_periods].std(axis=1) / base_features[ma_periods].mean(
                    axis=1)

            # Bollinger Band and RSI interaction
            if 'bb_position_20_20' in base_features and 'rsi_14' in base_features:
                features['bb_rsi_squeeze'] = (
                                                     (base_features['bb_position_20_20'] > 0.8) &
                                                     (base_features['rsi_14'] > 70)
                                             ).astype(int) - (
                                                     (base_features['bb_position_20_20'] < 0.2) &
                                                     (base_features['rsi_14'] < 30)
                                             ).astype(int)

            # Multi-timeframe momentum
            momentum_cols = [col for col in base_features.columns if col.startswith('return_') and 'd' in col]
            if len(momentum_cols) >= 3:
                # Momentum alignment across timeframes
                momentum_signs = np.sign(base_features[momentum_cols])
                features['momentum_alignment'] = momentum_signs.sum(axis=1) / len(momentum_cols)

                # Momentum acceleration
                short_momentum = base_features[[col for col in momentum_cols if int(col.split('_')[1][:-1]) <= 5]].mean(
                    axis=1)
                long_momentum = base_features[[col for col in momentum_cols if int(col.split('_')[1][:-1]) >= 10]].mean(
                    axis=1)
                features['momentum_acceleration'] = short_momentum - long_momentum

        except Exception as e:
            logger.error(f"Error in interaction features: {e}")

        return features

    def _create_regime_features(self, df: pd.DataFrame) -> Dict:
        """Create market regime features"""
        features = {}

        try:
            # Ensure float64 for TA-Lib
            close = df['close'].astype('float64').values

            # Trend regime
            if len(df) >= 200:
                try:
                    sma_20 = talib.SMA(close, 20)
                    sma_50 = talib.SMA(close, 50)
                    sma_200 = talib.SMA(close, 200)

                    features['trend_regime_bullish'] = (
                            (df['close'] > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
                    ).astype(int)

                    features['trend_regime_bearish'] = (
                            (df['close'] < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
                    ).astype(int)
                except Exception as e:
                    logger.debug(f"Error calculating trend regime: {e}")
                    features['trend_regime_bullish'] = 0
                    features['trend_regime_bearish'] = 0
            else:
                # Simplified regime for less data
                if len(df) >= 50:
                    sma_20 = df['close'].rolling(20, min_periods=10).mean()
                    sma_50 = df['close'].rolling(50, min_periods=25).mean()
                    features['trend_regime_bullish'] = ((df['close'] > sma_20) & (sma_20 > sma_50)).astype(int)
                    features['trend_regime_bearish'] = ((df['close'] < sma_20) & (sma_20 < sma_50)).astype(int)
                else:
                    features['trend_regime_bullish'] = 0
                    features['trend_regime_bearish'] = 0

            # Volatility regime (using rolling percentile)
            vol = df['close'].pct_change().rolling(20, min_periods=10).std()
            if len(df) >= 252:
                vol_percentile = vol.rolling(252, min_periods=126).rank(pct=True)
            else:
                vol_percentile = vol.rolling(len(df) // 2, min_periods=len(df) // 4).rank(pct=True)

            features['low_vol_regime'] = (vol_percentile < 0.3).astype(int)
            features['high_vol_regime'] = (vol_percentile > 0.7).astype(int)

            # Volume regime
            vol_ma = df['volume'].rolling(20, min_periods=10).mean()
            if len(df) >= 252:
                vol_percentile = vol_ma.rolling(252, min_periods=126).rank(pct=True)
            else:
                vol_percentile = vol_ma.rolling(len(df) // 2, min_periods=len(df) // 4).rank(pct=True)

            features['high_volume_regime'] = (vol_percentile > 0.7).astype(int)
            features['low_volume_regime'] = (vol_percentile < 0.3).astype(int)

            # Momentum regime
            if len(df) >= 20:
                try:
                    roc_20 = talib.ROC(close, 20)
                    if len(df) >= 252:
                        momentum_percentile = pd.Series(roc_20).rolling(252, min_periods=126).rank(pct=True)
                    else:
                        momentum_percentile = pd.Series(roc_20).rolling(len(df) // 2, min_periods=len(df) // 4).rank(
                            pct=True)

                    features['strong_momentum_regime'] = (momentum_percentile > 0.8).astype(int)
                    features['weak_momentum_regime'] = (momentum_percentile < 0.2).astype(int)
                except Exception as e:
                    logger.debug(f"Error calculating momentum regime: {e}")
                    features['strong_momentum_regime'] = 0
                    features['weak_momentum_regime'] = 0

            # Market efficiency regime
            efficiency_ratio = self._calculate_efficiency_ratio(df['close'], 20)
            features['trending_market'] = (efficiency_ratio > 0.7).astype(int)
            features['ranging_market'] = (efficiency_ratio < 0.3).astype(int)

        except Exception as e:
            logger.error(f"Error in regime features: {e}")

        return features

    def _create_ml_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create ML-discovered features"""
        features = {}

        try:
            # Polynomial features for key indicators
            key_features = ['rsi_14', 'macd_hist_12_26_9', 'bb_position_20_20', 'volume_ratio_20']

            for feat in key_features:
                if feat in base_features:
                    # Non-linear transformations
                    features[f'{feat}_squared'] = base_features[feat] ** 2
                    features[f'{feat}_cubed'] = base_features[feat] ** 3
                    features[f'{feat}_sqrt'] = np.sqrt(np.abs(base_features[feat]))
                    features[f'{feat}_log'] = np.log(np.abs(base_features[feat]) + 1)

            # Interaction polynomials
            if 'rsi_14' in base_features and 'volume_ratio_20' in base_features:
                features['rsi_volume_interaction'] = base_features['rsi_14'] * base_features['volume_ratio_20']
                features['rsi_volume_poly'] = (base_features['rsi_14'] ** 2) * base_features['volume_ratio_20']

            # Fourier features for cyclical patterns
            if len(df) >= 252:  # At least 1 year of data
                close_detrended = df['close'] - df['close'].rolling(252, min_periods=126).mean()

                # FFT to find dominant frequencies
                fft = np.fft.fft(close_detrended.fillna(0).values)
                frequencies = np.fft.fftfreq(len(close_detrended))

                # Get top 3 frequencies
                power = np.abs(fft)
                top_freq_idx = np.argsort(power)[-4:-1]  # Exclude DC component

                for i, idx in enumerate(top_freq_idx):
                    if frequencies[idx] > 0:  # Only positive frequencies
                        period = int(1 / frequencies[idx])
                        if 5 <= period <= 100:  # Reasonable period range
                            features[f'cycle_{period}d_sin'] = np.sin(2 * np.pi * np.arange(len(df)) / period)
                            features[f'cycle_{period}d_cos'] = np.cos(2 * np.pi * np.arange(len(df)) / period)

            # Wavelet features (simplified)
            if len(df) >= 64:
                close_array = df['close'].values

                # Simple wavelet-like decomposition using differences at multiple scales
                for scale in [2, 4, 8, 16]:
                    if len(close_array) >= scale * 2:
                        features[f'wavelet_d{scale}'] = pd.Series(close_array).diff(scale).values
                        features[f'wavelet_smooth{scale}'] = pd.Series(close_array).rolling(scale,
                                                                                            min_periods=scale // 2).mean().values

            # Entropy-based features
            for window in [20, 50]:
                if len(df) >= window * 2:
                    returns = df['close'].pct_change()

                    # Sample entropy
                    features[f'sample_entropy_{window}'] = returns.rolling(window, min_periods=window // 2).apply(
                        lambda x: self._sample_entropy(x.values, 2, 0.2 * x.std()) if len(x) == window else 0
                    )

            # Fractal dimension
            for window in [30, 60]:
                if len(df) >= window * 2:
                    features[f'fractal_dim_{window}'] = df['close'].rolling(window, min_periods=window // 2).apply(
                        lambda x: self._calculate_fractal_dimension(x.values) if len(x) == window else 1.5
                    )

        except Exception as e:
            logger.error(f"Error in ML features: {e}")

        return features

    def _create_advanced_interaction_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create advanced interaction features for sophisticated trading strategies"""
        features = {}

        try:
            # Ensure we have the required base features
            close = df['close'].astype('float64')

            # MOVING AVERAGE CROSSOVERS AND RELATIONSHIPS
            # Golden Cross / Death Cross
            if 'sma_50' in base_features and 'sma_200' in base_features:
                sma_50 = base_features['sma_50']
                sma_200 = base_features['sma_200']

                # Golden cross (50 crosses above 200)
                features['golden_cross'] = ((sma_50 > sma_200) &
                                            (sma_50.shift(1) <= sma_200.shift(1))).astype(int)

                # Death cross (50 crosses below 200)
                features['death_cross'] = ((sma_50 < sma_200) &
                                           (sma_50.shift(1) >= sma_200.shift(1))).astype(int)

                # Distance from golden/death cross
                features['sma_50_200_ratio'] = sma_50 / sma_200
                features['sma_50_200_diff'] = (sma_50 - sma_200) / sma_200

                # Days since golden/death cross
                golden_cross_cumsum = features['golden_cross'].cumsum()
                death_cross_cumsum = features['death_cross'].cumsum()
                features['golden_cross_days_since'] = df.groupby(golden_cross_cumsum).cumcount()
                features['death_cross_days_since'] = df.groupby(death_cross_cumsum).cumcount()

            # Short-term MA crossovers
            if 'sma_5' in base_features and 'sma_20' in base_features:
                sma_5 = base_features['sma_5']
                sma_20 = base_features['sma_20']

                features['sma_5_20_cross_up'] = ((sma_5 > sma_20) &
                                                 (sma_5.shift(1) <= sma_20.shift(1))).astype(int)
                features['sma_5_20_cross_down'] = ((sma_5 < sma_20) &
                                                   (sma_5.shift(1) >= sma_20.shift(1))).astype(int)

            # EMA crossovers
            if 'ema_12' in base_features and 'ema_26' in base_features:
                ema_12 = base_features['ema_12']
                ema_26 = base_features['ema_26']

                features['ema_cross_bullish'] = ((ema_12 > ema_26) &
                                                 (ema_12.shift(1) <= ema_26.shift(1))).astype(int)
                features['ema_cross_bearish'] = ((ema_12 < ema_26) &
                                                 (ema_12.shift(1) >= ema_26.shift(1))).astype(int)

            # Multiple timeframe MA alignment
            ma_cols = ['sma_20', 'sma_50', 'sma_100', 'sma_200']
            available_ma_cols = [col for col in ma_cols if col in base_features]

            if len(available_ma_cols) >= 3:
                # Count how many MAs the price is above
                ma_above_count = 0
                for ma_col in available_ma_cols:
                    ma_above_count += (close > base_features[ma_col]).astype(int)

                features['ma_alignment_score'] = ma_above_count / len(available_ma_cols)
                features['price_above_all_ma'] = (ma_above_count == len(available_ma_cols)).astype(int)
                features['price_below_all_ma'] = (ma_above_count == 0).astype(int)

            # MOMENTUM DIVERGENCES
            # RSI Divergence
            if 'rsi_14' in base_features:
                rsi = base_features['rsi_14']

                # Price makes higher high, RSI makes lower high (bearish divergence)
                price_higher = close > close.rolling(20, min_periods=10).max().shift(1)
                rsi_lower = rsi < rsi.rolling(20, min_periods=10).max().shift(1)
                features['bearish_divergence'] = (price_higher & rsi_lower).astype(int)

                # Price makes lower low, RSI makes higher low (bullish divergence)
                price_lower = close < close.rolling(20, min_periods=10).min().shift(1)
                rsi_higher = rsi > rsi.rolling(20, min_periods=10).min().shift(1)
                features['bullish_divergence'] = (price_lower & rsi_higher).astype(int)

            # MACD divergences and histogram analysis
            if 'macd_12_26_9' in base_features and 'macd_hist_12_26_9' in base_features:
                macd = base_features['macd_12_26_9']
                macd_signal = base_features['macd_signal_12_26_9']
                macd_hist = base_features['macd_hist_12_26_9']

                # MACD crossovers
                features['macd_cross_up'] = ((macd > macd_signal) &
                                             (macd.shift(1) <= macd_signal.shift(1))).astype(int)
                features['macd_cross_down'] = ((macd < macd_signal) &
                                               (macd.shift(1) >= macd_signal.shift(1))).astype(int)

                # MACD histogram analysis
                features['macd_histogram_growing'] = (macd_hist > macd_hist.shift(1)).astype(int)
                features['macd_histogram_accelerating'] = ((macd_hist > macd_hist.shift(1)) &
                                                           (macd_hist.shift(1) > macd_hist.shift(2))).astype(int)

            # Stochastic + RSI combination
            if 'rsi_14' in base_features and 'stoch_k_14_3' in base_features:
                rsi = base_features['rsi_14']
                stoch_k = base_features['stoch_k_14_3']

                features['oversold_combo'] = ((rsi < 30) & (stoch_k < 20)).astype(int)
                features['overbought_combo'] = ((rsi > 70) & (stoch_k > 80)).astype(int)

            # VOLUME-PRICE CONFIRMATIONS
            if 'volume_ratio_20' in base_features:
                vol_ratio = base_features['volume_ratio_20']
                returns = df['close'].pct_change()

                # Volume confirmation of price moves
                features['volume_price_confirm'] = ((returns > 0) & (vol_ratio > 1)).astype(int)
                features['volume_price_diverge'] = ((returns > 0) & (vol_ratio < 0.8)).astype(int)

                # High volume reversal
                features['high_volume_reversal'] = ((returns * returns.shift(1) < 0) &
                                                    (vol_ratio > 1.5)).astype(int)

            # On-Balance Volume divergence
            if 'obv' in base_features:
                obv = base_features['obv']

                # OBV rising while price falling (bullish)
                obv_rising = obv > obv.rolling(20, min_periods=10).mean()
                price_falling = close < close.rolling(20, min_periods=10).mean()
                features['obv_bullish_divergence'] = (obv_rising & price_falling).astype(int)

                # OBV falling while price rising (bearish)
                obv_falling = obv < obv.rolling(20, min_periods=10).mean()
                price_rising = close > close.rolling(20, min_periods=10).mean()
                features['obv_bearish_divergence'] = (obv_falling & price_rising).astype(int)

            # VOLATILITY PATTERN INTERACTIONS
            # Bollinger Band patterns
            if 'bb_width_20_20' in base_features and 'bb_position_20_20' in base_features:
                bb_width = base_features['bb_width_20_20']
                bb_position = base_features['bb_position_20_20']

                # Bollinger squeeze (low volatility)
                if len(df) >= 100:
                    bb_width_percentile = bb_width.rolling(100, min_periods=50).rank(pct=True)
                else:
                    bb_width_percentile = bb_width.rolling(len(df) // 2, min_periods=len(df) // 4).rank(pct=True)

                features['bb_squeeze'] = (bb_width_percentile < 0.2).astype(int)
                features['bb_expansion'] = (bb_width_percentile > 0.8).astype(int)

                # Bollinger bounce patterns
                features['bb_lower_bounce'] = ((bb_position < 0.1) &
                                               (bb_position > bb_position.shift(1))).astype(int)
                features['bb_upper_bounce'] = ((bb_position > 0.9) &
                                               (bb_position < bb_position.shift(1))).astype(int)

            # Volatility breakout
            if 'volatility_20d' in base_features and 'volume_ratio_20' in base_features:
                vol_20 = base_features['volatility_20d']
                vol_ratio = base_features['volume_ratio_20']

                features['volatility_breakout'] = ((vol_20 > vol_20.shift(1) * 1.5) &
                                                   (vol_ratio > 1.5)).astype(int)

            # Keltner + Bollinger combination
            if 'bb_upper_20_20' in base_features and 'kc_upper_20_20' in base_features:
                bb_upper = base_features['bb_upper_20_20']
                bb_lower = base_features['bb_lower_20_20']
                kc_upper = base_features['kc_upper_20_20']
                kc_lower = base_features['kc_lower_20_20']

                # TTM Squeeze indicator
                features['kc_bb_squeeze'] = ((bb_upper < kc_upper) &
                                             (bb_lower > kc_lower)).astype(int)

            # SUPPORT/RESISTANCE INTERACTIONS
            if 'dist_from_resistance_20d' in base_features and 'dist_from_support_20d' in base_features:
                dist_resistance = base_features['dist_from_resistance_20d']
                dist_support = base_features['dist_from_support_20d']

                features['near_resistance'] = (dist_resistance < 0.02).astype(int)
                features['near_support'] = (dist_support < 0.02).astype(int)
                features['sr_squeeze'] = ((dist_resistance < 0.05) & (dist_support < 0.05)).astype(int)

            # TREND STRENGTH COMBINATIONS
            if 'adx_14' in base_features and 'trend_strength_14' in base_features:
                adx = base_features['adx_14']
                trend_strength = base_features['trend_strength_14']

                features['strong_uptrend'] = ((adx > 25) & (trend_strength > 0.5)).astype(int)
                features['strong_downtrend'] = ((adx > 25) & (trend_strength < -0.5)).astype(int)
                features['weak_trend'] = (adx < 20).astype(int)

            # PATTERN + VOLUME COMBINATIONS
            if 'cdl_engulfing' in base_features and 'volume_ratio_20' in base_features:
                engulfing = base_features['cdl_engulfing']
                vol_ratio = base_features['volume_ratio_20']

                features['bullish_engulfing_volume'] = ((engulfing > 0) & (vol_ratio > 1.2)).astype(int)
                features['bearish_engulfing_volume'] = ((engulfing < 0) & (vol_ratio > 1.2)).astype(int)

            # MULTIPLE TIMEFRAME MOMENTUM
            momentum_cols = ['return_5d', 'return_10d', 'return_20d']
            available_momentum = [col for col in momentum_cols if col in base_features]

            if len(available_momentum) >= 2:
                # Momentum alignment
                momentum_positive = 0
                for col in available_momentum:
                    momentum_positive += (base_features[col] > 0).astype(int)

                features['momentum_alignment'] = momentum_positive / len(available_momentum)
                features['all_momentum_positive'] = (momentum_positive == len(available_momentum)).astype(int)
                features['all_momentum_negative'] = (momentum_positive == 0).astype(int)

            # COMPOSITE MARKET SCORES
            # Bull Market Score (0-1)
            bull_score_components = []

            if 'trend_regime_bullish' in base_features:
                bull_score_components.append(base_features['trend_regime_bullish'])
            if 'ma_alignment_score' in features:
                bull_score_components.append((features['ma_alignment_score'] > 0.75).astype(int))
            if 'momentum_alignment' in features:
                bull_score_components.append((features['momentum_alignment'] > 0.66).astype(int))
            if 'volume_price_confirm' in features:
                bull_score_components.append(features['volume_price_confirm'])
            if 'adx_14' in base_features:
                bull_score_components.append((base_features['adx_14'] > 20).astype(int))

            if bull_score_components:
                features['bull_market_score'] = sum(bull_score_components) / len(bull_score_components)

            # Mean Reversion Setup Score (0-1)
            reversion_components = []

            if 'rsi_14' in base_features:
                reversion_components.append((base_features['rsi_14'] < 30).astype(int))
            if 'price_to_sma_20' in base_features:
                reversion_components.append((base_features['price_to_sma_20'] < 0.97).astype(int))
            if 'bb_position_20_20' in base_features:
                reversion_components.append((base_features['bb_position_20_20'] < 0.1).astype(int))
            if 'volume_ratio_20' in base_features:
                reversion_components.append((base_features['volume_ratio_20'] > 1).astype(int))

            if reversion_components:
                features['mean_reversion_setup'] = sum(reversion_components) / len(reversion_components)

            # Breakout Setup Score (0-1)
            breakout_components = []

            if 'near_resistance' in features:
                breakout_components.append(features['near_resistance'])
            if 'volume_ratio_20' in base_features:
                breakout_components.append((base_features['volume_ratio_20'] > 1.5).astype(int))
            if 'bb_squeeze' in features:
                breakout_components.append(features['bb_squeeze'])
            if 'momentum_alignment' in features:
                breakout_components.append((features['momentum_alignment'] > 0.66).astype(int))
            if 'volatility_breakout' in features:
                breakout_components.append(features['volatility_breakout'])

            if breakout_components:
                features['breakout_setup'] = sum(breakout_components) / len(breakout_components)

            # Trend Exhaustion Score (0-1)
            exhaustion_components = []

            if 'rsi_14' in base_features:
                exhaustion_components.append(((base_features['rsi_14'] > 70) |
                                              (base_features['rsi_14'] < 30)).astype(int))
            if 'price_to_sma_200' in base_features:
                exhaustion_components.append((abs(base_features['price_to_sma_200'] - 1) > 0.15).astype(int))
            if 'volume_ratio_20' in base_features:
                exhaustion_components.append((base_features['volume_ratio_20'] < 0.8).astype(int))
            if 'bearish_divergence' in features:
                exhaustion_components.append(features['bearish_divergence'])

            if exhaustion_components:
                features['trend_exhaustion'] = sum(exhaustion_components) / len(exhaustion_components)

            # RISK INDICATORS
            # Volatility regime change
            if 'vol_regime' in base_features:
                vol_regime = base_features['vol_regime']
                features['volatility_regime_change'] = (vol_regime != vol_regime.shift(5)).astype(int)

            # Multiple indicator confirmation
            if 'rsi_14' in base_features and 'macd_hist_12_26_9' in base_features and 'stoch_k_14_3' in base_features:
                rsi = base_features['rsi_14']
                macd_hist = base_features['macd_hist_12_26_9']
                stoch_k = base_features['stoch_k_14_3']

                # Triple confirmation buy
                features['triple_buy_signal'] = ((rsi < 40) &
                                                 (macd_hist > 0) &
                                                 (stoch_k < 30)).astype(int)

                # Triple confirmation sell
                features['triple_sell_signal'] = ((rsi > 60) &
                                                  (macd_hist < 0) &
                                                  (stoch_k > 70)).astype(int)

            # Market breadth proxy (using volume and price action)
            if 'volume_ratio_20' in base_features:
                returns = df['close'].pct_change()
                vol_ratio = base_features['volume_ratio_20']

                # Accumulation days (positive return with high volume)
                features['accumulation_day'] = ((returns > 0.01) & (vol_ratio > 1.2)).astype(int)

                # Distribution days (negative return with high volume)
                features['distribution_day'] = ((returns < -0.01) & (vol_ratio > 1.2)).astype(int)

                # Rolling accumulation/distribution count
                features['accumulation_count_20d'] = features['accumulation_day'].rolling(20, min_periods=10).sum()
                features['distribution_count_20d'] = features['distribution_day'].rolling(20, min_periods=10).sum()
                features['acc_dist_ratio'] = (features['accumulation_count_20d'] /
                                              (features['distribution_count_20d'] + 1))

            # Price action quality
            if 'high_low_range' in base_features:
                hl_range = base_features['high_low_range']
                returns = df['close'].pct_change()

                # Strong close (closing near high of day)
                features['strong_close'] = ((close - df['low']) / (df['high'] - df['low'] + 1e-10) > 0.8).astype(int)
                features['weak_close'] = ((close - df['low']) / (df['high'] - df['low'] + 1e-10) < 0.2).astype(int)

                # Range expansion
                features['range_expansion'] = (hl_range > hl_range.rolling(20, min_periods=10).mean() * 1.5).astype(int)
                features['range_contraction'] = (hl_range < hl_range.rolling(20, min_periods=10).mean() * 0.5).astype(
                    int)

        except Exception as e:
            logger.error(f"Error in advanced interaction features: {e}")

        return features

    def _create_seasonal_features(self, df: pd.DataFrame) -> Dict:
        """Create seasonal and calendar-based features - METHOD 29"""
        features = {}

        try:
            # Get datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Day of week features
            features['day_of_week'] = df.index.dayofweek
            features['is_monday'] = (features['day_of_week'] == 0).astype(int)
            features['is_friday'] = (features['day_of_week'] == 4).astype(int)

            # Month features
            features['month'] = df.index.month
            features['quarter'] = df.index.quarter
            features['is_month_start'] = df.index.is_month_start.astype(int)
            features['is_month_end'] = df.index.is_month_end.astype(int)
            features['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            features['is_quarter_end'] = df.index.is_quarter_end.astype(int)

            # Trading day of month
            features['trading_day_of_month'] = df.groupby(pd.Grouper(freq='M')).cumcount() + 1

            # Days until month end
            month_end = df.index.to_period('M').to_timestamp('M')
            features['days_until_month_end'] = (month_end - df.index).days

            # Seasonal patterns
            features['is_january'] = (features['month'] == 1).astype(int)  # January effect
            features['is_december'] = (features['month'] == 12).astype(int)  # Year-end effect
            features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)  # Summer doldrums
            features['is_earning_season'] = features['month'].isin([1, 4, 7, 10]).astype(int)

            # Options expiration week (third Friday)
            features['is_opex_week'] = df.index.map(self._is_opex_week).astype(int)

            # Holiday effects (simplified)
            features['days_from_holiday'] = df.index.map(self._days_from_nearest_holiday)
            features['is_pre_holiday'] = (features['days_from_holiday'] == -1).astype(int)
            features['is_post_holiday'] = (features['days_from_holiday'] == 1).astype(int)

            # Turn of month effect (last 4 and first 3 days)
            features['turn_of_month'] = ((features['trading_day_of_month'] <= 3) |
                                         (features['days_until_month_end'] <= 4)).astype(int)

            # Sine and cosine encoding for cyclical features
            features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 5)
            features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 5)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

        except Exception as e:
            logger.error(f"Error in seasonal features: {e}")

        return features

    def _create_momentum_features(self, df: pd.DataFrame, base_features: pd.DataFrame) -> Dict:
        """Create advanced momentum features - METHOD 30"""
        features = {}

        try:
            # Time-series momentum
            for lookback in [20, 60, 120, 252]:
                if len(df) >= lookback + 20:
                    # Absolute momentum
                    features[f'momentum_{lookback}d'] = df['close'] / df['close'].shift(lookback) - 1

                    # Relative momentum (vs rolling average)
                    ma = df['close'].rolling(lookback, min_periods=lookback // 2).mean()
                    features[f'relative_momentum_{lookback}d'] = df['close'] / ma - 1

                    # Momentum quality (consistency)
                    returns = df['close'].pct_change()
                    positive_days = (returns > 0).rolling(lookback, min_periods=lookback // 2).sum()
                    features[f'momentum_quality_{lookback}d'] = positive_days / lookback

            # Dual momentum
            if 'momentum_60d' in features and 'momentum_252d' in features:
                features['dual_momentum'] = (features['momentum_60d'] + features['momentum_252d']) / 2
                features['momentum_spread'] = features['momentum_60d'] - features['momentum_252d']

            # Momentum acceleration
            for period in [20, 60]:
                if f'momentum_{period}d' in features:
                    features[f'momentum_accel_{period}d'] = features[f'momentum_{period}d'].diff(5)

            # Price momentum oscillator
            if len(df) >= 35:
                roc1 = df['close'].pct_change(10) * 100
                roc2 = roc1.rolling(10, min_periods=5).mean()
                features['price_momentum_oscillator'] = roc1 - roc2

            # Momentum divergence index
            if 'rsi_14' in base_features and 'momentum_20d' in features:
                price_momentum = features['momentum_20d']
                rsi_momentum = base_features['rsi_14'].pct_change(20)
                features['momentum_divergence_index'] = price_momentum - rsi_momentum

            # Sector relative momentum (if available)
            # This would compare symbol momentum to sector momentum
            features['relative_strength'] = 0  # Placeholder - would need sector data

            # Momentum regime
            if 'momentum_60d' in features:
                mom_60 = features['momentum_60d']
                features['momentum_regime_strong'] = (
                            mom_60 > mom_60.rolling(252, min_periods=126).quantile(0.8)).astype(int)
                features['momentum_regime_weak'] = (mom_60 < mom_60.rolling(252, min_periods=126).quantile(0.2)).astype(
                    int)

            # Time-based momentum decay
            for period in [5, 10, 20]:
                if f'return_{period}d' in base_features:
                    # Weight recent returns more heavily
                    weights = np.exp(-np.arange(period) / period)
                    weights = weights / weights.sum()

                    weighted_returns = df['close'].pct_change().rolling(period).apply(
                        lambda x: np.sum(x.values * weights[:len(x)]) if len(x) == period else np.nan
                    )
                    features[f'weighted_momentum_{period}d'] = weighted_returns

            # Momentum consistency score
            returns = df['close'].pct_change()
            for period in [20, 60]:
                if len(df) >= period:
                    # Calculate rolling correlation of returns with time
                    time_series = np.arange(period)

                    def trend_strength(x):
                        if len(x) == period:
                            return np.corrcoef(x, time_series)[0, 1]
                        return 0

                    features[f'momentum_consistency_{period}d'] = returns.rolling(period).apply(trend_strength)

            # Momentum breakout
            if 'momentum_20d' in features:
                mom_20 = features['momentum_20d']
                mom_high = mom_20.rolling(60, min_periods=30).max()
                mom_low = mom_20.rolling(60, min_periods=30).min()

                features['momentum_breakout_up'] = (mom_20 > mom_high.shift(1)).astype(int)
                features['momentum_breakout_down'] = (mom_20 < mom_low.shift(1)).astype(int)

        except Exception as e:
            logger.error(f"Error in momentum features: {e}")

        return features

    def _calculate_divergence(self, price: np.ndarray, indicator: np.ndarray,
                              window: int = 20) -> np.ndarray:
        """Calculate divergence between price and indicator"""

        if isinstance(price, pd.Series):
            price = price.values
        if isinstance(indicator, pd.Series):
            indicator = indicator.values

        divergence = np.zeros(len(price))

        if len(price) < window * 2:
            return divergence

        for i in range(window, len(price)):
            if i < window * 2:
                continue

            # Get recent window
            price_window = price[i - window:i]
            indicator_window = indicator[i - window:i]

            # Skip if NaN values
            if np.isnan(price_window).any() or np.isnan(indicator_window).any():
                continue

            # Find local extremes
            price_max_idx = np.argmax(price_window)
            price_min_idx = np.argmin(price_window)
            ind_max_idx = np.argmax(indicator_window)
            ind_min_idx = np.argmin(indicator_window)

            # Check for divergence
            # Bearish divergence: price makes higher high, indicator makes lower high
            if price_max_idx > window // 2:  # Recent high
                prev_price_max = np.max(price_window[:window // 2])
                prev_ind_max = np.max(indicator_window[:window // 2])

                if (price_window[price_max_idx] > prev_price_max and
                        indicator_window[ind_max_idx] < prev_ind_max):
                    divergence[i] = -1  # Bearish

            # Bullish divergence: price makes lower low, indicator makes higher low
            if price_min_idx > window // 2:  # Recent low
                prev_price_min = np.min(price_window[:window // 2])
                prev_ind_min = np.min(indicator_window[:window // 2])

                if (price_window[price_min_idx] < prev_price_min and
                        indicator_window[ind_min_idx] > prev_ind_min):
                    divergence[i] = 1  # Bullish

        return divergence

    def _calculate_hurst_exponent(self, series: np.ndarray) -> float:
        """Calculate Hurst exponent for mean reversion/trending detection"""

        if len(series) < 20:
            return 0.5

        # Calculate the range of cumulative deviations
        mean = np.mean(series)
        deviations = series - mean
        cumulative_deviations = np.cumsum(deviations)
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

        # Calculate standard deviation
        S = np.std(series, ddof=1)

        if S == 0:
            return 0.5

        # Calculate the Hurst exponent
        if R == 0:
            return 0.5

        return np.log(R / S) / np.log(len(series) / 2)

    def _calculate_efficiency_ratio(self, price: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""

        if len(price) < period:
            return pd.Series(index=price.index, dtype=float)

        # Direction (net change over period)
        direction = abs(price.diff(period))

        # Volatility (sum of absolute changes)
        volatility = price.diff().abs().rolling(period, min_periods=period // 2).sum()

        # Efficiency ratio
        efficiency_ratio = direction / (volatility + 1e-10)
        efficiency_ratio = efficiency_ratio.fillna(0)

        return efficiency_ratio

    def _sample_entropy(self, series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy of time series"""

        N = len(series)
        if N < m + 1:
            return 0

        def _maxdist(xi, xj, m):
            """Calculate maximum distance between patterns"""
            return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])

        def _phi(m):
            """Calculate phi(m)"""
            templates = np.array([series[i:i + m] for i in range(N - m + 1)])
            C = 0
            for i in range(N - m + 1):
                template_i = templates[i]
                C += sum([1 for j in range(N - m + 1) if i != j and
                          _maxdist(template_i, templates[j], m) <= r])
            return C / (N - m + 1) / (N - m) if (N - m + 1) * (N - m) > 0 else 0

        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            if phi_m > 0 and phi_m1 > 0:
                return -np.log(phi_m1 / phi_m)
            else:
                return 0
        except:
            return 0

    def _calculate_fractal_dimension(self, series: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""

        if len(series) < 10:
            return 1.5

        # Normalize series
        series = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-10)

        # Box sizes
        n = len(series)
        scales = []
        counts = []

        for scale in [2, 4, 8, 16, 32]:
            if scale >= n:
                break

            # Count boxes needed
            boxes = 0
            for i in range(0, n - scale, scale):
                segment = series[i:i + scale]
                boxes += int(np.ceil(np.max(segment) * scale) - np.floor(np.min(segment) * scale)) + 1

            scales.append(np.log(scale))
            counts.append(np.log(boxes))

        if len(scales) < 2:
            return 1.5

        # Linear regression
        try:
            slope, _ = np.polyfit(scales, counts, 1)
            return -slope
        except:
            return 1.5

    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with multiple strategies"""

        # First, forward fill for time series continuity
        features = features.fillna(method='ffill')

        # Then backward fill for any remaining NaN at the beginning
        features = features.fillna(method='bfill')

        # For any remaining NaN, use column-specific strategies
        for col in features.columns:
            if features[col].isna().any():
                # For ratio/percentage features, use 1.0 (neutral)
                if any(x in col for x in ['ratio', 'pct', 'position', 'score']):
                    features[col] = features[col].fillna(1.0)
                # For binary features, use 0
                elif any(x in col for x in ['signal', 'cross', 'oversold', 'overbought', 'regime']):
                    features[col] = features[col].fillna(0)
                # For distance/difference features, use 0
                elif any(x in col for x in ['dist', 'diff', 'spread']):
                    features[col] = features[col].fillna(0)
                # For everything else, use median
                else:
                    median_val = features[col].median()
                    if pd.notna(median_val):
                        features[col] = features[col].fillna(median_val)
                    else:
                        features[col] = features[col].fillna(0)

        # Final check - if any NaN remains, drop those rows
        initial_len = len(features)
        features = features.dropna()
        if len(features) < initial_len:
            logger.debug(f"Dropped {initial_len - len(features)} rows with remaining NaN values")

        return features

    def _is_opex_week(self, date: pd.Timestamp) -> bool:
        """Check if date is in options expiration week (third Friday)"""
        # Find third Friday of the month
        first_day = date.replace(day=1)
        first_friday = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + pd.Timedelta(weeks=2)

        # Check if current date is within 5 days of third Friday
        return abs((date - third_friday).days) <= 5

    def _days_from_nearest_holiday(self, date: pd.Timestamp) -> int:
        """Calculate days from nearest major market holiday (simplified)"""
        # Major US market holidays (simplified list)
        holidays = [
            pd.Timestamp(date.year, 1, 1),  # New Year
            pd.Timestamp(date.year, 7, 4),  # Independence Day
            pd.Timestamp(date.year, 12, 25),  # Christmas
        ]

        # Find nearest holiday
        min_distance = float('inf')
        for holiday in holidays:
            distance = (date - holiday).days
            if abs(distance) < abs(min_distance):
                min_distance = distance

        return min_distance

    def _apply_feature_selection(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply feature selection based on importance and correlation"""

        # For now, return all features
        # In production, you would:
        # 1. Remove highly correlated features
        # 2. Remove low importance features
        # 3. Apply dimensionality reduction if needed

        return features

    def _create_price_features_gpu(self, features_df, close, high, low, open_price):
        """GPU-accelerated price feature creation"""
        # This would contain GPU-specific implementations
        # For now, it's a placeholder
        return features_df

    def _create_volume_features_gpu(self, features_df, close, volume):
        """GPU-accelerated volume feature creation"""
        # This would contain GPU-specific implementations
        # For now, it's a placeholder
        return features_df

    def get_feature_summary(self) -> Dict:
        """Get summary of generated features"""

        if not self.feature_names:
            return {"total_features": 0, "by_category": {}, "key_interactions": {}}

        summary = {
            "total_features": len(self.feature_names),
            "by_category": {
                "price": len(
                    [f for f in self.feature_names if any(x in f for x in ['price', 'return', 'close', 'sma', 'ema'])]),
                "volume": len([f for f in self.feature_names if 'volume' in f or 'vol' in f]),
                "volatility": len(
                    [f for f in self.feature_names if any(x in f for x in ['volatility', 'atr', 'bb', 'std'])]),
                "technical": len(
                    [f for f in self.feature_names if any(x in f for x in ['rsi', 'macd', 'stoch', 'cci', 'mfi'])]),
                "microstructure": len(
                    [f for f in self.feature_names if any(x in f for x in ['spread', 'noise', 'amihud'])]),
                "pattern": len([f for f in self.feature_names if 'cdl' in f or 'pattern' in f]),
                "regime": len([f for f in self.feature_names if 'regime' in f]),
                "ml": len([f for f in self.feature_names if any(x in f for x in ['entropy', 'fractal', 'hurst'])]),
                "interaction": len([f for f in self.feature_names if
                                    any(x in f for x in ['cross', 'divergence', 'signal', 'confluence'])]),
                "seasonal": len([f for f in self.feature_names if
                                 any(x in f for x in ['day_of_week', 'month', 'quarter', 'holiday'])]),
                "momentum": len([f for f in self.feature_names if 'momentum' in f])
            },
            "key_interactions": {
                "ma_crossovers": len([f for f in self.feature_names if 'cross' in f and ('sma' in f or 'ema' in f)]),
                "divergences": len([f for f in self.feature_names if 'divergence' in f]),
                "market_scores": len([f for f in self.feature_names if 'score' in f]),
                "regime_features": len([f for f in self.feature_names if 'regime' in f]),
                "composite_signals": len(
                    [f for f in self.feature_names if any(x in f for x in ['triple_', 'combo', 'setup'])])
            }
        }

        return summary