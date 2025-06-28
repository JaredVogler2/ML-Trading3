# models/ensemble_gpu_hedge_fund.py
"""
Enhanced GPU-Accelerated Ensemble Model for Hedge Fund ML Trading
Integrates with existing codebase and enhanced features
No RAPIDS dependency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings
import gc
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler, random_split
import torch.nn.functional as F
import torch.cuda.amp as amp  # For mixed precision training

# XGBoost with GPU
import xgboost as xgb

# LightGBM with GPU
import lightgbm as lgb

# CatBoost with GPU
try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Import enhanced features
from models.enhanced_features import EnhancedFeatureEngineer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration for training"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True
    batch_size: int = 512  # Larger batch size for hedge fund scale
    gradient_accumulation_steps: int = 4
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Memory management
    empty_cache_freq: int = 10  # Empty cache every N batches
    gradient_checkpointing: bool = True

    # Training parameters
    sequence_length: int = 30
    prediction_horizon: int = 5  # 5-day forward prediction
    n_epochs: int = 100
    early_stopping_patience: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 0.01


class AttentionLSTM(nn.Module):
    """Enhanced LSTM with multi-head self-attention mechanism"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=4, n_heads=8, dropout=0.3):
        super(AttentionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers with larger hidden dimension for complex patterns
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Multi-head attention with more heads
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Deeper output network
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # LSTM encoding
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out  # Residual connection

        # Global pooling (both mean and max)
        avg_pool = torch.mean(combined, dim=1)
        max_pool, _ = torch.max(combined, dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)

        # Reshape for BatchNorm
        batch_size = pooled.shape[0]
        if batch_size > 1:
            out = self.output_net(pooled)
        else:
            # Skip BatchNorm for single sample
            out = pooled
            for layer in self.output_net:
                if not isinstance(layer, nn.BatchNorm1d):
                    out = layer(out)

        return out.squeeze()


class TransformerModel(nn.Module):
    """Transformer model for financial time series"""

    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)

        # Global pooling
        x = x.mean(dim=1)

        return self.output_projection(x).squeeze()


class CNNLSTM(nn.Module):
    """Enhanced CNN-LSTM hybrid model"""

    def __init__(self, input_dim, hidden_dim=128, num_filters=128, dropout=0.3):
        super(CNNLSTM, self).__init__()

        # Multi-scale CNN layers
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=7, padding=3)

        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters)
        self.batch_norm3 = nn.BatchNorm1d(num_filters * 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout1 = nn.Dropout(dropout)

        # LSTM layers
        self.lstm = nn.LSTM(
            num_filters * 2, hidden_dim,
            num_layers=3, batch_first=True, dropout=dropout, bidirectional=True
        )

        # Output layers
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2)

        # Multi-scale CNN feature extraction
        conv1_out = F.relu(self.batch_norm1(self.conv1(x)))
        conv2_out = F.relu(self.batch_norm2(self.conv2(x)))
        conv3_out = F.relu(self.batch_norm3(self.conv3(x)))

        # Use the highest resolution features for LSTM
        x = self.dropout1(conv3_out)

        # Prepare for LSTM (batch, length, channels)
        x = x.transpose(1, 2)

        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Take last output
        out = lstm_out[:, -1, :]

        # Output network
        out = self.output_net(out)

        return out.squeeze()


class HedgeFundGPUEnsemble:
    """GPU-accelerated ensemble model for hedge fund trading"""

    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()
        self.models = {}
        self.model_weights = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = []
        self.feature_engineer = EnhancedFeatureEngineer(use_gpu=torch.cuda.is_available())

        # Feature names storage
        self.feature_names = None

        # Initialize device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            logger.warning("GPU not available, using CPU")

    def prepare_training_data(self, train_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare training data with enhanced features for all symbols"""

        logger.info("Preparing training data with enhanced features...")

        X_train_all = []
        y_train_all = []
        symbol_info = []

        for symbol, df in train_data.items():
            if len(df) < 200:  # Need enough data for features
                logger.warning(f"Skipping {symbol}: insufficient data")
                continue

            try:
                # Create enhanced features
                features = self.feature_engineer.create_all_features(df, symbol)

                if features.empty or len(features) < 100:
                    logger.warning(f"Skipping {symbol}: insufficient features")
                    continue

                # Create target (5-day forward return > 2%)
                forward_return = df['close'].pct_change(self.config.prediction_horizon).shift(
                    -self.config.prediction_horizon)
                target = (forward_return > 0.02).astype(int)

                # Align features and target
                min_len = min(len(features), len(target))
                features = features.iloc[:min_len]
                target = target.iloc[:min_len]

                # Remove last prediction_horizon rows and any NaN
                features = features[:-self.config.prediction_horizon]
                target = target[:-self.config.prediction_horizon]

                # Remove NaN
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_idx]
                target = target[valid_idx]

                if len(features) > 50:
                    X_train_all.append(features)
                    y_train_all.append(target)
                    symbol_info.append({
                        'symbol': symbol,
                        'n_samples': len(features),
                        'positive_rate': target.mean()
                    })

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not X_train_all:
            raise ValueError("No valid training data")

        # Combine all data
        X_train = pd.concat(X_train_all, ignore_index=True)
        y_train = pd.concat(y_train_all, ignore_index=True)

        # Store feature names
        self.feature_names = X_train.columns.tolist()

        # Log statistics
        logger.info(f"Combined training data: {len(X_train)} samples from {len(symbol_info)} symbols")
        logger.info(f"Feature count: {len(self.feature_names)}")
        logger.info(f"Positive class rate: {y_train.mean():.2%}")

        # Log feature categories
        feature_categories = {
            'price': len([f for f in self.feature_names if any(x in f for x in ['price', 'return', 'close'])]),
            'volume': len([f for f in self.feature_names if 'volume' in f]),
            'technical': len([f for f in self.feature_names if any(x in f for x in ['rsi', 'macd', 'bb', 'ma'])]),
            'interaction': len(
                [f for f in self.feature_names if any(x in f for x in ['cross', 'divergence', 'signal'])]),
            'regime': len([f for f in self.feature_names if 'regime' in f]),
            'microstructure': len([f for f in self.feature_names if any(x in f for x in ['spread', 'noise', 'amihud'])])
        }
        logger.info(f"Feature categories: {feature_categories}")

        # Calculate class weights
        class_weights = self._calculate_class_weights(y_train.values)
        sample_weights = y_train.map(class_weights).values

        return X_train, y_train, {
            'class_weights': class_weights,
            'sample_weights': sample_weights,
            'symbol_info': symbol_info
        }

    def _calculate_class_weights(self, y):
        """Calculate class weights for imbalanced dataset"""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))

    def train_combined(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       sample_weights: np.ndarray = None) -> None:
        """Train all models with GPU acceleration"""

        logger.info(f"Training ensemble on {len(X_train)} samples with {X_train.shape[1]} features")

        # Scale features
        if 'robust' not in self.scalers:
            self.scalers['robust'] = RobustScaler()
            X_train_scaled = self.scalers['robust'].fit_transform(X_train)
        else:
            X_train_scaled = self.scalers['robust'].transform(X_train)

        X_val_scaled = self.scalers['robust'].transform(X_val)

        # Convert to numpy arrays
        X_train_scaled = np.nan_to_num(X_train_scaled, 0).astype(np.float32)
        X_val_scaled = np.nan_to_num(X_val_scaled, 0).astype(np.float32)
        y_train_np = y_train.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)

        # Train deep learning models
        self._train_deep_models(X_train_scaled, y_train_np, X_val_scaled, y_val_np, sample_weights)

        # Train tree-based models
        self._train_tree_models(X_train_scaled, y_train_np, X_val_scaled, y_val_np, sample_weights)

        # Calculate ensemble weights based on validation performance
        self._optimize_ensemble_weights(X_val_scaled, y_val_np)

        # Calculate feature importance
        self._calculate_feature_importance()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _train_deep_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           sample_weights: np.ndarray = None) -> None:
        """Train deep learning models with GPU acceleration"""

        logger.info("Training deep learning models on GPU...")

        # Prepare sequences
        X_train_seq, y_train_seq, weights_train_seq = self._prepare_sequences(X_train, y_train, sample_weights)
        X_val_seq, y_val_seq, _ = self._prepare_sequences(X_val, y_val)

        if len(X_train_seq) < 100:
            logger.warning("Insufficient sequences for deep learning training")
            return

        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.FloatTensor(y_train_seq)
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_seq),
            torch.FloatTensor(y_val_seq)
        )

        # Create data loaders with weighted sampling
        if weights_train_seq is not None:
            sampler = WeightedRandomSampler(
                weights=weights_train_seq,
                num_samples=len(weights_train_seq),
                replacement=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                sampler=sampler, num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                shuffle=True, num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )

        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size * 2,
            shuffle=False, num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        # Train models
        input_dim = X_train_seq.shape[-1]

        # Train Attention LSTM
        self.models['attention_lstm'] = self._train_single_deep_model(
            AttentionLSTM(input_dim, hidden_dim=256, num_layers=4, dropout=0.3),
            train_loader, val_loader, "Attention LSTM"
        )

        # Train CNN-LSTM
        self.models['cnn_lstm'] = self._train_single_deep_model(
            CNNLSTM(input_dim, hidden_dim=128, num_filters=128, dropout=0.3),
            train_loader, val_loader, "CNN-LSTM"
        )

        # Train Transformer
        self.models['transformer'] = self._train_single_deep_model(
            TransformerModel(input_dim, d_model=256, n_heads=8, n_layers=4, dropout=0.1),
            train_loader, val_loader, "Transformer"
        )

    def _train_single_deep_model(self, model: nn.Module, train_loader: DataLoader,
                                 val_loader: DataLoader, model_name: str) -> nn.Module:
        """Train a single deep learning model with mixed precision"""

        logger.info(f"Training {model_name} on GPU...")

        model = model.to(self.config.device)

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                weight_decay=self.config.weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.learning_rate * 3,
            epochs=self.config.n_epochs, steps_per_epoch=len(train_loader),
            pct_start=0.3
        )

        # Mixed precision scaler
        scaler = amp.GradScaler() if self.config.mixed_precision else None

        # Training loop
        best_val_auc = 0
        patience_counter = 0

        for epoch in range(self.config.n_epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                # Mixed precision forward pass
                if self.config.mixed_precision:
                    with amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.config.mixed_precision:
                    scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                scheduler.step()

                train_loss += loss.item() * self.config.gradient_accumulation_steps
                train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                train_targets.extend(batch_y.detach().cpu().numpy())

                # Memory management
                if batch_idx % self.config.empty_cache_freq == 0:
                    torch.cuda.empty_cache()

            # Validation phase
            model.eval()
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)

                    outputs = model(batch_X)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # Calculate metrics
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping {model_name} at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"{model_name} Epoch {epoch}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

        # Load best model
        model.load_state_dict(best_model_state)
        logger.info(f"{model_name} training completed. Best Val AUC: {best_val_auc:.4f}")

        return model

    def _train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           sample_weights: np.ndarray = None) -> None:
        """Train tree-based models with GPU acceleration"""

        logger.info("Training tree-based models...")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Train XGBoost
        logger.info("Training XGBoost with GPU...")
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
        dval = xgb.DMatrix(X_val, label=y_val)

        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'tree_method': 'gpu_hist' if self.config.device == 'cuda' else 'hist',
            'gpu_id': 0,
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 5,
            'scale_pos_weight': scale_pos_weight,
            'seed': 42
        }

        self.models['xgboost'] = xgb.train(
            xgb_params, dtrain, num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50, verbose_eval=100
        )

        # Train LightGBM
        logger.info("Training LightGBM...")
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        lgb_params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'min_data_in_leaf': 20,
            'scale_pos_weight': scale_pos_weight,
            'seed': 42,
            'verbosity': -1
        }

        self.models['lightgbm'] = lgb.train(
            lgb_params, lgb_train, num_boost_round=1000,
            valid_sets=[lgb_val], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        # Train CatBoost
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost with GPU...")
            cb_train = cb.Pool(X_train, y_train, weight=sample_weights)
            cb_val = cb.Pool(X_val, y_val)

            cb_model = cb.CatBoostClassifier(
                iterations=1000,
                learning_rate=0.02,
                depth=8,
                loss_function='Logloss',
                eval_metric='AUC',
                task_type='GPU' if self.config.device == 'cuda' else 'CPU',
                devices='0',
                scale_pos_weight=scale_pos_weight,
                random_seed=42,
                early_stopping_rounds=50,
                verbose=100
            )

            cb_model.fit(cb_train, eval_set=cb_val, use_best_model=True)
            self.models['catboost'] = cb_model

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray,
                           sample_weights: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        sequences = []
        targets = []
        weights = []

        for i in range(self.config.sequence_length, len(X)):
            sequences.append(X[i - self.config.sequence_length:i])
            targets.append(y[i])

            if sample_weights is not None:
                weights.append(sample_weights[i])

        return (np.array(sequences, dtype=np.float32),
                np.array(targets, dtype=np.float32),
                np.array(weights, dtype=np.float32) if weights else None)

    def _optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Optimize ensemble weights using validation performance"""

        val_predictions = {}
        val_scores = {}

        # Get predictions from each model
        for name, model in self.models.items():
            if name in ['xgboost']:
                dval = xgb.DMatrix(X_val)
                pred = model.predict(dval)
            elif name in ['lightgbm']:
                pred = model.predict(X_val, num_iteration=model.best_iteration)
            elif name in ['catboost'] and CATBOOST_AVAILABLE:
                pred = model.predict_proba(X_val)[:, 1]
            elif name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                pred = self._predict_deep_model(model, X_val)
            else:
                continue

            val_predictions[name] = pred
            val_scores[name] = roc_auc_score(y_val, pred)
            logger.info(f"{name} validation AUC: {val_scores[name]:.4f}")

        # Calculate weights based on performance (could use more sophisticated optimization)
        total_score = sum(val_scores.values())
        self.model_weights = {name: score / total_score for name, score in val_scores.items()}

        logger.info(f"Optimized ensemble weights: {self.model_weights}")

    def _predict_deep_model(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Get predictions from deep learning model"""
        model.eval()

        # Prepare sequences
        X_seq, _, _ = self._prepare_sequences(X, np.zeros(len(X)))

        if len(X_seq) == 0:
            return np.array([0.5])

        # Create dataset and loader
        dataset = TensorDataset(torch.FloatTensor(X_seq))
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 2, shuffle=False)

        predictions = []

        with torch.no_grad():
            for batch in loader:
                batch_X = batch[0].to(self.config.device)
                outputs = model(batch_X)
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())

        return np.array(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""

        # Align features
        if self.feature_names:
            # Add missing features
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            # Select only expected features in correct order
            X = X[self.feature_names]

        # Scale features
        X_scaled = self.scalers['robust'].transform(X.values.astype(np.float32))
        X_scaled = np.nan_to_num(X_scaled, 0)

        predictions = []

        for name, model in self.models.items():
            weight = self.model_weights.get(name, 0)
            if weight == 0:
                continue

            if name in ['xgboost']:
                dtest = xgb.DMatrix(X_scaled)
                pred = model.predict(dtest)
            elif name in ['lightgbm']:
                pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            elif name in ['catboost'] and CATBOOST_AVAILABLE:
                pred = model.predict_proba(X_scaled)[:, 1]
            elif name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                pred = self._predict_deep_model(model, X_scaled)
                # Ensure prediction length matches
                if len(pred) < len(X):
                    pred = np.pad(pred, (0, len(X) - len(pred)), constant_values=0.5)
                elif len(pred) > len(X):
                    pred = pred[-len(X):]
            else:
                continue

            predictions.append(pred * weight)

        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)

        return ensemble_pred

    def _calculate_feature_importance(self) -> None:
        """Calculate aggregated feature importance"""

        importance_dict = {}

        # XGBoost importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].get_score(importance_type='gain')
            for idx, score in xgb_importance.items():
                if idx.startswith('f'):
                    feature_idx = int(idx[1:])
                    if feature_idx < len(self.feature_names):
                        feature_name = self.feature_names[feature_idx]
                        importance_dict[feature_name] = importance_dict.get(feature_name, 0) + score

        # LightGBM importance
        if 'lightgbm' in self.models:
            lgb_importance = self.models['lightgbm'].feature_importance(importance_type='gain')
            for idx, score in enumerate(lgb_importance):
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    importance_dict[feature_name] = importance_dict.get(feature_name, 0) + score

        # CatBoost importance
        if 'catboost' in self.models and CATBOOST_AVAILABLE:
            cb_importance = self.models['catboost'].feature_importances_
            for idx, score in enumerate(cb_importance):
                if idx < len(self.feature_names):
                    feature_name = self.feature_names[idx]
                    importance_dict[feature_name] = importance_dict.get(feature_name, 0) + score

        # Normalize
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            self.feature_importance = {k: v / total_importance for k, v in importance_dict.items()}

        # Sort by importance
        self.feature_importance = dict(sorted(self.feature_importance.items(),
                                              key=lambda x: x[1], reverse=True))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance

    def validate(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Validate ensemble performance"""

        predictions = self.predict_proba(X_val)
        auc_score = roc_auc_score(y_val, predictions)

        # Calculate additional metrics
        pred_binary = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y_val, pred_binary)
        precision = precision_score(y_val, pred_binary)
        recall = recall_score(y_val, pred_binary)
        f1 = f1_score(y_val, pred_binary)

        logger.info(f"Validation Metrics:")
        logger.info(f"  AUC: {auc_score:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")

        return auc_score

    def get_individual_predictions(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Get predictions from each model individually"""

        individual_preds = []

        # Prepare data
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            X = X[self.feature_names]

        X_scaled = self.scalers['robust'].transform(X.values.astype(np.float32))
        X_scaled = np.nan_to_num(X_scaled, 0)

        for name, model in self.models.items():
            if name in ['xgboost']:
                dtest = xgb.DMatrix(X_scaled)
                pred = model.predict(dtest)
            elif name in ['lightgbm']:
                pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            elif name in ['catboost'] and CATBOOST_AVAILABLE:
                pred = model.predict_proba(X_scaled)[:, 1]
            elif name in ['attention_lstm', 'cnn_lstm', 'transformer']:
                pred = self._predict_deep_model(model, X_scaled)
                if len(pred) < len(X):
                    pred = np.pad(pred, (0, len(X) - len(pred)), constant_values=0.5)
                elif len(pred) > len(X):
                    pred = pred[-len(X):]
            else:
                continue

            individual_preds.append(pred)

        return individual_preds

    def save_models(self, path: str) -> None:
        """Save all models and configurations"""

        os.makedirs(path, exist_ok=True)

        # Save configuration
        joblib.dump({
            'model_weights': self.model_weights,
            'scalers': self.scalers,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names
        }, os.path.join(path, 'ensemble_config.pkl'))

        # Save tree models
        if 'xgboost' in self.models:
            self.models['xgboost'].save_model(os.path.join(path, 'xgboost_model.json'))

        if 'lightgbm' in self.models:
            self.models['lightgbm'].save_model(os.path.join(path, 'lightgbm_model.txt'))

        if 'catboost' in self.models and CATBOOST_AVAILABLE:
            self.models['catboost'].save_model(os.path.join(path, 'catboost_model.cbm'))

        # Save deep learning models
        for name in ['attention_lstm', 'cnn_lstm', 'transformer']:
            if name in self.models:
                torch.save(self.models[name].state_dict(),
                           os.path.join(path, f'{name}_model.pth'))

    def load_models(self, path: str) -> None:
        """Load all models and configurations"""

        # Load configuration
        config = joblib.load(os.path.join(path, 'ensemble_config.pkl'))
        self.model_weights = config['model_weights']
        self.scalers = config['scalers']
        self.feature_importance = config['feature_importance']
        self.feature_names = config.get('feature_names', [])

        # Load tree models
        if os.path.exists(os.path.join(path, 'xgboost_model.json')):
            self.models['xgboost'] = xgb.Booster()
            self.models['xgboost'].load_model(os.path.join(path, 'xgboost_model.json'))

        if os.path.exists(os.path.join(path, 'lightgbm_model.txt')):
            self.models['lightgbm'] = lgb.Booster(model_file=os.path.join(path, 'lightgbm_model.txt'))

        if CATBOOST_AVAILABLE and os.path.exists(os.path.join(path, 'catboost_model.cbm')):
            self.models['catboost'] = cb.CatBoostClassifier()
            self.models['catboost'].load_model(os.path.join(path, 'catboost_model.cbm'))

        # Load deep learning models - need to reconstruct architecture first
        n_features = len(self.feature_names) if self.feature_names else 600

        if os.path.exists(os.path.join(path, 'attention_lstm_model.pth')):
            model = AttentionLSTM(n_features, hidden_dim=256, num_layers=4, dropout=0.3)
            model.load_state_dict(torch.load(os.path.join(path, 'attention_lstm_model.pth'),
                                             map_location=self.config.device))
            model.to(self.config.device)
            model.eval()
            self.models['attention_lstm'] = model

        if os.path.exists(os.path.join(path, 'cnn_lstm_model.pth')):
            model = CNNLSTM(n_features, hidden_dim=128, num_filters=128, dropout=0.3)
            model.load_state_dict(torch.load(os.path.join(path, 'cnn_lstm_model.pth'),
                                             map_location=self.config.device))
            model.to(self.config.device)
            model.eval()
            self.models['cnn_lstm'] = model

        if os.path.exists(os.path.join(path, 'transformer_model.pth')):
            model = TransformerModel(n_features, d_model=256, n_heads=8, n_layers=4, dropout=0.1)
            model.load_state_dict(torch.load(os.path.join(path, 'transformer_model.pth'),
                                             map_location=self.config.device))
            model.to(self.config.device)
            model.eval()
            self.models['transformer'] = model