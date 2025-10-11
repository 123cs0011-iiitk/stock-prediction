"""
Artificial Neural Network (ANN) for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging

# Try to import TensorFlow/Keras, fall back to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    from sklearn.neural_network import MLPRegressor
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ANNRegressor:
    """
    Artificial Neural Network for stock price prediction
    """
    
    def __init__(self, 
                 hidden_layers: List[int] = [64, 32, 16],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize ANN Regressor
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            random_state: Random state for reproducibility
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(random_state)
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model = None
        self.training_history = None
        
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for ANN training
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Tuple of (features, target)
        """
        # Calculate technical indicators
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Moving average ratios
        df['ma_5_ratio'] = df['close'] / df['ma_5']
        df['ma_10_ratio'] = df['close'] / df['ma_10']
        df['ma_20_ratio'] = df['close'] / df['ma_20']
        df['ma_50_ratio'] = df['close'] / df['ma_50']
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # High-Low features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_2', 'price_change_5', 'price_change_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio', 'volume_ratio_10',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names + [target_column]].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean[target_column].values
        
        return X, y
    
    def _create_tensorflow_model(self, input_dim: int) -> Sequential:
        """
        Create TensorFlow/Keras ANN model
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Keras Sequential model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation=self.activation, input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(layer_size, activation=self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _create_sklearn_model(self) -> MLPRegressor:
        """
        Create scikit-learn MLPRegressor model
        
        Returns:
            MLPRegressor model
        """
        return MLPRegressor(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=self.batch_size,
            validation_fraction=self.validation_split,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10
        )
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Train the ANN model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training ANN model")
            
            # Prepare features
            X, y = self.prepare_features(df, target_column)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state
            )
            
            # Create and train model
            if TENSORFLOW_AVAILABLE:
                self.model = self._create_tensorflow_model(X_train.shape[1])
                
                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
                )
                
                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                self.training_history = {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history['mae'],
                    'val_mae': history.history['val_mae']
                }
                
                # Make predictions
                y_pred = self.model.predict(X_test).flatten()
                
            else:
                # Use scikit-learn MLPRegressor
                self.model = self._create_sklearn_model()
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                self.training_history = {
                    'loss': self.model.loss_curve_,
                    'val_loss': getattr(self.model, 'validation_scores_', []),
                    'mae': [],
                    'val_mae': []
                }
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            self.is_trained = True
            
            results = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'mape': float(mape),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names,
                'hidden_layers': self.hidden_layers,
                'training_history': self.training_history,
                'framework': 'TensorFlow' if TENSORFLOW_AVAILABLE else 'scikit-learn'
            }
            
            logger.info(f"ANN training completed. R²: {r2:.4f}, RMSE: {rmse:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training ANN: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Make predictions using trained model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            X, _ = self.prepare_features(df, target_column)
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            if TENSORFLOW_AVAILABLE:
                predictions = self.model.predict(X_scaled).flatten()
            else:
                predictions = self.model.predict(X_scaled)
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            current_price = df[target_column].iloc[-1]
            
            # Calculate change
            price_change = latest_prediction - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Determine direction
            direction = "up" if price_change > 0 else "down"
            
            results = {
                'predicted_price': float(latest_prediction),
                'current_price': float(current_price),
                'price_change': float(price_change),
                'price_change_percent': float(price_change_percent),
                'direction': direction,
                'model_type': 'Artificial Neural Network',
                'hidden_layers': self.hidden_layers,
                'framework': 'TensorFlow' if TENSORFLOW_AVAILABLE else 'scikit-learn'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Model information dictionary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "Artificial Neural Network",
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "framework": 'TensorFlow' if TENSORFLOW_AVAILABLE else 'scikit-learn',
            "is_trained": self.is_trained
        }
        
        if TENSORFLOW_AVAILABLE and self.model:
            info["total_params"] = self.model.count_params()
        
        return info
