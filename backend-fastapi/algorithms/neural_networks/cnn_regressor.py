"""
Convolutional Neural Network (CNN) for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging

# Try to import TensorFlow/Keras, fall back to basic implementation if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class CNNRegressor:
    """
    Convolutional Neural Network for stock price prediction using time series data
    """
    
    def __init__(self, 
                 sequence_length: int = 30,
                 filters: int = 64,
                 kernel_size: int = 3,
                 hidden_layers: List[int] = [32, 16],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize CNN Regressor
        
        Args:
            sequence_length: Length of time series sequences
            filters: Number of filters in conv layers
            kernel_size: Size of convolution kernel
            hidden_layers: List of hidden layer sizes for dense layers
            activation: Activation function
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            random_state: Random state for reproducibility
        """
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_size = kernel_size
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
        
    def prepare_sequences(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time series sequences for CNN training
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Tuple of (sequences, targets)
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
        
        # Moving average ratios
        df['ma_5_ratio'] = df['close'] / df['ma_5']
        df['ma_10_ratio'] = df['close'] / df['ma_10']
        df['ma_20_ratio'] = df['close'] / df['ma_20']
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
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
        
        # High-Low features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_2', 'price_change_5', 'price_change_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names + [target_column]].dropna()
        
        if len(df_clean) < self.sequence_length + 1:
            raise ValueError(f"Insufficient data. Need at least {self.sequence_length + 1} samples")
        
        # Prepare features and target
        features = df_clean[self.feature_names].values
        target = df_clean[target_column].values
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def _create_cnn_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Create TensorFlow/Keras CNN model
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            
        Returns:
            Keras Sequential model
        """
        model = Sequential()
        
        # Convolutional layers
        model.add(Conv1D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            activation=self.activation,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.dropout_rate))
        
        # Second conv layer
        model.add(Conv1D(
            filters=self.filters//2, 
            kernel_size=self.kernel_size, 
            activation=self.activation
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(self.dropout_rate))
        
        # Flatten and dense layers
        model.add(Flatten())
        
        for layer_size in self.hidden_layers:
            model.add(Dense(layer_size, activation=self.activation))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Train the CNN model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Training results dictionary
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for CNN implementation")
            
            logger.info("Training CNN model")
            
            # Prepare sequences
            X, y = self.prepare_sequences(df, target_column)
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state
            )
            
            # Create and train model
            self.model = self._create_cnn_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7
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
                'sequence_length': self.sequence_length,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'hidden_layers': self.hidden_layers,
                'training_history': self.training_history,
                'framework': 'TensorFlow'
            }
            
            logger.info(f"CNN training completed. R²: {r2:.4f}, RMSE: {rmse:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training CNN: {str(e)}")
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
            # Prepare sequences
            X, _ = self.prepare_sequences(df, target_column)
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Make predictions
            predictions = self.model.predict(X_scaled).flatten()
            
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
                'model_type': 'Convolutional Neural Network',
                'sequence_length': self.sequence_length,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'hidden_layers': self.hidden_layers,
                'framework': 'TensorFlow'
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
            "model_type": "Convolutional Neural Network",
            "sequence_length": self.sequence_length,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "framework": "TensorFlow",
            "is_trained": self.is_trained
        }
        
        if self.model:
            info["total_params"] = self.model.count_params()
        
        return info
