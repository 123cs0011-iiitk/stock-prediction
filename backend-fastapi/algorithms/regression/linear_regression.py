"""
Linear Regression for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LinearRegression:
    """
    Linear Regression model for stock price prediction
    """
    
    def __init__(self, 
                 fit_intercept: bool = True,
                 normalize: bool = False,
                 random_state: int = 42):
        """
        Initialize Linear Regression model
        
        Args:
            fit_intercept: Whether to calculate the intercept
            normalize: Whether to normalize features
            random_state: Random state for reproducibility
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.random_state = random_state
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            normalize=normalize
        )
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for regression
        
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
        
        # Moving average ratios
        df['ma_5_ratio'] = df['close'] / df['ma_5']
        df['ma_10_ratio'] = df['close'] / df['ma_10']
        df['ma_20_ratio'] = df['close'] / df['ma_20']
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        
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
        
        # High-Low features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_2', 'price_change_5', 'price_change_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_10',
            'volume_change', 'volume_ratio',
            'rsi', 'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names + [target_column]].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean[target_column].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Train the Linear Regression model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Linear Regression model")
            
            # Prepare features
            X, y = self.prepare_features(df, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate coefficients
            coefficients = dict(zip(self.feature_names, self.model.coef_))
            
            self.is_trained = True
            
            results = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2_score': float(r2),
                'coefficients': coefficients,
                'intercept': float(self.model.intercept_),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names
            }
            
            logger.info(f"Linear Regression training completed. R²: {r2:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Linear Regression: {str(e)}")
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
            
            # Make predictions
            predictions = self.model.predict(X)
            
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
                'model_type': 'Linear Regression',
                'coefficients': dict(zip(self.feature_names, self.model.coef_)),
                'intercept': float(self.model.intercept_)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_coefficients(self) -> Dict:
        """
        Get model coefficients
        
        Returns:
            Dictionary of coefficients
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return {
            'coefficients': dict(zip(self.feature_names, self.model.coef_)),
            'intercept': float(self.model.intercept_)
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Model information dictionary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "model_type": "Linear Regression",
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
