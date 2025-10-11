"""
Logistic Regression for Stock Price Direction Prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LogisticRegression:
    """
    Logistic Regression model for stock price direction prediction (up/down)
    """
    
    def __init__(self, 
                 penalty: str = 'l2',
                 C: float = 1.0,
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize Logistic Regression model
        
        Args:
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse of regularization strength
            solver: Algorithm to use ('lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga')
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for classification
        
        Args:
            df: DataFrame with stock data
            
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
        
        # High-Low features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        df['close_low_ratio'] = df['close'] / df['low']
        
        # Target: 1 if next day price is higher, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_2', 'price_change_5', 'price_change_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio', 'volume_ratio_10',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names + ['target']].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the Logistic Regression model
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Logistic Regression model")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate coefficients
            coefficients = dict(zip(self.feature_names, self.model.coef_[0]))
            
            self.is_trained = True
            
            results = {
                'accuracy': float(accuracy),
                'auc_score': float(auc_score),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'coefficients': coefficients,
                'intercept': float(self.model.intercept_[0]),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names
            }
            
            logger.info(f"Logistic Regression training completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Logistic Regression: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make predictions using trained model
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            X, _ = self.prepare_features(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            latest_probability = probabilities[-1]
            
            # Determine confidence and direction
            confidence = max(latest_probability)
            direction = "up" if latest_prediction == 1 else "down"
            
            results = {
                'prediction': direction,
                'confidence': float(confidence),
                'probability_up': float(latest_probability[1]),
                'probability_down': float(latest_probability[0]),
                'model_type': 'Logistic Regression',
                'coefficients': dict(zip(self.feature_names, self.model.coef_[0])),
                'intercept': float(self.model.intercept_[0])
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
            'coefficients': dict(zip(self.feature_names, self.model.coef_[0])),
            'intercept': float(self.model.intercept_[0])
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
            "model_type": "Logistic Regression",
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
