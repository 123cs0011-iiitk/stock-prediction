"""
Naive Bayes Classifier for Stock Price Direction Prediction
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier for predicting stock price direction (up/down)
    """
    
    def __init__(self, 
                 var_smoothing: float = 1e-9,
                 random_state: int = 42):
        """
        Initialize Naive Bayes Classifier
        
        Args:
            var_smoothing: Portion of the largest variance of all features added to variances for calculation stability
            random_state: Random state for reproducibility
        """
        self.var_smoothing = var_smoothing
        self.random_state = random_state
        
        self.model = GaussianNB(var_smoothing=var_smoothing)
        self.scaler = StandardScaler()
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
        
        # Target: 1 if next day price is higher, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
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
        df_clean = df[self.feature_names + ['target']].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the Naive Bayes Classifier
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Naive Bayes Classifier")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Scale features (optional for Naive Bayes, but can help)
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate class probabilities
            class_probabilities = self.model.class_prior_
            
            self.is_trained = True
            
            results = {
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'class_probabilities': class_probabilities.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names,
                'var_smoothing': self.var_smoothing,
                'n_features': len(self.feature_names)
            }
            
            logger.info(f"Naive Bayes training completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Naive Bayes: {str(e)}")
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
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            latest_probability = probabilities[-1]
            
            # Determine confidence and direction
            confidence = max(latest_probability)
            direction = "up" if latest_prediction == 1 else "down"
            
            # Calculate log probabilities for additional insight
            log_probabilities = self.model.predict_log_proba(X_scaled)
            latest_log_probs = log_probabilities[-1]
            
            results = {
                'prediction': direction,
                'confidence': float(confidence),
                'probability_up': float(latest_probability[1]),
                'probability_down': float(latest_probability[0]),
                'log_probability_up': float(latest_log_probs[1]),
                'log_probability_down': float(latest_log_probs[0]),
                'model_type': 'Naive Bayes',
                'var_smoothing': self.var_smoothing,
                'feature_count': len(self.feature_names)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from Naive Bayes model (based on variance)
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # For Gaussian Naive Bayes, we can use the feature variances as importance
        feature_variances = self.model.sigma_[0]  # Variances for class 0
        feature_importance = dict(zip(self.feature_names, feature_variances))
        
        return {
            'feature_importance': feature_importance,
            'interpretation': 'Lower variance indicates more important features for classification'
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
            "model_type": "Naive Bayes Classifier",
            "var_smoothing": self.var_smoothing,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "class_prior": self.model.class_prior_.tolist(),
            "n_classes": len(self.model.classes_),
            "classes": self.model.classes_.tolist(),
            "is_trained": self.is_trained
        }
