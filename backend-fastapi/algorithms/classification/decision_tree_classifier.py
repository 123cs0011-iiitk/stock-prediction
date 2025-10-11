"""
Decision Tree Classifier for Stock Price Direction Prediction
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier for predicting stock price direction (up/down)
    """
    
    def __init__(self, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Initialize Decision Tree Classifier
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            random_state: Random state for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.feature_importance_ = None
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
        
        # RSI approximation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Target: 1 if next day price is higher, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Select features
        feature_columns = [
            'price_change', 'price_change_2', 'price_change_5',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_10',
            'volume_change', 'volume_ratio',
            'rsi'
        ]
        
        # Remove rows with NaN values
        df_clean = df[feature_columns + ['target']].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[feature_columns].values
        y = df_clean['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the Decision Tree Classifier
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Decision Tree Classifier")
            
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
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Store feature importance
            self.feature_importance_ = dict(zip(
                ['price_change', 'price_change_2', 'price_change_5',
                 'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
                 'volatility_5', 'volatility_10',
                 'volume_change', 'volume_ratio', 'rsi'],
                self.model.feature_importances_
            ))
            
            self.is_trained = True
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': self.feature_importance_,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_depth': self.model.get_depth(),
                'model_leaves': self.model.get_n_leaves()
            }
            
            logger.info(f"Decision Tree training completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training Decision Tree: {str(e)}")
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
            
            # Determine confidence
            confidence = max(latest_probability)
            direction = "up" if latest_prediction == 1 else "down"
            
            results = {
                'prediction': direction,
                'confidence': float(confidence),
                'probability_up': float(latest_probability[1]),
                'probability_down': float(latest_probability[0]),
                'model_type': 'Decision Tree',
                'feature_importance': self.feature_importance_
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance from trained model
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        return self.feature_importance_
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Model information dictionary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "model_type": "Decision Tree Classifier",
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "actual_depth": self.model.get_depth(),
            "actual_leaves": self.model.get_n_leaves(),
            "feature_count": len(self.feature_importance_),
            "is_trained": self.is_trained
        }
