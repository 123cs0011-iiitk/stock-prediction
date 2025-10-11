"""
Support Vector Machine (SVM) Classifier for Stock Price Direction Prediction
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SVMClassifier:
    """
    Support Vector Machine Classifier for predicting stock price direction (up/down)
    """
    
    def __init__(self, 
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 degree: int = 3,
                 probability: bool = True,
                 random_state: int = 42):
        """
        Initialize SVM Classifier
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            degree: Degree for polynomial kernel
            probability: Whether to enable probability estimates
            random_state: Random state for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=probability,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.support_vectors_ = None
        self.n_support_ = None
        
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
        
        # Target: 1 if next day price is higher, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_2', 'price_change_5', 'price_change_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names + ['target']].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean['target'].values
        
        return X, y
    
    def optimize_hyperparameters(self, df: pd.DataFrame, cv_folds: int = 3) -> Dict:
        """
        Optimize hyperparameters using GridSearchCV
        
        Args:
            df: DataFrame with stock data
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with optimization results
        """
        try:
            logger.info("Optimizing SVM hyperparameters")
            
            # Prepare features
            X, y = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Define parameter grid
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            
            # Create SVM model
            svm_model = SVC(probability=True, random_state=self.random_state)
            
            # Perform grid search
            grid_search = GridSearchCV(
                svm_model, 
                param_grid, 
                cv=cv_folds, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_scaled, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            self.C = self.model.C
            self.gamma = self.model.gamma
            self.kernel = self.model.kernel
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
            logger.info(f"SVM hyperparameter optimization completed. Best score: {grid_search.best_score_:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error optimizing SVM hyperparameters: {str(e)}")
            raise
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the SVM Classifier
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training SVM Classifier")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Scale features (important for SVM)
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Store model attributes
            self.support_vectors_ = self.model.support_vectors_
            self.n_support_ = self.model.n_support_
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if self.probability else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate additional SVM-specific metrics
            n_support_vectors = len(self.support_vectors_)
            support_vector_ratio = n_support_vectors / len(X_train)
            
            results = {
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names,
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'degree': self.degree,
                'n_support_vectors': n_support_vectors,
                'support_vector_ratio': float(support_vector_ratio),
                'n_support_per_class': self.n_support_.tolist() if self.n_support_ is not None else None
            }
            
            self.is_trained = True
            
            logger.info(f"SVM training completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training SVM: {str(e)}")
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
            probabilities = self.model.predict_proba(X_scaled) if self.probability else None
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            
            # Determine confidence and direction
            direction = "up" if latest_prediction == 1 else "down"
            
            if probabilities is not None:
                latest_probability = probabilities[-1]
                confidence = max(latest_probability)
                probability_up = latest_probability[1]
                probability_down = latest_probability[0]
            else:
                confidence = 1.0  # No probability estimates available
                probability_up = 1.0 if latest_prediction == 1 else 0.0
                probability_down = 1.0 if latest_prediction == 0 else 0.0
            
            # Calculate distance to decision boundary
            decision_function = self.model.decision_function(X_scaled[-1:])
            distance_to_boundary = abs(decision_function[0])
            
            results = {
                'prediction': direction,
                'confidence': float(confidence),
                'probability_up': float(probability_up),
                'probability_down': float(probability_down),
                'decision_function': float(decision_function[0]),
                'distance_to_boundary': float(distance_to_boundary),
                'model_type': 'Support Vector Machine',
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'support_vectors_count': len(self.support_vectors_) if self.support_vectors_ is not None else 0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def get_support_vectors_info(self) -> Dict:
        """
        Get information about support vectors
        
        Returns:
            Dictionary with support vector information
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.support_vectors_ is None:
            return {"error": "No support vectors available"}
        
        return {
            'n_support_vectors': len(self.support_vectors_),
            'support_vector_ratio': len(self.support_vectors_) / (self.support_vectors_.shape[0] + len(self.support_vectors_)),
            'n_support_per_class': self.n_support_.tolist() if self.n_support_ is not None else None,
            'feature_count': self.support_vectors_.shape[1],
            'support_vectors_shape': self.support_vectors_.shape
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Model information dictionary
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        
        info = {
            "model_type": "Support Vector Machine",
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "degree": self.degree,
            "probability": self.probability,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
        
        if self.support_vectors_ is not None:
            info["n_support_vectors"] = len(self.support_vectors_)
            info["n_support_per_class"] = self.n_support_.tolist() if self.n_support_ is not None else None
        
        return info
