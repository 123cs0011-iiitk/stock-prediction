"""
K-Nearest Neighbors (K-NN) Classifier for Stock Price Direction Prediction
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier for predicting stock price direction (up/down)
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'uniform',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 p: int = 2,
                 random_state: int = 42):
        """
        Initialize K-NN Classifier
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function used in prediction ('uniform', 'distance')
            algorithm: Algorithm used to compute nearest neighbors
            leaf_size: Leaf size passed to BallTree or KDTree
            p: Power parameter for Minkowski metric
            random_state: Random state for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.random_state = random_state
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )
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
    
    def find_optimal_k(self, df: pd.DataFrame, max_k: int = 20) -> Dict:
        """
        Find optimal number of neighbors using cross-validation
        
        Args:
            df: DataFrame with stock data
            max_k: Maximum k to test
            
        Returns:
            Dictionary with optimal k analysis
        """
        try:
            logger.info("Finding optimal number of neighbors")
            
            # Prepare features
            X, y = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Test different values of k
            k_range = range(3, min(max_k + 1, len(X) // 2))
            cv_scores = []
            
            for k in k_range:
                knn = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=self.weights,
                    algorithm=self.algorithm,
                    leaf_size=self.leaf_size,
                    p=self.p
                )
                scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())
            
            # Find optimal k
            optimal_k = k_range[np.argmax(cv_scores)]
            best_score = max(cv_scores)
            
            results = {
                'k_range': list(k_range),
                'cv_scores': cv_scores,
                'optimal_k': optimal_k,
                'best_score': best_score,
                'recommendation': f"Optimal k: {optimal_k} with CV score: {best_score:.4f}"
            }
            
            logger.info(f"Optimal k found: {optimal_k}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding optimal k: {str(e)}")
            raise
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train the K-NN Classifier
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info(f"Training K-NN Classifier with {self.n_neighbors} neighbors")
            
            # Prepare features
            X, y = self.prepare_features(df)
            
            # Scale features (important for K-NN)
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
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
            
            self.is_trained = True
            
            results = {
                'accuracy': float(accuracy),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': self.feature_names,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm
            }
            
            logger.info(f"K-NN training completed. Accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training K-NN: {str(e)}")
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
            
            # Get distances to neighbors for the latest prediction
            distances, indices = self.model.kneighbors(X_scaled[-1:])
            latest_distances = distances[0]
            latest_indices = indices[0]
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            latest_probability = probabilities[-1]
            
            # Determine confidence and direction
            confidence = max(latest_probability)
            direction = "up" if latest_prediction == 1 else "down"
            
            # Analyze neighbor distances
            avg_distance = np.mean(latest_distances)
            max_distance = np.max(latest_distances)
            min_distance = np.min(latest_distances)
            
            results = {
                'prediction': direction,
                'confidence': float(confidence),
                'probability_up': float(latest_probability[1]),
                'probability_down': float(latest_probability[0]),
                'neighbor_distances': {
                    'average': float(avg_distance),
                    'maximum': float(max_distance),
                    'minimum': float(min_distance),
                    'all_distances': latest_distances.tolist()
                },
                'n_neighbors': self.n_neighbors,
                'model_type': 'K-Nearest Neighbors',
                'weights': self.weights,
                'algorithm': self.algorithm
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
        
        return {
            "model_type": "K-Nearest Neighbors Classifier",
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_trained": self.is_trained
        }
