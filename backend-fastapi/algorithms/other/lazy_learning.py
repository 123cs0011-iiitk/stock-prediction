"""
Lazy Learning Implementation for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LazyLearning:
    """
    Lazy Learning implementation using K-Nearest Neighbors for stock price prediction
    Lazy learning algorithms defer computation until prediction time
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 weights: str = 'distance',
                 algorithm: str = 'auto',
                 leaf_size: int = 30,
                 p: int = 2,
                 random_state: int = 42):
        """
        Initialize Lazy Learning model
        
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
        
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_data = None
        self.training_targets = None
        
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for lazy learning
        
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
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        X = df_clean[self.feature_names].values
        y = df_clean[target_column].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Train the Lazy Learning model (stores training data for lazy evaluation)
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training Lazy Learning model")
            
            # Prepare features
            X, y = self.prepare_features(df, target_column)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Store training data (lazy learning characteristic)
            self.training_data = X_scaled
            self.training_targets = y
            
            # Train model (just stores the data)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
            results = {
                'training_samples': len(X_scaled),
                'feature_names': self.feature_names,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'algorithm': self.algorithm,
                'feature_count': len(self.feature_names),
                'lazy_learning': True,
                'message': 'Training data stored for lazy evaluation'
            }
            
            logger.info(f"Lazy Learning model prepared with {len(X_scaled)} training samples")
            return results
            
        except Exception as e:
            logger.error(f"Error training Lazy Learning: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Make predictions using lazy learning (computation happens here)
        
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
            
            # Make predictions (lazy computation happens here)
            predictions = self.model.predict(X_scaled)
            
            # Get distances to neighbors for the latest prediction
            distances, indices = self.model.kneighbors(X_scaled[-1:])
            latest_distances = distances[0]
            latest_indices = indices[0]
            
            # Get the latest prediction
            latest_prediction = predictions[-1]
            current_price = df[target_column].iloc[-1]
            
            # Calculate change
            price_change = latest_prediction - current_price
            price_change_percent = (price_change / current_price) * 100
            
            # Determine direction
            direction = "up" if price_change > 0 else "down"
            
            # Analyze neighbor characteristics
            neighbor_targets = self.training_targets[latest_indices]
            neighbor_distances = latest_distances
            
            # Calculate confidence based on neighbor agreement
            neighbor_changes = neighbor_targets - current_price
            positive_neighbors = np.sum(neighbor_changes > 0)
            confidence = positive_neighbors / len(neighbor_targets) if direction == "up" else (len(neighbor_targets) - positive_neighbors) / len(neighbor_targets)
            
            # Weighted prediction based on distances
            if self.weights == 'distance':
                weights = 1 / (neighbor_distances + 1e-8)  # Avoid division by zero
                weighted_prediction = np.sum(neighbor_targets * weights) / np.sum(weights)
                weighted_change = weighted_prediction - current_price
                weighted_change_percent = (weighted_change / current_price) * 100
            else:
                weighted_prediction = latest_prediction
                weighted_change = price_change
                weighted_change_percent = price_change_percent
            
            results = {
                'predicted_price': float(latest_prediction),
                'weighted_predicted_price': float(weighted_prediction),
                'current_price': float(current_price),
                'price_change': float(price_change),
                'price_change_percent': float(price_change_percent),
                'weighted_price_change': float(weighted_change),
                'weighted_price_change_percent': float(weighted_change_percent),
                'direction': direction,
                'confidence': float(confidence),
                'neighbor_analysis': {
                    'n_neighbors': len(latest_indices),
                    'average_distance': float(np.mean(latest_distances)),
                    'max_distance': float(np.max(latest_distances)),
                    'min_distance': float(np.min(latest_distances)),
                    'neighbor_targets': neighbor_targets.tolist(),
                    'neighbor_distances': latest_distances.tolist(),
                    'positive_neighbors': int(positive_neighbors),
                    'negative_neighbors': int(len(neighbor_targets) - positive_neighbors)
                },
                'model_type': 'Lazy Learning (K-NN)',
                'weights': self.weights,
                'algorithm': self.algorithm,
                'lazy_computation': True
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def evaluate_neighbors(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Evaluate the quality of neighbors for predictions
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Neighbor evaluation results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        try:
            # Prepare features
            X, _ = self.prepare_features(df, target_column)
            X_scaled = self.scaler.transform(X)
            
            # Get neighbors for all samples
            distances, indices = self.model.kneighbors(X_scaled)
            
            # Analyze neighbor quality
            neighbor_analysis = []
            
            for i in range(len(X_scaled)):
                neighbor_targets = self.training_targets[indices[i]]
                neighbor_distances = distances[i]
                
                # Calculate neighbor similarity
                target_std = np.std(neighbor_targets)
                distance_std = np.std(neighbor_distances)
                
                neighbor_analysis.append({
                    'sample_index': i,
                    'neighbor_targets': neighbor_targets.tolist(),
                    'neighbor_distances': neighbor_distances.tolist(),
                    'target_variance': float(target_std),
                    'distance_variance': float(distance_std),
                    'neighbor_consistency': float(1 - (target_std / np.mean(neighbor_targets))) if np.mean(neighbor_targets) != 0 else 0
                })
            
            # Overall statistics
            avg_target_variance = np.mean([na['target_variance'] for na in neighbor_analysis])
            avg_distance_variance = np.mean([na['distance_variance'] for na in neighbor_analysis])
            avg_consistency = np.mean([na['neighbor_consistency'] for na in neighbor_analysis])
            
            results = {
                'neighbor_analysis': neighbor_analysis,
                'overall_statistics': {
                    'average_target_variance': float(avg_target_variance),
                    'average_distance_variance': float(avg_distance_variance),
                    'average_consistency': float(avg_consistency),
                    'total_samples': len(neighbor_analysis)
                },
                'interpretation': {
                    'high_consistency': avg_consistency > 0.8,
                    'low_target_variance': avg_target_variance < np.std(self.training_targets),
                    'good_neighbor_quality': avg_consistency > 0.7 and avg_target_variance < np.std(self.training_targets)
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating neighbors: {str(e)}")
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
            "model_type": "Lazy Learning (K-Nearest Neighbors)",
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "training_samples": len(self.training_data) if self.training_data is not None else 0,
            "lazy_learning": True,
            "is_trained": self.is_trained
        }
