"""
t-SNE (t-Distributed Stochastic Neighbor Embedding) for Stock Data Visualization and Analysis
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TSNEClustering:
    """
    t-SNE for dimensionality reduction and clustering of stock market data
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 perplexity: float = 30.0,
                 learning_rate: float = 200.0,
                 n_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize t-SNE model
        
        Args:
            n_components: Dimension of the embedded space
            perplexity: Related to the number of nearest neighbors
            learning_rate: Learning rate for optimization
            n_iter: Maximum number of iterations
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.embedded_data = None
        self.cluster_labels = None
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for t-SNE analysis
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Feature matrix
        """
        # Calculate technical indicators
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
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
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio', 'volume_ratio_10',
            'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
            'high_low_ratio', 'close_high_ratio', 'close_low_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        return df_clean[self.feature_names].values
    
    def fit_transform(self, df: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """
        Fit t-SNE and perform clustering on embedded data
        
        Args:
            df: DataFrame with stock data
            n_clusters: Number of clusters for K-means clustering
            
        Returns:
            Results dictionary with embedded data and cluster analysis
        """
        try:
            logger.info(f"Fitting t-SNE with {self.n_components} components")
            
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit t-SNE
            self.embedded_data = self.model.fit_transform(X_scaled)
            
            # Perform clustering on embedded data
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            self.cluster_labels = kmeans.fit_predict(self.embedded_data)
            
            # Calculate clustering metrics
            silhouette_avg = silhouette_score(self.embedded_data, self.cluster_labels)
            
            # Analyze clusters in embedded space
            cluster_analysis = self._analyze_embedded_clusters(df, self.cluster_labels)
            
            self.is_fitted = True
            
            results = {
                'n_components': self.n_components,
                'perplexity': self.perplexity,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'embedded_data': self.embedded_data.tolist(),
                'cluster_labels': self.cluster_labels.tolist(),
                'silhouette_score': float(silhouette_avg),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_analysis': cluster_analysis,
                'feature_names': self.feature_names,
                'training_samples': len(X),
                'kl_divergence': float(self.model.kl_divergence_)
            }
            
            logger.info(f"t-SNE completed. Silhouette score: {silhouette_avg:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting t-SNE: {str(e)}")
            raise
    
    def _analyze_embedded_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """
        Analyze clusters in the embedded space
        
        Args:
            df: Original DataFrame
            labels: Cluster labels
            
        Returns:
            Cluster analysis dictionary
        """
        cluster_analysis = {}
        
        for cluster_id in range(len(set(labels))):
            cluster_mask = labels == cluster_id
            cluster_data = df.iloc[cluster_mask]
            
            if len(cluster_data) > 0:
                # Calculate cluster characteristics
                analysis = {
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(labels) * 100),
                    'avg_price': float(cluster_data['close'].mean()),
                    'avg_volume': float(cluster_data['volume'].mean()),
                    'avg_volatility': float(cluster_data['close'].pct_change().std()),
                    'price_trend': self._calculate_trend(cluster_data['close']),
                    'volume_trend': self._calculate_trend(cluster_data['volume'])
                }
                
                # Calculate embedded space statistics
                embedded_cluster = self.embedded_data[cluster_mask]
                analysis['embedded_stats'] = {
                    'mean_x': float(np.mean(embedded_cluster[:, 0])),
                    'mean_y': float(np.mean(embedded_cluster[:, 1])),
                    'std_x': float(np.std(embedded_cluster[:, 0])),
                    'std_y': float(np.std(embedded_cluster[:, 1])),
                    'spread': float(np.max(embedded_cluster) - np.min(embedded_cluster))
                }
                
                cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """
        Calculate trend direction for a series
        
        Args:
            series: Time series data
            
        Returns:
            Trend direction
        """
        if len(series) < 2:
            return "insufficient_data"
        
        first_half = series[:len(series)//2].mean()
        second_half = series[len(series)//2:].mean()
        
        change = (second_half - first_half) / first_half * 100
        
        if change > 5:
            return "strong_uptrend"
        elif change > 1:
            return "uptrend"
        elif change < -5:
            return "strong_downtrend"
        elif change < -1:
            return "downtrend"
        else:
            return "sideways"
    
    def transform(self, df: pd.DataFrame) -> Dict:
        """
        Transform new data using fitted t-SNE model
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Transformation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Note: t-SNE doesn't support transform on new data
            # This is a limitation of t-SNE
            # We'll return the original data with a warning
            
            results = {
                'transformed_data': X_scaled.tolist(),
                'warning': 't-SNE cannot transform new data. Use fit_transform for new datasets.',
                'feature_names': self.feature_names,
                'model_type': 't-SNE',
                'samples': len(X)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def get_visualization_data(self) -> Dict:
        """
        Get data formatted for visualization
        
        Returns:
            Visualization data dictionary
        """
        if not self.is_fitted:
            return {"error": "Model must be fitted first"}
        
        return {
            'embedded_data': self.embedded_data.tolist(),
            'cluster_labels': self.cluster_labels.tolist(),
            'n_components': self.n_components,
            'feature_names': self.feature_names
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the fitted model
        
        Returns:
            Model information dictionary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "model_type": "t-SNE",
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "kl_divergence": float(self.model.kl_divergence_) if hasattr(self.model, 'kl_divergence_') else None,
            "is_fitted": self.is_fitted
        }
