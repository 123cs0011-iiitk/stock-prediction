"""
K-Means Clustering for Stock Market Segmentation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KMeansClustering:
    """
    K-Means clustering for stock market analysis and segmentation
    """
    
    def __init__(self, 
                 n_clusters: int = 3,
                 init: str = 'k-means++',
                 max_iter: int = 300,
                 random_state: int = 42):
        """
        Initialize K-Means clustering
        
        Args:
            n_clusters: Number of clusters
            init: Initialization method
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.cluster_centers_ = None
        self.labels_ = None
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for clustering
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Feature matrix
        """
        # Calculate technical indicators
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        
        # Volatility features
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_ratio'] = df['close'] / df['ma_20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # High-Low features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_high_ratio'] = df['close'] / df['high']
        
        # Select features
        self.feature_names = [
            'price_change', 'price_change_5', 'price_change_10',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_change', 'volume_ratio',
            'ma_ratio', 'rsi', 'high_low_ratio', 'close_high_ratio'
        ]
        
        # Remove rows with NaN values
        df_clean = df[self.feature_names].dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature preparation")
        
        return df_clean[self.feature_names].values
    
    def find_optimal_clusters(self, df: pd.DataFrame, max_clusters: int = 10) -> Dict:
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            df: DataFrame with stock data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with optimal cluster analysis
        """
        try:
            logger.info("Finding optimal number of clusters")
            
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Test different numbers of clusters
            cluster_range = range(2, min(max_clusters + 1, len(X) // 2))
            inertias = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                labels = kmeans.fit_predict(X_scaled)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
                davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
            
            # Find optimal k (highest silhouette score)
            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            
            results = {
                'cluster_range': list(cluster_range),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'optimal_clusters': optimal_k,
                'best_silhouette_score': max(silhouette_scores),
                'recommendation': f"Optimal number of clusters: {optimal_k}"
            }
            
            logger.info(f"Optimal clusters found: {optimal_k}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            raise
    
    def fit(self, df: pd.DataFrame) -> Dict:
        """
        Fit K-Means clustering to the data
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Fitting results dictionary
        """
        try:
            logger.info(f"Fitting K-Means clustering with {self.n_clusters} clusters")
            
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit the model
            self.model.fit(X_scaled)
            
            # Get cluster labels and centers
            self.labels_ = self.model.labels_
            self.cluster_centers_ = self.model.cluster_centers_
            
            # Calculate metrics
            silhouette_avg = silhouette_score(X_scaled, self.labels_)
            calinski_score = calinski_harabasz_score(X_scaled, self.labels_)
            davies_bouldin = davies_bouldin_score(X_scaled, self.labels_)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, self.labels_)
            
            self.is_fitted = True
            
            results = {
                'n_clusters': self.n_clusters,
                'silhouette_score': float(silhouette_avg),
                'calinski_harabasz_score': float(calinski_score),
                'davies_bouldin_score': float(davies_bouldin),
                'inertia': float(self.model.inertia_),
                'cluster_centers': self.cluster_centers_.tolist(),
                'cluster_analysis': cluster_analysis,
                'feature_names': self.feature_names,
                'training_samples': len(X)
            }
            
            logger.info(f"K-Means clustering completed. Silhouette score: {silhouette_avg:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting K-Means: {str(e)}")
            raise
    
    def _analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """
        Analyze characteristics of each cluster
        
        Args:
            df: Original DataFrame
            labels: Cluster labels
            
        Returns:
            Cluster analysis dictionary
        """
        # Prepare features for analysis
        df_features = df.copy()
        df_features['cluster'] = labels
        
        cluster_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df_features[df_features['cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                analysis = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df_features) * 100,
                    'avg_price': float(cluster_data['close'].mean()),
                    'avg_volume': float(cluster_data['volume'].mean()),
                    'avg_volatility': float(cluster_data['close'].pct_change().std()),
                    'price_range': {
                        'min': float(cluster_data['close'].min()),
                        'max': float(cluster_data['close'].max())
                    },
                    'volume_range': {
                        'min': float(cluster_data['volume'].min()),
                        'max': float(cluster_data['volume'].max())
                    }
                }
                
                # Calculate average price change
                price_changes = cluster_data['close'].pct_change().dropna()
                if len(price_changes) > 0:
                    analysis['avg_price_change'] = float(price_changes.mean())
                    analysis['positive_change_ratio'] = float((price_changes > 0).mean())
                
                cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict cluster assignments for new data
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            Prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Prepare features
            X = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Predict clusters
            cluster_labels = self.model.predict(X_scaled)
            
            # Calculate distances to cluster centers
            distances = self.model.transform(X_scaled)
            min_distances = np.min(distances, axis=1)
            
            # Get latest prediction
            latest_cluster = cluster_labels[-1]
            latest_distance = min_distances[-1]
            
            # Determine cluster characteristics
            cluster_characteristics = self._get_cluster_characteristics(latest_cluster)
            
            results = {
                'predicted_cluster': int(latest_cluster),
                'cluster_distance': float(latest_distance),
                'cluster_characteristics': cluster_characteristics,
                'all_predictions': cluster_labels.tolist(),
                'model_type': 'K-Means Clustering',
                'n_clusters': self.n_clusters
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _get_cluster_characteristics(self, cluster_id: int) -> Dict:
        """
        Get characteristics of a specific cluster
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            Cluster characteristics
        """
        if self.cluster_centers_ is None:
            return {}
        
        cluster_center = self.cluster_centers_[cluster_id]
        
        characteristics = {
            'cluster_id': int(cluster_id),
            'center_values': dict(zip(self.feature_names, cluster_center.tolist()))
        }
        
        # Add interpretation based on center values
        interpretations = []
        
        for feature, value in characteristics['center_values'].items():
            if 'price_change' in feature:
                if value > 0.01:
                    interpretations.append(f"High positive {feature}")
                elif value < -0.01:
                    interpretations.append(f"High negative {feature}")
                else:
                    interpretations.append(f"Stable {feature}")
            elif 'volatility' in feature:
                if value > 0.02:
                    interpretations.append("High volatility")
                else:
                    interpretations.append("Low volatility")
            elif 'volume' in feature:
                if value > 0.5:
                    interpretations.append("High volume activity")
                else:
                    interpretations.append("Low volume activity")
        
        characteristics['interpretation'] = interpretations
        
        return characteristics
    
    def get_model_info(self) -> Dict:
        """
        Get information about the fitted model
        
        Returns:
            Model information dictionary
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "model_type": "K-Means Clustering",
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
