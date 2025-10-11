import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class EnsembleStockPredictor:
    """
    Ensemble Machine Learning Stock Price Predictor
    
    This class combines multiple ML algorithms:
    1. Linear Regression - Linear relationship modeling
    2. Random Forest Regressor - Non-linear pattern recognition
    3. Ensemble Method - Weighted combination of predictions
    
    The ensemble approach improves prediction accuracy by combining
    the strengths of different algorithms.
    """
    
    def __init__(self):
        # Algorithm 1: Linear Regression
        self.linear_regression_model = LinearRegression()
        
        # Algorithm 2: Random Forest Regressor (100 trees)
        self.random_forest_model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Feature scaling for Linear Regression
        self.feature_scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.model_performance = {}
        
    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based technical indicators"""
        # Price change features
        df['price_change'] = df['price'].pct_change()
        df['price_change_2d'] = df['price'].pct_change(2)
        df['price_change_5d'] = df['price'].pct_change(5)
        df['price_change_10d'] = df['price'].pct_change(10)
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple Moving Averages (SMA)"""
        # Short-term moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # Long-term moving averages
        df['sma_50'] = df['price'].rolling(window=50).mean()
        
        return df
    
    def _calculate_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-to-moving-average ratios"""
        df['price_sma5_ratio'] = df['price'] / df['sma_5']
        df['price_sma10_ratio'] = df['price'] / df['sma_10']
        df['price_sma20_ratio'] = df['price'] / df['sma_20']
        df['price_sma50_ratio'] = df['price'] / df['sma_50']
        
        return df
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features"""
        df['volatility_5d'] = df['price_change'].rolling(window=5).std()
        df['volatility_10d'] = df['price_change'].rolling(window=10).std()
        df['volatility_20d'] = df['price_change'].rolling(window=20).std()
        
        return df
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        df['momentum_5d'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_10d'] = df['price'] / df['price'].shift(10) - 1
        df['momentum_20d'] = df['price'] / df['price'].shift(20) - 1
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators"""
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + (df['volatility_20d'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['volatility_20d'] * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        return df
    
    def _calculate_volume_features(self, df: pd.DataFrame, volume: List[float]) -> pd.DataFrame:
        """Calculate volume-based features"""
        if volume is not None and len(volume) == len(df):
            df['volume'] = volume
            df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
            df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
            
            # Price-Volume relationship
            df['price_volume_correlation'] = df['price_change'].rolling(window=10).corr(df['volume'].pct_change())
        
        return df
    
    def create_features(self, prices: List[float], volume: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Create comprehensive technical features from price and volume data
        
        Args:
            prices: List of historical prices
            volume: Optional list of volume data
            
        Returns:
            DataFrame with engineered features
        """
        df = pd.DataFrame({'price': prices})
        
        # Apply feature engineering pipeline
        df = self._calculate_price_features(df)
        df = self._calculate_moving_averages(df)
        df = self._calculate_price_ratios(df)
        df = self._calculate_volatility_features(df)
        df = self._calculate_momentum_features(df)
        df = self._calculate_rsi_indicator(df)
        df = self._calculate_bollinger_bands(df)
        df = self._calculate_volume_features(df, volume)
        
        # Remove NaN values created by rolling calculations
        df = df.dropna()
        
        return df
    
    def _calculate_rsi_indicator(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index (RSI) technical indicator"""
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI-based features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target variables for training
        
        Args:
            df: DataFrame with engineered features
            target_days: Number of days ahead to predict
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        # Create target (future price)
        df['target'] = df['price'].shift(-target_days)
        
        # Remove rows with NaN targets
        df = df.dropna()
        
        # Select feature columns (exclude price and target)
        feature_cols = [col for col in df.columns if col not in ['price', 'target']]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, feature_cols
    
    def _train_linear_regression(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train Linear Regression model and return performance metrics"""
        self.linear_regression_model.fit(X_scaled, y)
        
        # Calculate performance metrics
        y_pred = self.linear_regression_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def _train_random_forest(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train Random Forest model and return performance metrics"""
        self.random_forest_model.fit(X_scaled, y)
        
        # Calculate performance metrics
        y_pred = self.random_forest_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        return {
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
    
    def train_ensemble_models(self, prices: List[float], volume: Optional[List[float]] = None) -> Tuple[bool, str]:
        """
        Train the ensemble of ML models (Linear Regression + Random Forest)
        
        Args:
            prices: Historical price data
            volume: Optional volume data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Create comprehensive features
            df = self.create_features(prices, volume)
            
            if len(df) < 100:  # Need sufficient data for robust training
                return False, f"Insufficient data for training (need at least 100 data points, got {len(df)})"
            
            # Prepare training data
            X, y, self.feature_names = self.prepare_training_data(df)
            
            if len(X) < 50:  # Need sufficient training samples
                return False, f"Insufficient training samples after feature creation (need at least 50, got {len(X)})"
            
            # Scale features for Linear Regression
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train Algorithm 1: Linear Regression
            linear_performance = self._train_linear_regression(X_scaled, y)
            
            # Train Algorithm 2: Random Forest
            random_forest_performance = self._train_random_forest(X_scaled, y)
            
            # Store performance metrics
            self.model_performance = {
                'linear_regression': linear_performance,
                'random_forest': random_forest_performance,
                'training_samples': len(X),
                'feature_count': len(self.feature_names),
                'feature_names': self.feature_names
            }
            
            self.is_trained = True
            return True, "Ensemble models trained successfully"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def _predict_with_linear_regression(self, X_scaled: np.ndarray) -> float:
        """Make prediction using Linear Regression model"""
        return self.linear_regression_model.predict(X_scaled)[0]
    
    def _predict_with_random_forest(self, X_scaled: np.ndarray) -> float:
        """Make prediction using Random Forest model"""
        return self.random_forest_model.predict(X_scaled)[0]
    
    def _calculate_ensemble_prediction(self, lr_pred: float, rf_pred: float) -> Dict[str, float]:
        """
        Calculate ensemble prediction using weighted combination
        
        Args:
            lr_pred: Linear Regression prediction
            rf_pred: Random Forest prediction
            
        Returns:
            Dictionary with ensemble prediction and confidence
        """
        # Weighted ensemble (Random Forest gets higher weight due to better performance on non-linear data)
        lr_weight = 0.3
        rf_weight = 0.7
        ensemble_pred = (lr_pred * lr_weight) + (rf_pred * rf_weight)
        
        # Calculate confidence based on model agreement
        pred_diff = abs(lr_pred - rf_pred)
        confidence = max(50, 100 - (pred_diff / ensemble_pred * 100))
        
        return {
            'ensemble_prediction': ensemble_pred,
            'confidence': confidence,
            'prediction_variance': pred_diff
        }
    
    def predict_next_day_price(self, recent_prices: List[float], recent_volume: Optional[List[float]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Predict next day's stock price using ensemble of trained models
        
        Args:
            recent_prices: Recent historical prices
            recent_volume: Optional recent volume data
            
        Returns:
            Tuple of (prediction_dict, error_message)
        """
        if not self.is_trained:
            return None, "Ensemble models not trained"
        
        try:
            # Create features from recent data
            df = self.create_features(recent_prices, recent_volume)
            
            if len(df) == 0:
                return None, "No features generated from recent data"
            
            # Get the most recent feature vector
            latest_features = df.iloc[-1:]
            feature_cols = [col for col in latest_features.columns if col not in ['price', 'target']]
            X_latest = latest_features[feature_cols].values
            
            # Scale features
            X_scaled = self.feature_scaler.transform(X_latest)
            
            # Make individual predictions
            lr_pred = self._predict_with_linear_regression(X_scaled)
            rf_pred = self._predict_with_random_forest(X_scaled)
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble_prediction(lr_pred, rf_pred)
            
            # Prepare comprehensive prediction result
            prediction_result = {
                'algorithm_1_linear_regression': {
                    'prediction': round(lr_pred, 2),
                    'r2_score': round(self.model_performance['linear_regression']['r2_score'], 3),
                    'rmse': round(self.model_performance['linear_regression']['rmse'], 3)
                },
                'algorithm_2_random_forest': {
                    'prediction': round(rf_pred, 2),
                    'r2_score': round(self.model_performance['random_forest']['r2_score'], 3),
                    'rmse': round(self.model_performance['random_forest']['rmse'], 3)
                },
                'ensemble_prediction': {
                    'final_prediction': round(ensemble_result['ensemble_prediction'], 2),
                    'confidence': round(ensemble_result['confidence'], 1),
                    'prediction_variance': round(ensemble_result['prediction_variance'], 2)
                },
                'model_info': {
                    'algorithms_used': ['Linear Regression', 'Random Forest Regressor'],
                    'ensemble_method': 'Weighted Average (LR: 30%, RF: 70%)',
                    'training_samples': self.model_performance['training_samples'],
                    'feature_count': self.model_performance['feature_count']
                }
            }
            
            return prediction_result, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_ensemble_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the trained ensemble models"""
        if not self.is_trained:
            return {
                'status': 'Not trained',
                'algorithms': ['Linear Regression', 'Random Forest Regressor'],
                'ensemble_method': 'Weighted Average',
                'feature_count': 0,
                'training_samples': 0
            }
        
        return {
            'status': 'Trained',
            'algorithms': ['Linear Regression', 'Random Forest Regressor'],
            'ensemble_method': 'Weighted Average (LR: 30%, RF: 70%)',
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': self.model_performance['training_samples'],
            'performance_metrics': {
                'linear_regression': self.model_performance['linear_regression'],
                'random_forest': self.model_performance['random_forest']
            }
        }

class ARIMATimeSeriesPredictor:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Time Series Predictor
    
    This class implements a simplified ARIMA model for time series forecasting:
    - AR (AutoRegressive): Uses past values to predict future values
    - I (Integrated): Uses differencing to make the series stationary
    - MA (Moving Average): Uses past forecast errors to predict future values
    
    This is a simplified implementation for educational purposes.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA predictor
        
        Args:
            order: ARIMA order (p, d, q) where:
                   p = number of autoregressive terms
                   d = number of differences
                   q = number of moving average terms
        """
        self.order = order
        self.is_trained = False
        self.coefficients = {}
        self.model_performance = {}
        
    def _calculate_autoregressive_component(self, prices: List[float]) -> float:
        """Calculate AR (AutoRegressive) component coefficient"""
        if self.order[0] > 0 and len(prices) > 1:
            # Simplified AR: correlation between consecutive price changes
            diff_prices = np.diff(prices)
            if len(diff_prices) > 1:
                return np.corrcoef(diff_prices[:-1], diff_prices[1:])[0, 1]
        return 0.0
    
    def _calculate_moving_average_component(self, prices: List[float]) -> float:
        """Calculate MA (Moving Average) component coefficient"""
        if self.order[2] > 0:
            # Simplified MA: average of price changes
            diff_prices = np.diff(prices)
            return np.mean(diff_prices)
        return 0.0
    
    def _calculate_trend_component(self, prices: List[float]) -> float:
        """Calculate linear trend component"""
        if len(prices) > 1:
            # Linear trend using least squares
            x = np.arange(len(prices))
            trend_coeff, _ = np.polyfit(x, prices, 1)
            return trend_coeff
        return 0.0
    
    def _calculate_mean_reversion(self, prices: List[float]) -> float:
        """Calculate mean reversion component"""
        return np.mean(prices)
    
    def train_arima_model(self, prices: List[float]) -> Tuple[bool, str]:
        """
        Train the ARIMA model with simplified parameter estimation
        
        Args:
            prices: Historical price data
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if len(prices) < 50:
                return False, f"Insufficient data for ARIMA training (need at least 50 data points, got {len(prices)})"
            
            # Calculate ARIMA components
            ar_coeff = self._calculate_autoregressive_component(prices)
            ma_coeff = self._calculate_moving_average_component(prices)
            trend_coeff = self._calculate_trend_component(prices)
            mean_price = self._calculate_mean_reversion(prices)
            
            # Store coefficients
            self.coefficients = {
                'ar_coefficient': ar_coeff,
                'ma_coefficient': ma_coeff,
                'trend_coefficient': trend_coeff,
                'mean_price': mean_price,
                'order': self.order,
                'training_samples': len(prices)
            }
            
            # Calculate model performance metrics (simplified)
            self.model_performance = {
                'training_samples': len(prices),
                'model_type': 'ARIMA',
                'order': f"ARIMA{self.order[0]},{self.order[1]},{self.order[2]}",
                'coefficients': self.coefficients
            }
            
            self.is_trained = True
            return True, "ARIMA model trained successfully"
            
        except Exception as e:
            return False, f"ARIMA training error: {str(e)}"
    
    def _predict_single_step(self, recent_prices: List[float], step: int) -> float:
        """Predict a single future step using ARIMA model"""
        if not self.is_trained or len(recent_prices) < 2:
            return recent_prices[-1] if recent_prices else 0.0
        
        # AR component: autoregressive term
        ar_term = self.coefficients['ar_coefficient'] * (recent_prices[-1] - recent_prices[-2]) if len(recent_prices) > 1 else 0
        
        # MA component: moving average term
        ma_term = self.coefficients['ma_coefficient']
        
        # Trend component: linear trend
        trend_term = self.coefficients['trend_coefficient'] * step
        
        # Mean reversion component
        mean_reversion = (self.coefficients['mean_price'] - recent_prices[-1]) * 0.1
        
        # Combine all components
        prediction = recent_prices[-1] + ar_term + ma_term + trend_term + mean_reversion
        
        # Ensure non-negative price
        return max(0.0, prediction)
    
    def predict_time_series(self, recent_prices: List[float], steps: int = 5) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Predict future values using trained ARIMA model
        
        Args:
            recent_prices: Recent historical prices
            steps: Number of future steps to predict
            
        Returns:
            Tuple of (prediction_dict, error_message)
        """
        if not self.is_trained:
            return None, "ARIMA model not trained"
        
        try:
            if len(recent_prices) < 2:
                return None, "Insufficient recent data for prediction"
            
            predictions = []
            current_prices = recent_prices.copy()
            
            # Generate multi-step predictions
            for step in range(1, steps + 1):
                pred = self._predict_single_step(current_prices, step)
                predictions.append(pred)
                
                # Update price list for next prediction (sliding window)
                if len(current_prices) > 20:  # Keep only recent prices
                    current_prices = current_prices[-19:] + [pred]
                else:
                    current_prices.append(pred)
            
            # Calculate prediction confidence (based on model stability)
            pred_variance = np.var(predictions) if len(predictions) > 1 else 0
            confidence = max(50, 100 - (pred_variance / np.mean(predictions) * 100)) if np.mean(predictions) > 0 else 50
            
            prediction_result = {
                'algorithm': 'ARIMA Time Series',
                'model_info': {
                    'model_type': 'ARIMA',
                    'order': f"ARIMA{self.order[0]},{self.order[1]},{self.order[2]}",
                    'training_samples': self.coefficients['training_samples'],
                    'coefficients': self.coefficients
                },
                'predictions': {
                    'next_day': round(predictions[0], 2),
                    'multi_step': [round(p, 2) for p in predictions],
                    'confidence': round(confidence, 1),
                    'prediction_variance': round(pred_variance, 2)
                }
            }
            
            return prediction_result, None
            
        except Exception as e:
            return None, f"ARIMA prediction error: {str(e)}"
    
    def get_arima_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the trained ARIMA model"""
        if not self.is_trained:
            return {
                'status': 'Not trained',
                'algorithm': 'ARIMA Time Series',
                'model_type': 'ARIMA',
                'order': f"ARIMA{self.order[0]},{self.order[1]},{self.order[2]}",
                'training_samples': 0
            }
        
        return {
            'status': 'Trained',
            'algorithm': 'ARIMA Time Series',
            'model_type': 'ARIMA',
            'order': f"ARIMA{self.order[0]},{self.order[1]},{self.order[2]}",
            'training_samples': self.coefficients['training_samples'],
            'performance_metrics': self.model_performance
        }
