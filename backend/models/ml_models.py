import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Machine Learning-based stock price predictor"""
    
    def __init__(self):
        self.linear_model = LinearRegression()
        self.random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, prices, volume=None):
        """Create technical features from price data"""
        df = pd.DataFrame({'price': prices})
        
        # Price-based features
        df['price_change'] = df['price'].pct_change()
        df['price_change_2d'] = df['price'].pct_change(2)
        df['price_change_5d'] = df['price'].pct_change(5)
        
        # Moving averages
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        
        # Price ratios
        df['price_sma5_ratio'] = df['price'] / df['sma_5']
        df['price_sma10_ratio'] = df['price'] / df['sma_10']
        df['price_sma20_ratio'] = df['price'] / df['sma_20']
        
        # Volatility features
        df['volatility_5d'] = df['price_change'].rolling(window=5).std()
        df['volatility_10d'] = df['price_change'].rolling(window=10).std()
        
        # Momentum features
        df['momentum_5d'] = df['price'] / df['price'].shift(5) - 1
        df['momentum_10d'] = df['price'] / df['price'].shift(10) - 1
        
        # RSI approximation
        df['rsi'] = self._calculate_rsi(df['price'])
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + (df['volatility_10d'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['volatility_10d'] * 2)
        df['bb_position'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features (if available)
        if volume is not None:
            df['volume'] = volume
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_training_data(self, df, target_days=1):
        """Prepare features and target for training"""
        # Create target (future price)
        df['target'] = df['price'].shift(-target_days)
        
        # Remove rows with NaN targets
        df = df.dropna()
        
        # Select feature columns (exclude price and target)
        feature_cols = [col for col in df.columns if col not in ['price', 'target']]
        
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, feature_cols
    
    def train_models(self, prices, volume=None):
        """Train both Linear Regression and Random Forest models"""
        try:
            # Create features
            df = self.create_features(prices, volume)
            
            if len(df) < 50:  # Need sufficient data
                return False, "Insufficient data for training (need at least 50 data points)"
            
            # Prepare training data
            X, y, self.feature_names = self.prepare_training_data(df)
            
            if len(X) < 30:  # Need sufficient training samples
                return False, "Insufficient training samples after feature creation"
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Linear Regression
            self.linear_model.fit(X_scaled, y)
            
            # Train Random Forest
            self.random_forest.fit(X_scaled, y)
            
            # Calculate model performance
            lr_pred = self.linear_model.predict(X_scaled)
            rf_pred = self.random_forest.predict(X_scaled)
            
            self.linear_score = r2_score(y, lr_pred)
            self.random_forest_score = r2_score(y, rf_pred)
            
            self.is_trained = True
            return True, "Models trained successfully"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict_next_day(self, recent_prices, recent_volume=None):
        """Predict next day's price using trained models"""
        if not self.is_trained:
            return None, "Models not trained"
        
        try:
            # Create features from recent data
            df = self.create_features(recent_prices, recent_volume)
            
            if len(df) == 0:
                return None, "No features generated"
            
            # Get the most recent feature vector
            latest_features = df.iloc[-1:]
            feature_cols = [col for col in latest_features.columns if col not in ['price', 'target']]
            X_latest = latest_features[feature_cols].values
            
            # Scale features
            X_scaled = self.scaler.transform(X_latest)
            
            # Make predictions
            lr_pred = self.linear_model.predict(X_scaled)[0]
            rf_pred = self.random_forest.predict(X_scaled)[0]
            
            # Ensemble prediction (weighted average)
            lr_weight = 0.3
            rf_weight = 0.7
            ensemble_pred = (lr_pred * lr_weight) + (rf_pred * rf_weight)
            
            # Calculate confidence based on model agreement
            pred_diff = abs(lr_pred - rf_pred)
            confidence = max(50, 100 - (pred_diff / ensemble_pred * 100))
            
            return {
                'linear_regression': round(lr_pred, 2),
                'random_forest': round(rf_pred, 2),
                'ensemble': round(ensemble_pred, 2),
                'confidence': round(confidence, 1),
                'model_scores': {
                    'linear_regression': round(self.linear_score, 3),
                    'random_forest': round(self.random_forest_score, 3)
                }
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_model_info(self):
        """Get information about trained models"""
        if not self.is_trained:
            return {
                'status': 'Not trained',
                'feature_count': 0,
                'training_samples': 0
            }
        
        return {
            'status': 'Trained',
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'linear_regression_score': round(self.linear_score, 3),
            'random_forest_score': round(self.random_forest_score, 3),
            'models': ['Linear Regression', 'Random Forest']
        }

class ARIMAPredictor:
    """ARIMA-based time series predictor (simplified implementation)"""
    
    def __init__(self):
        self.is_trained = False
        self.coefficients = None
        
    def train(self, prices, order=(1, 1, 1)):
        """Train ARIMA model (simplified version)"""
        try:
            if len(prices) < 30:
                return False, "Insufficient data for ARIMA training"
            
            # Simplified ARIMA: use differencing and moving average
            diff_prices = np.diff(prices)
            
            # Calculate coefficients based on order
            if order[0] > 0:  # AR component
                ar_coeff = np.corrcoef(diff_prices[:-1], diff_prices[1:])[0, 1]
            else:
                ar_coeff = 0
                
            if order[2] > 0:  # MA component
                ma_coeff = np.mean(diff_prices)
            else:
                ma_coeff = 0
            
            self.coefficients = {
                'ar': ar_coeff,
                'ma': ma_coeff,
                'mean': np.mean(prices),
                'trend': np.polyfit(range(len(prices)), prices, 1)[0]
            }
            
            self.is_trained = True
            return True, "ARIMA model trained successfully"
            
        except Exception as e:
            return False, f"ARIMA training error: {str(e)}"
    
    def predict(self, recent_prices, steps=1):
        """Predict future values using ARIMA model"""
        if not self.is_trained:
            return None, "Model not trained"
        
        try:
            last_price = recent_prices[-1]
            predictions = []
            
            for step in range(1, steps + 1):
                # Simplified ARIMA prediction
                ar_term = self.coefficients['ar'] * (recent_prices[-1] - recent_prices[-2]) if len(recent_prices) > 1 else 0
                ma_term = self.coefficients['ma']
                trend_term = self.coefficients['trend'] * step
                
                next_price = last_price + ar_term + ma_term + trend_term
                predictions.append(max(0, next_price))  # Ensure non-negative price
                last_price = next_price
            
            return {
                'predictions': [round(p, 2) for p in predictions],
                'model_type': 'ARIMA',
                'coefficients': self.coefficients
            }, None
            
        except Exception as e:
            return None, f"ARIMA prediction error: {str(e)}"
