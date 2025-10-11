#!/usr/bin/env python3
"""
Machine Learning Prediction Script
Executes ensemble ML models for stock price prediction
"""

import sys
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def create_features(prices, volumes):
    """Create technical features from price and volume data"""
    prices = np.array(prices)
    volumes = np.array(volumes)
    
    features = []
    
    # Price-based features
    for i in range(len(prices)):
        feature_row = []
        
        # Current price
        feature_row.append(prices[i])
        
        # Price changes (1, 2, 3, 5, 10 day)
        for lag in [1, 2, 3, 5, 10]:
            if i >= lag:
                feature_row.append(prices[i] - prices[i-lag])
                feature_row.append((prices[i] - prices[i-lag]) / prices[i-lag])
            else:
                feature_row.extend([0, 0])
        
        # Moving averages (5, 10, 20 day)
        for window in [5, 10, 20]:
            if i >= window:
                ma = np.mean(prices[i-window:i])
                feature_row.append(ma)
                feature_row.append(prices[i] / ma)  # Price to MA ratio
            else:
                feature_row.extend([prices[i], 1])
        
        # Volatility (standard deviation of returns)
        if i >= 10:
            returns = np.diff(prices[i-10:i]) / prices[i-10:i-1]
            feature_row.append(np.std(returns))
        else:
            feature_row.append(0)
        
        # Volume features
        feature_row.append(volumes[i])
        if i >= 5:
            feature_row.append(np.mean(volumes[i-5:i]))
        else:
            feature_row.append(volumes[i])
        
        # Volume-price trend
        if i > 0:
            feature_row.append(volumes[i] * prices[i])
        else:
            feature_row.append(volumes[i] * prices[i])
        
        features.append(feature_row)
    
    return np.array(features)

def train_models(X, y):
    """Train ensemble models"""
    results = {}
    
    # Split data for training and validation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    
    results['linear_regression'] = {
        'model': 'LinearRegression',
        'r2_score': float(lr_r2),
        'rmse': float(lr_rmse),
        'coefficients': lr_model.coef_.tolist(),
        'intercept': float(lr_model.intercept_)
    }
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    
    results['random_forest'] = {
        'model': 'RandomForestRegressor',
        'r2_score': float(rf_r2),
        'rmse': float(rf_rmse),
        'feature_importance': rf_model.feature_importances_.tolist()
    }
    
    # Ensemble prediction (weighted average)
    ensemble_pred = 0.3 * lr_pred + 0.7 * rf_pred
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    results['ensemble'] = {
        'model': 'Ensemble',
        'r2_score': float(ensemble_r2),
        'rmse': float(ensemble_rmse),
        'weights': {'linear_regression': 0.3, 'random_forest': 0.7}
    }
    
    # Next day prediction
    latest_features = X[-1:].reshape(1, -1)
    latest_features_scaled = scaler.transform(latest_features)
    
    lr_next_pred = lr_model.predict(latest_features_scaled)[0]
    rf_next_pred = rf_model.predict(latest_features_scaled)[0]
    ensemble_next_pred = 0.3 * lr_next_pred + 0.7 * rf_next_pred
    
    # Calculate confidence based on model agreement
    pred_diff = abs(lr_next_pred - rf_next_pred)
    confidence = max(0, 100 - (pred_diff / lr_next_pred * 100))
    
    results['prediction'] = {
        'next_day_price': float(ensemble_next_pred),
        'linear_regression_pred': float(lr_next_pred),
        'random_forest_pred': float(rf_next_pred),
        'confidence': float(confidence),
        'current_price': float(y[-1])
    }
    
    return results

def main():
    """Main execution function"""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        symbol = input_data.get('symbol', 'UNKNOWN')
        prices = input_data.get('prices', [])
        volumes = input_data.get('volumes', [])
        
        if len(prices) < 50:
            result = {
                'success': False,
                'error': 'Insufficient data for ML training (need at least 50 data points)',
                'data_points': len(prices)
            }
        else:
            # Create features
            X = create_features(prices, volumes)
            
            # Prepare target variable (next day prices)
            y = prices[1:]  # Target is next day's price
            
            # Ensure X and y have same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            # Train models and get predictions
            results = train_models(X, y)
            
            result = {
                'success': True,
                'symbol': symbol,
                'data_points': len(prices),
                'features_created': X.shape[1],
                'models': results,
                'timestamp': input_data.get('timestamp')
            }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result))

if __name__ == '__main__':
    main()
