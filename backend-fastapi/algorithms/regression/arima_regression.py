"""
ARIMA (AutoRegressive Integrated Moving Average) for Stock Price Prediction
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ARIMARegression:
    """
    ARIMA model for time series stock price prediction
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 auto_arima: bool = False):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            auto_arima: Whether to automatically determine best parameters
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.model = None
        self.fitted_model = None
        self.is_trained = False
        self.best_order = None
        self.best_aic = None
        
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series to test
            
        Returns:
            Dictionary with stationarity test results
        """
        try:
            result = adfuller(series.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            logger.error(f"Error checking stationarity: {str(e)}")
            return {'is_stationary': False, 'error': str(e)}
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Make time series stationary by differencing
        
        Args:
            series: Time series to make stationary
            max_diff: Maximum number of differences to apply
            
        Returns:
            Tuple of (stationary_series, number_of_differences)
        """
        current_series = series.copy()
        diff_count = 0
        
        for i in range(max_diff + 1):
            stationarity_result = self.check_stationarity(current_series)
            
            if stationarity_result['is_stationary']:
                break
            
            if i < max_diff:
                current_series = current_series.diff().dropna()
                diff_count += 1
        
        return current_series, diff_count
    
    def find_best_order(self, series: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
        """
        Find best ARIMA order using AIC
        
        Args:
            series: Time series to model
            max_p: Maximum AR order
            max_q: Maximum MA order
            
        Returns:
            Best (p, d, q) order
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Determine differencing order
        _, d = self.make_stationary(series)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        
                except Exception:
                    continue
        
        self.best_order = best_order
        self.best_aic = best_aic
        
        return best_order
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'close') -> pd.Series:
        """
        Prepare time series data for ARIMA modeling
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Time series data
        """
        # Ensure data is sorted by date
        if 'date' in df.columns:
            df = df.sort_values('date')
        elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        
        # Create time series
        series = pd.Series(df[target_column].values, index=df.index)
        
        # Remove any NaN values
        series = series.dropna()
        
        if len(series) < 10:
            raise ValueError("Insufficient data for ARIMA modeling")
        
        return series
    
    def train(self, df: pd.DataFrame, target_column: str = 'close') -> Dict:
        """
        Train the ARIMA model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Training ARIMA model")
            
            # Prepare data
            series = self.prepare_data(df, target_column)
            
            # Determine best order if auto_arima is enabled
            if self.auto_arima:
                order = self.find_best_order(series)
                logger.info(f"Best ARIMA order found: {order}, AIC: {self.best_aic}")
            else:
                order = self.order
            
            # Create and fit model
            self.model = ARIMA(series, order=order, seasonal_order=self.seasonal_order)
            self.fitted_model = self.model.fit()
            
            # Make in-sample predictions for evaluation
            fitted_values = self.fitted_model.fittedvalues
            
            # Calculate metrics
            actual_values = series.iloc[1:]  # Skip first value due to differencing
            predicted_values = fitted_values.iloc[1:]
            
            mse = mean_squared_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_values, predicted_values)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            self.is_trained = True
            
            results = {
                'order': order,
                'seasonal_order': self.seasonal_order,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'training_samples': len(series),
                'model_summary': str(self.fitted_model.summary()),
                'auto_arima': self.auto_arima
            }
            
            if self.auto_arima:
                results['best_aic'] = self.best_aic
                results['search_info'] = f"Auto-selected order {order} with AIC {self.best_aic}"
            
            logger.info(f"ARIMA training completed. Order: {order}, RMSE: {rmse:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training ARIMA: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, target_column: str = 'close', steps: int = 1) -> Dict:
        """
        Make predictions using trained ARIMA model
        
        Args:
            df: DataFrame with stock data
            target_column: Column to use as target
            steps: Number of steps ahead to predict
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare data
            series = self.prepare_data(df, target_column)
            
            # Create new model with latest data
            current_model = ARIMA(series, order=self.fitted_model.model.order)
            current_fitted = current_model.fit()
            
            # Make predictions
            forecast = current_fitted.forecast(steps=steps)
            confidence_intervals = current_fitted.get_forecast(steps=steps).conf_int()
            
            # Get latest actual price
            current_price = series.iloc[-1]
            
            # Calculate predictions
            if steps == 1:
                predicted_price = forecast.iloc[0]
                price_change = predicted_price - current_price
                price_change_percent = (price_change / current_price) * 100
                direction = "up" if price_change > 0 else "down"
                
                results = {
                    'predicted_price': float(predicted_price),
                    'current_price': float(current_price),
                    'price_change': float(price_change),
                    'price_change_percent': float(price_change_percent),
                    'direction': direction,
                    'confidence_interval_lower': float(confidence_intervals.iloc[0, 0]),
                    'confidence_interval_upper': float(confidence_intervals.iloc[0, 1]),
                    'model_type': 'ARIMA',
                    'order': self.fitted_model.model.order,
                    'aic': float(self.fitted_model.aic)
                }
            else:
                # Multiple steps prediction
                predictions = forecast.tolist()
                ci_lower = confidence_intervals.iloc[:, 0].tolist()
                ci_upper = confidence_intervals.iloc[:, 1].tolist()
                
                results = {
                    'predictions': [float(p) for p in predictions],
                    'confidence_intervals_lower': [float(ci) for ci in ci_lower],
                    'confidence_intervals_upper': [float(ci) for ci in ci_upper],
                    'current_price': float(current_price),
                    'model_type': 'ARIMA',
                    'order': self.fitted_model.model.order,
                    'aic': float(self.fitted_model.aic),
                    'steps': steps
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error making ARIMA prediction: {str(e)}")
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
            "model_type": "ARIMA",
            "order": self.fitted_model.model.order,
            "seasonal_order": self.fitted_model.model.seasonal_order,
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "auto_arima": self.auto_arima,
            "is_trained": self.is_trained
        }
