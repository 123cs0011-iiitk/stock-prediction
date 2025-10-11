"""
Stock Price Insight Arena - Prediction Service
Machine learning prediction service for stock price forecasting.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import math
import random

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models.stock_models import (
    PredictionResult, EnsemblePrediction, PredictionModel,
    HistoricalData, StockQuote
)
from services.stock_service import stock_service
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PredictionService:
    """
    Service for generating stock price predictions using machine learning models
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self.cache = {}  # Simple in-memory cache for predictions
        self.cache_ttl = settings.PREDICTION_CACHE_TTL
    
    async def generate_prediction(
        self,
        symbol: str,
        models: List[PredictionModel] = None,
        time_horizon: str = "1 day",
        include_confidence: bool = True,
        include_risk_analysis: bool = True
    ) -> EnsemblePrediction:
        """
        Generate stock price prediction using specified ML models
        
        Args:
            symbol: Stock symbol to predict
            models: List of ML models to use
            time_horizon: Prediction time horizon
            include_confidence: Include confidence scores
            include_risk_analysis: Include risk analysis
            
        Returns:
            EnsemblePrediction object with prediction results
        """
        if models is None:
            models = [PredictionModel.ENSEMBLE]
        
        symbol = symbol.upper().strip()
        logger.info(f"Generating prediction for {symbol} using models: {models}")
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{'-'.join(models)}_{time_horizon}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if datetime.utcnow() - cached_data["timestamp"] < timedelta(seconds=self.cache_ttl):
                    logger.info(f"Returning cached prediction for {symbol}")
                    return cached_data["prediction"]
            
            # Get current stock data
            stock_data = await stock_service.get_complete_stock_data(symbol)
            
            if not stock_data.historical_data or len(stock_data.historical_data) < settings.MIN_HISTORICAL_DATA_POINTS:
                raise ValueError(f"Insufficient historical data for {symbol}. Need at least {settings.MIN_HISTORICAL_DATA_POINTS} data points.")
            
            # Prepare data for ML models
            prices = [data.close for data in stock_data.historical_data]
            volumes = [data.volume for data in stock_data.historical_data]
            
            # Generate individual model predictions
            predictions = []
            
            for model_type in models:
                if model_type == PredictionModel.ENSEMBLE:
                    # Ensemble model combines multiple approaches
                    ensemble_preds = await self._generate_ensemble_prediction(symbol, prices, volumes)
                    predictions.extend(ensemble_preds)
                else:
                    pred_result = await self._generate_single_model_prediction(
                        model_type, symbol, prices, volumes, time_horizon
                    )
                    if pred_result:
                        predictions.append(pred_result)
            
            if not predictions:
                raise ValueError("No predictions could be generated")
            
            # Calculate ensemble prediction
            final_prediction = self._calculate_ensemble_prediction(predictions)
            confidence = self._calculate_prediction_confidence(predictions)
            
            # Calculate trend
            current_price = stock_data.quote.price
            trend = self._determine_trend(final_prediction, current_price)
            
            # Calculate support and resistance levels
            support_level, resistance_level = self._calculate_support_resistance(
                prices, current_price
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(
                stock_data, final_prediction, confidence
            ) if include_risk_analysis else 50.0
            
            # Create ensemble prediction result
            ensemble_prediction = EnsemblePrediction(
                symbol=symbol,
                current_price=current_price,
                predictions=predictions,
                final_prediction=final_prediction,
                confidence=confidence,
                trend=trend,
                support_level=support_level,
                resistance_level=resistance_level,
                risk_score=risk_score
            )
            
            # Cache the result
            self.cache[cache_key] = {
                "prediction": ensemble_prediction,
                "timestamp": datetime.utcnow()
            }
            
            logger.info(f"Generated prediction for {symbol}: {final_prediction:.2f} (confidence: {confidence:.1f}%)")
            return ensemble_prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {str(e)}")
            raise
    
    async def _generate_ensemble_prediction(
        self, symbol: str, prices: List[float], volumes: List[float]
    ) -> List[PredictionResult]:
        """
        Generate ensemble prediction combining multiple models
        
        Args:
            symbol: Stock symbol
            prices: Historical prices
            volumes: Historical volumes
            
        Returns:
            List of PredictionResult objects
        """
        predictions = []
        
        # Linear Regression prediction
        lr_pred = await self._generate_single_model_prediction(
            PredictionModel.LINEAR_REGRESSION, symbol, prices, volumes, "1 day"
        )
        if lr_pred:
            predictions.append(lr_pred)
        
        # Random Forest prediction
        rf_pred = await self._generate_single_model_prediction(
            PredictionModel.RANDOM_FOREST, symbol, prices, volumes, "1 day"
        )
        if rf_pred:
            predictions.append(rf_pred)
        
        # ARIMA prediction
        arima_pred = await self._generate_single_model_prediction(
            PredictionModel.ARIMA, symbol, prices, volumes, "1 day"
        )
        if arima_pred:
            predictions.append(arima_pred)
        
        return predictions
    
    async def _generate_single_model_prediction(
        self,
        model_type: PredictionModel,
        symbol: str,
        prices: List[float],
        volumes: List[float],
        time_horizon: str
    ) -> Optional[PredictionResult]:
        """
        Generate prediction using a single ML model
        
        Args:
            model_type: Type of ML model to use
            symbol: Stock symbol
            prices: Historical prices
            volumes: Historical volumes
            time_horizon: Prediction time horizon
            
        Returns:
            PredictionResult object or None if failed
        """
        try:
            if model_type == PredictionModel.LINEAR_REGRESSION:
                return await self._linear_regression_prediction(symbol, prices, volumes, time_horizon)
            elif model_type == PredictionModel.RANDOM_FOREST:
                return await self._random_forest_prediction(symbol, prices, volumes, time_horizon)
            elif model_type == PredictionModel.ARIMA:
                return await self._arima_prediction(symbol, prices, volumes, time_horizon)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating {model_type} prediction for {symbol}: {str(e)}")
            return None
    
    async def _linear_regression_prediction(
        self, symbol: str, prices: List[float], volumes: List[float], time_horizon: str
    ) -> PredictionResult:
        """
        Generate prediction using Linear Regression
        
        Args:
            symbol: Stock symbol
            prices: Historical prices
            volumes: Historical volumes
            time_horizon: Prediction time horizon
            
        Returns:
            PredictionResult object
        """
        # Create features for linear regression
        X, y = self._create_features(prices, volumes)
        
        if len(X) < 10:
            raise ValueError("Insufficient data for linear regression")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Generate next day prediction
        latest_features = X[-1:].reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        predicted_price = model.predict(latest_features_scaled)[0]
        
        # Calculate confidence based on R² score
        confidence = max(0, min(100, r2 * 100))
        
        return PredictionResult(
            model_name=PredictionModel.LINEAR_REGRESSION,
            predicted_price=round(predicted_price, 2),
            confidence=round(confidence, 1),
            time_horizon=time_horizon,
            model_metrics={
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "training_samples": len(X_train)
            }
        )
    
    async def _random_forest_prediction(
        self, symbol: str, prices: List[float], volumes: List[float], time_horizon: str
    ) -> PredictionResult:
        """
        Generate prediction using Random Forest Regressor
        
        Args:
            symbol: Stock symbol
            prices: Historical prices
            volumes: Historical volumes
            time_horizon: Prediction time horizon
            
        Returns:
            PredictionResult object
        """
        # Create features for random forest
        X, y = self._create_features(prices, volumes)
        
        if len(X) < 20:
            raise ValueError("Insufficient data for random forest")
        
        # Split data for training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Generate next day prediction
        latest_features = X[-1:].reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        predicted_price = model.predict(latest_features_scaled)[0]
        
        # Calculate confidence based on R² score
        confidence = max(0, min(100, r2 * 100))
        
        # Get feature importance
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance.append({
                    "feature": f"feature_{i}",
                    "importance": round(importance, 4)
                })
        
        return PredictionResult(
            model_name=PredictionModel.RANDOM_FOREST,
            predicted_price=round(predicted_price, 2),
            confidence=round(confidence, 1),
            time_horizon=time_horizon,
            model_metrics={
                "r2_score": round(r2, 4),
                "rmse": round(rmse, 4),
                "training_samples": len(X_train)
            },
            feature_importance=feature_importance
        )
    
    async def _arima_prediction(
        self, symbol: str, prices: List[float], volumes: List[float], time_horizon: str
    ) -> PredictionResult:
        """
        Generate prediction using ARIMA (simplified implementation)
        
        Args:
            symbol: Stock symbol
            prices: Historical prices
            volumes: Historical volumes
            time_horizon: Prediction time horizon
            
        Returns:
            PredictionResult object
        """
        # Simplified ARIMA implementation using linear trend
        if len(prices) < 10:
            raise ValueError("Insufficient data for ARIMA")
        
        # Calculate trend using simple linear regression
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Simple linear trend
        slope, intercept = np.polyfit(x, y, 1)
        
        # Predict next value
        next_x = len(prices)
        predicted_price = slope * next_x + intercept
        
        # Calculate confidence based on trend strength
        trend_strength = abs(slope) / np.mean(prices)
        confidence = min(100, max(20, trend_strength * 1000))
        
        return PredictionResult(
            model_name=PredictionModel.ARIMA,
            predicted_price=round(predicted_price, 2),
            confidence=round(confidence, 1),
            time_horizon=time_horizon,
            model_metrics={
                "trend_slope": round(slope, 6),
                "trend_intercept": round(intercept, 2),
                "trend_strength": round(trend_strength, 4)
            }
        )
    
    def _create_features(self, prices: List[float], volumes: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features for ML models from price and volume data
        
        Args:
            prices: Historical prices
            volumes: Historical volumes
            
        Returns:
            Tuple of (features, targets)
        """
        features = []
        targets = []
        
        for i in range(5, len(prices)):  # Need at least 5 previous values
            feature_row = []
            
            # Price-based features
            feature_row.append(prices[i-1])  # Previous price
            feature_row.append(prices[i-2])  # Price 2 days ago
            feature_row.append(prices[i-3])  # Price 3 days ago
            feature_row.append(prices[i-4])  # Price 4 days ago
            feature_row.append(prices[i-5])  # Price 5 days ago
            
            # Price changes
            feature_row.append(prices[i-1] - prices[i-2])
            feature_row.append(prices[i-2] - prices[i-3])
            feature_row.append(prices[i-3] - prices[i-4])
            
            # Moving averages
            feature_row.append(sum(prices[i-5:i]) / 5)  # 5-day MA
            feature_row.append(sum(prices[i-10:i]) / 10 if i >= 10 else prices[i-1])  # 10-day MA
            
            # Volume features
            feature_row.append(volumes[i-1])
            feature_row.append(sum(volumes[i-5:i]) / 5)  # 5-day volume MA
            
            # Volatility
            recent_prices = prices[i-5:i]
            if len(recent_prices) > 1:
                returns = [(recent_prices[j] - recent_prices[j-1]) / recent_prices[j-1] 
                          for j in range(1, len(recent_prices))]
                volatility = np.std(returns) if returns else 0
                feature_row.append(volatility)
            else:
                feature_row.append(0)
            
            features.append(feature_row)
            targets.append(prices[i])  # Target is next day's price
        
        return np.array(features), np.array(targets)
    
    def _calculate_ensemble_prediction(self, predictions: List[PredictionResult]) -> float:
        """
        Calculate ensemble prediction from individual model predictions
        
        Args:
            predictions: List of individual model predictions
            
        Returns:
            Ensemble prediction value
        """
        if not predictions:
            return 0.0
        
        # Weighted average based on confidence
        total_weight = sum(pred.confidence for pred in predictions)
        if total_weight == 0:
            return sum(pred.predicted_price for pred in predictions) / len(predictions)
        
        weighted_sum = sum(pred.predicted_price * pred.confidence for pred in predictions)
        return weighted_sum / total_weight
    
    def _calculate_prediction_confidence(self, predictions: List[PredictionResult]) -> float:
        """
        Calculate overall confidence for ensemble prediction
        
        Args:
            predictions: List of individual model predictions
            
        Returns:
            Overall confidence percentage
        """
        if not predictions:
            return 0.0
        
        # Average confidence with penalty for disagreement
        avg_confidence = sum(pred.confidence for pred in predictions) / len(predictions)
        
        # Calculate disagreement penalty
        prices = [pred.predicted_price for pred in predictions]
        if len(prices) > 1:
            price_std = np.std(prices)
            price_mean = np.mean(prices)
            disagreement = price_std / price_mean if price_mean > 0 else 0
            disagreement_penalty = min(20, disagreement * 100)  # Max 20% penalty
        else:
            disagreement_penalty = 0
        
        final_confidence = max(0, avg_confidence - disagreement_penalty)
        return round(final_confidence, 1)
    
    def _determine_trend(self, predicted_price: float, current_price: float) -> str:
        """
        Determine market trend based on prediction
        
        Args:
            predicted_price: Predicted price
            current_price: Current price
            
        Returns:
            Trend direction (bullish, bearish, neutral)
        """
        change_percent = (predicted_price - current_price) / current_price * 100
        
        if change_percent > 2:
            return "bullish"
        elif change_percent < -2:
            return "bearish"
        else:
            return "neutral"
    
    def _calculate_support_resistance(
        self, prices: List[float], current_price: float
    ) -> Tuple[float, float]:
        """
        Calculate support and resistance levels
        
        Args:
            prices: Historical prices
            current_price: Current price
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if len(prices) < 20:
            # Use simple percentage-based levels
            support = current_price * 0.95
            resistance = current_price * 1.05
        else:
            # Use recent price ranges
            recent_prices = prices[-20:]
            support = min(recent_prices) * 0.98
            resistance = max(recent_prices) * 1.02
        
        return round(support, 2), round(resistance, 2)
    
    def _calculate_risk_score(
        self, stock_data: Any, predicted_price: float, confidence: float
    ) -> float:
        """
        Calculate investment risk score
        
        Args:
            stock_data: Complete stock data
            predicted_price: Predicted price
            confidence: Prediction confidence
            
        Returns:
            Risk score (0-100, higher is riskier)
        """
        risk_factors = []
        
        # Volatility risk
        if stock_data.technical_indicators and stock_data.technical_indicators.volatility:
            volatility = stock_data.technical_indicators.volatility
            if volatility > 30:
                risk_factors.append(30)
            elif volatility > 20:
                risk_factors.append(20)
            else:
                risk_factors.append(10)
        else:
            risk_factors.append(20)
        
        # Prediction confidence risk
        if confidence < 50:
            risk_factors.append(30)
        elif confidence < 70:
            risk_factors.append(20)
        else:
            risk_factors.append(10)
        
        # Price change risk
        current_price = stock_data.quote.price
        change_percent = abs(predicted_price - current_price) / current_price * 100
        if change_percent > 10:
            risk_factors.append(25)
        elif change_percent > 5:
            risk_factors.append(15)
        else:
            risk_factors.append(5)
        
        return min(100, sum(risk_factors))
    
    async def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for models trained on a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with model performance metrics
        """
        if symbol in self.model_performance:
            return self.model_performance[symbol]
        
        # Generate performance metrics
        try:
            stock_data = await stock_service.get_complete_stock_data(symbol)
            if not stock_data.historical_data:
                return {"error": "No historical data available"}
            
            prices = [data.close for data in stock_data.historical_data]
            volumes = [data.volume for data in stock_data.historical_data]
            
            performance = {
                "symbol": symbol,
                "data_points": len(prices),
                "models": {}
            }
            
            # Test each model
            for model_type in [PredictionModel.LINEAR_REGRESSION, PredictionModel.RANDOM_FOREST]:
                try:
                    pred_result = await self._generate_single_model_prediction(
                        model_type, symbol, prices, volumes, "1 day"
                    )
                    if pred_result:
                        performance["models"][model_type.value] = pred_result.model_metrics
                except Exception as e:
                    performance["models"][model_type.value] = {"error": str(e)}
            
            self.model_performance[symbol] = performance
            return performance
            
        except Exception as e:
            logger.error(f"Error getting model performance for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the prediction service
        
        Returns:
            Dictionary with health status information
        """
        return {
            "service": "PredictionService",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "models_available": [model.value for model in PredictionModel],
            "cache_size": len(self.cache),
            "model_performance_tracked": len(self.model_performance)
        }


# Create singleton instance
prediction_service = PredictionService()
