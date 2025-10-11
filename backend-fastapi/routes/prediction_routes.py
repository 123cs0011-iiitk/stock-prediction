"""
Stock Price Insight Arena - Prediction Routes
API routes for stock price predictions using machine learning models.
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import List, Optional
import logging
from datetime import datetime

from models.stock_models import (
    PredictionRequest, PredictionResponse, PredictionModel,
    EnsemblePrediction, ErrorResponse
)
from services.prediction_service import prediction_service
from services.stock_service import stock_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=dict)
async def prediction_service_health():
    """
    Health check endpoint for prediction service
    """
    try:
        health_status = await prediction_service.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Prediction service health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction service health check failed")


@router.get("/predict/{symbol}", response_model=PredictionResponse)
async def predict_stock_price(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, GOOGL)"),
    models: Optional[str] = Query("ensemble", description="Comma-separated list of models (linear_regression,random_forest,arima,ensemble)"),
    time_horizon: Optional[str] = Query("1 day", description="Prediction time horizon"),
    include_confidence: Optional[bool] = Query(True, description="Include confidence scores"),
    include_risk_analysis: Optional[bool] = Query(True, description="Include risk analysis")
):
    """
    Generate stock price prediction using machine learning models
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    - **models**: ML models to use (linear_regression, random_forest, arima, ensemble)
    - **time_horizon**: Prediction time horizon (e.g., "1 day", "1 week", "1 month")
    - **include_confidence**: Include confidence scores in response
    - **include_risk_analysis**: Include risk analysis in response
    
    Returns ML-powered stock price prediction with:
    - Predicted price for specified time horizon
    - Confidence score based on model agreement
    - Trend analysis (bullish/bearish/neutral)
    - Support and resistance levels
    - Risk assessment
    """
    try:
        logger.info(f"Generating prediction for {symbol}")
        
        # Parse models parameter
        model_list = []
        if models.lower() == "ensemble":
            model_list = [PredictionModel.ENSEMBLE]
        else:
            model_names = [m.strip() for m in models.split(",")]
            for model_name in model_names:
                try:
                    model_list.append(PredictionModel(model_name))
                except ValueError:
                    logger.warning(f"Invalid model name: {model_name}")
                    continue
        
        if not model_list:
            raise HTTPException(
                status_code=400,
                detail="No valid models specified. Available models: linear_regression, random_forest, arima, ensemble"
            )
        
        # Generate prediction
        prediction = await prediction_service.generate_prediction(
            symbol=symbol,
            models=model_list,
            time_horizon=time_horizon,
            include_confidence=include_confidence,
            include_risk_analysis=include_risk_analysis
        )
        
        return PredictionResponse(
            success=True,
            message=f"Prediction generated successfully for {symbol}",
            data=prediction
        )
        
    except ValueError as e:
        logger.warning(f"Invalid prediction request for {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")


@router.post("/predict", response_model=PredictionResponse)
async def predict_stock_price_post(
    request: PredictionRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Generate stock price prediction using POST request with detailed parameters
    
    - **symbol**: Stock symbol to predict
    - **models**: List of ML models to use
    - **time_horizon**: Prediction time horizon
    - **include_confidence**: Include confidence scores
    - **include_risk_analysis**: Include risk analysis
    
    Returns comprehensive prediction with all requested analysis.
    """
    try:
        logger.info(f"Generating prediction for {request.symbol} via POST")
        
        # Generate prediction
        prediction = await prediction_service.generate_prediction(
            symbol=request.symbol,
            models=request.models,
            time_horizon=request.time_horizon,
            include_confidence=request.include_confidence,
            include_risk_analysis=request.include_risk_analysis
        )
        
        return PredictionResponse(
            success=True,
            message=f"Prediction generated successfully for {request.symbol}",
            data=prediction
        )
        
    except ValueError as e:
        logger.warning(f"Invalid prediction request for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating prediction for {request.symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate prediction")


@router.get("/models/performance/{symbol}")
async def get_model_performance(
    symbol: str = Path(..., description="Stock symbol")
):
    """
    Get performance metrics for ML models trained on a specific symbol
    
    - **symbol**: Stock symbol to get model performance for
    
    Returns detailed performance metrics including:
    - R² scores for each model
    - RMSE (Root Mean Square Error)
    - Training sample counts
    - Model-specific metrics
    """
    try:
        logger.info(f"Getting model performance for {symbol}")
        
        performance = await prediction_service.get_model_performance(symbol)
        
        if "error" in performance:
            raise HTTPException(
                status_code=404,
                detail=performance["error"]
            )
        
        return {
            "success": True,
            "message": f"Model performance retrieved for {symbol}",
            "data": performance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model performance for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model performance")


@router.get("/models/available")
async def get_available_models():
    """
    Get list of available ML models for predictions
    
    Returns information about all available machine learning models including:
    - Model names and descriptions
    - Capabilities and use cases
    - Performance characteristics
    """
    try:
        logger.info("Getting available models")
        
        models_info = {
            "linear_regression": {
                "name": "Linear Regression",
                "description": "Linear relationship modeling for price trends",
                "use_case": "Trend analysis and simple predictions",
                "strengths": ["Fast training", "Interpretable", "Good for linear trends"],
                "limitations": ["Assumes linear relationships", "Limited for complex patterns"]
            },
            "random_forest": {
                "name": "Random Forest Regressor",
                "description": "Ensemble of decision trees for non-linear pattern recognition",
                "use_case": "Complex pattern detection and robust predictions",
                "strengths": ["Handles non-linear patterns", "Feature importance", "Robust to outliers"],
                "limitations": ["Can overfit", "Less interpretable", "Requires more data"]
            },
            "arima": {
                "name": "ARIMA Time Series",
                "description": "AutoRegressive Integrated Moving Average for time series forecasting",
                "use_case": "Time series trend analysis and forecasting",
                "strengths": ["Time series specific", "Good for trends", "Statistical foundation"],
                "limitations": ["Assumes stationarity", "Limited for complex patterns"]
            },
            "ensemble": {
                "name": "Ensemble Model",
                "description": "Combines multiple models for improved accuracy",
                "use_case": "Best overall predictions with confidence scoring",
                "strengths": ["Highest accuracy", "Confidence scoring", "Robust predictions"],
                "limitations": ["More complex", "Requires more computational resources"]
            }
        }
        
        return {
            "success": True,
            "message": "Available models retrieved successfully",
            "models": models_info,
            "recommendations": {
                "best_overall": "ensemble",
                "fastest": "linear_regression",
                "most_accurate": "ensemble",
                "most_interpretable": "linear_regression"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get available models")


@router.get("/batch-predict")
async def batch_predict_stocks(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    models: Optional[str] = Query("ensemble", description="Models to use"),
    time_horizon: Optional[str] = Query("1 day", description="Prediction time horizon")
):
    """
    Generate predictions for multiple stocks in a single request
    
    - **symbols**: Comma-separated list of stock symbols (e.g., AAPL,GOOGL,MSFT)
    - **models**: ML models to use (default: ensemble)
    - **time_horizon**: Prediction time horizon (default: 1 day)
    
    Returns predictions for all requested symbols.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 symbols allowed per batch prediction request"
            )
        
        logger.info(f"Generating batch predictions for {len(symbol_list)} symbols")
        
        # Parse models
        model_list = [PredictionModel.ENSEMBLE]
        if models.lower() != "ensemble":
            model_names = [m.strip() for m in models.split(",")]
            model_list = []
            for model_name in model_names:
                try:
                    model_list.append(PredictionModel(model_name))
                except ValueError:
                    continue
        
        predictions = []
        errors = []
        
        for symbol in symbol_list:
            try:
                prediction = await prediction_service.generate_prediction(
                    symbol=symbol,
                    models=model_list,
                    time_horizon=time_horizon
                )
                predictions.append(prediction)
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        return {
            "success": True,
            "message": f"Batch predictions generated for {len(predictions)} symbols",
            "predictions": predictions,
            "errors": errors,
            "requested_count": len(symbol_list),
            "successful_count": len(predictions),
            "models_used": [model.value for model in model_list],
            "time_horizon": time_horizon,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid batch prediction request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating batch predictions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate batch predictions")


@router.get("/trend-analysis/{symbol}")
async def get_trend_analysis(
    symbol: str = Path(..., description="Stock symbol"),
    period: Optional[str] = Query("30d", description="Analysis period (7d, 30d, 90d, 1y)")
):
    """
    Get comprehensive trend analysis for a stock
    
    - **symbol**: Stock symbol to analyze
    - **period**: Analysis period (7d, 30d, 90d, 1y)
    
    Returns trend analysis including:
    - Short-term and long-term trends
    - Momentum indicators
    - Volatility analysis
    - Trend strength and direction
    """
    try:
        logger.info(f"Getting trend analysis for {symbol}, period: {period}")
        
        # Get historical data for trend analysis
        stock_data = await stock_service.get_complete_stock_data(symbol)
        
        if not stock_data.historical_data:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data available for {symbol}"
            )
        
        # Calculate trend metrics
        prices = [data.close for data in stock_data.historical_data]
        current_price = stock_data.quote.price
        
        # Short-term trend (last 5 days)
        short_term_trend = "neutral"
        if len(prices) >= 5:
            short_change = (prices[-1] - prices[-5]) / prices[-5] * 100
            if short_change > 2:
                short_term_trend = "bullish"
            elif short_change < -2:
                short_term_trend = "bearish"
        
        # Long-term trend (last 20 days)
        long_term_trend = "neutral"
        if len(prices) >= 20:
            long_change = (prices[-1] - prices[-20]) / prices[-20] * 100
            if long_change > 5:
                long_term_trend = "bullish"
            elif long_change < -5:
                long_term_trend = "bearish"
        
        # Volatility analysis
        volatility = 0
        if len(prices) > 1:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = sum([abs(r) for r in returns[-10:]]) / min(10, len(returns)) * 100
        
        trend_analysis = {
            "symbol": symbol,
            "current_price": current_price,
            "short_term_trend": {
                "direction": short_term_trend,
                "change_percent": round(short_change, 2) if len(prices) >= 5 else 0
            },
            "long_term_trend": {
                "direction": long_term_trend,
                "change_percent": round(long_change, 2) if len(prices) >= 20 else 0
            },
            "volatility": {
                "current": round(volatility, 2),
                "level": "high" if volatility > 3 else "medium" if volatility > 1 else "low"
            },
            "technical_indicators": stock_data.technical_indicators,
            "data_points": len(prices),
            "analysis_period": period
        }
        
        return {
            "success": True,
            "message": f"Trend analysis completed for {symbol}",
            "data": trend_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting trend analysis for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get trend analysis")


@router.get("/prediction-history/{symbol}")
async def get_prediction_history(
    symbol: str = Path(..., description="Stock symbol"),
    limit: Optional[int] = Query(10, ge=1, le=50, description="Number of historical predictions to return")
):
    """
    Get historical predictions for a stock (if available)
    
    - **symbol**: Stock symbol
    - **limit**: Number of historical predictions to return
    
    Returns list of previous predictions with accuracy metrics.
    """
    try:
        logger.info(f"Getting prediction history for {symbol}")
        
        # In a real implementation, this would fetch from database
        # For now, return a placeholder response
        prediction_history = {
            "symbol": symbol,
            "predictions": [],
            "accuracy_metrics": {
                "total_predictions": 0,
                "accurate_predictions": 0,
                "accuracy_percentage": 0.0,
                "average_error": 0.0
            },
            "message": "Prediction history feature coming soon"
        }
        
        return {
            "success": True,
            "message": f"Prediction history retrieved for {symbol}",
            "data": prediction_history,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting prediction history for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get prediction history")
