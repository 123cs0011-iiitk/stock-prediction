"""
Stock Price Insight Arena - Pydantic Models
Data schemas for request and response validation using Pydantic models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from enum import Enum


class TimeFrame(str, Enum):
    """Available time frames for historical data"""
    ONE_MINUTE = "1min"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    THIRTY_MINUTES = "30min"
    ONE_HOUR = "1hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class Currency(str, Enum):
    """Supported currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    INR = "INR"


class PredictionModel(str, Enum):
    """Available ML models for predictions"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    ARIMA = "arima"
    ENSEMBLE = "ensemble"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(..., description="Whether the request was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    message: Optional[str] = Field(None, description="Response message")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(False, description="Always False for error responses")
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Stock Data Models
class StockQuote(BaseModel):
    """Real-time stock quote data"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    price: float = Field(..., gt=0, description="Current stock price")
    change: float = Field(..., description="Price change from previous close")
    change_percent: float = Field(..., description="Percentage change from previous close")
    volume: int = Field(..., ge=0, description="Trading volume")
    high: float = Field(..., gt=0, description="Day's high price")
    low: float = Field(..., gt=0, description="Day's low price")
    open: float = Field(..., gt=0, description="Opening price")
    previous_close: float = Field(..., gt=0, description="Previous closing price")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    currency: Currency = Field(Currency.USD, description="Currency of the price")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Quote timestamp")
    data_source: str = Field(..., description="Data source (e.g., Alpha Vantage, Yahoo Finance)")


class CompanyInfo(BaseModel):
    """Company information model"""
    symbol: str = Field(..., description="Stock symbol")
    name: str = Field(..., description="Company name")
    description: Optional[str] = Field(None, description="Company description")
    sector: Optional[str] = Field(None, description="Business sector")
    industry: Optional[str] = Field(None, description="Industry")
    website: Optional[str] = Field(None, description="Company website")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees")
    founded: Optional[int] = Field(None, description="Founded year")
    ceo: Optional[str] = Field(None, description="CEO name")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    currency: Currency = Field(Currency.USD, description="Currency")


class HistoricalData(BaseModel):
    """Historical stock data point"""
    date: date = Field(..., description="Trading date")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: Optional[float] = Field(None, gt=0, description="Adjusted closing price")
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        """Validate that high is the highest price"""
        if 'low' in values and v < values['low']:
            raise ValueError('High price must be greater than or equal to low price')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        """Validate that low is the lowest price"""
        if 'high' in values and v > values['high']:
            raise ValueError('Low price must be less than or equal to high price')
        return v


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators"""
    # Moving Averages
    sma_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    ema_12: Optional[float] = Field(None, description="12-day Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-day Exponential Moving Average")
    
    # Oscillators
    rsi: Optional[float] = Field(None, ge=0, le=100, description="Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    
    # Bollinger Bands
    bb_upper: Optional[float] = Field(None, description="Bollinger Bands upper band")
    bb_middle: Optional[float] = Field(None, description="Bollinger Bands middle band")
    bb_lower: Optional[float] = Field(None, description="Bollinger Bands lower band")
    
    # Other indicators
    volatility: Optional[float] = Field(None, ge=0, description="Price volatility (annualized %)")
    atr: Optional[float] = Field(None, ge=0, description="Average True Range")
    
    # Volume indicators
    volume_sma: Optional[float] = Field(None, description="Volume Simple Moving Average")
    obv: Optional[float] = Field(None, description="On-Balance Volume")


class StockData(BaseModel):
    """Complete stock data including quote, company info, and historical data"""
    symbol: str = Field(..., description="Stock symbol")
    quote: StockQuote = Field(..., description="Current stock quote")
    company_info: Optional[CompanyInfo] = Field(None, description="Company information")
    historical_data: List[HistoricalData] = Field(default_factory=list, description="Historical price data")
    technical_indicators: Optional[TechnicalIndicators] = Field(None, description="Technical analysis indicators")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


# Prediction Models
class PredictionResult(BaseModel):
    """ML model prediction result"""
    model_name: PredictionModel = Field(..., description="ML model used for prediction")
    predicted_price: float = Field(..., gt=0, description="Predicted stock price")
    confidence: float = Field(..., ge=0, le=100, description="Prediction confidence percentage")
    time_horizon: str = Field(..., description="Prediction time horizon (e.g., '1 day', '1 week')")
    model_metrics: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    feature_importance: Optional[List[Dict[str, Any]]] = Field(None, description="Feature importance scores")


class EnsemblePrediction(BaseModel):
    """Ensemble prediction combining multiple models"""
    symbol: str = Field(..., description="Stock symbol")
    current_price: float = Field(..., gt=0, description="Current stock price")
    predictions: List[PredictionResult] = Field(..., description="Individual model predictions")
    final_prediction: float = Field(..., gt=0, description="Final ensemble prediction")
    confidence: float = Field(..., ge=0, le=100, description="Overall prediction confidence")
    trend: str = Field(..., description="Predicted trend (bullish, bearish, neutral)")
    support_level: float = Field(..., gt=0, description="Support level price")
    resistance_level: float = Field(..., gt=0, description="Resistance level price")
    risk_score: float = Field(..., ge=0, le=100, description="Investment risk score")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")


# Request Models
class StockQuoteRequest(BaseModel):
    """Request model for fetching stock quote"""
    symbol: str = Field(..., description="Stock symbol to fetch quote for")
    currency: Optional[Currency] = Field(Currency.USD, description="Desired currency for the quote")
    refresh: Optional[bool] = Field(False, description="Force refresh from data source")


class HistoricalDataRequest(BaseModel):
    """Request model for fetching historical data"""
    symbol: str = Field(..., description="Stock symbol")
    timeframe: TimeFrame = Field(TimeFrame.DAILY, description="Data timeframe")
    start_date: Optional[date] = Field(None, description="Start date for historical data")
    end_date: Optional[date] = Field(None, description="End date for historical data")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum number of data points")
    include_indicators: Optional[bool] = Field(True, description="Include technical indicators")


class PredictionRequest(BaseModel):
    """Request model for generating predictions"""
    symbol: str = Field(..., description="Stock symbol to predict")
    models: Optional[List[PredictionModel]] = Field(
        default_factory=lambda: [PredictionModel.ENSEMBLE],
        description="ML models to use for prediction"
    )
    time_horizon: Optional[str] = Field("1 day", description="Prediction time horizon")
    include_confidence: Optional[bool] = Field(True, description="Include confidence scores")
    include_risk_analysis: Optional[bool] = Field(True, description="Include risk analysis")


class SearchRequest(BaseModel):
    """Request model for searching stocks"""
    query: str = Field(..., min_length=1, max_length=100, description="Search query")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Maximum number of results")
    include_quotes: Optional[bool] = Field(False, description="Include current quotes in results")


# Response Models
class StockQuoteResponse(BaseResponse):
    """Response model for stock quote endpoint"""
    data: StockQuote = Field(..., description="Stock quote data")


class StockDataResponse(BaseResponse):
    """Response model for complete stock data endpoint"""
    data: StockData = Field(..., description="Complete stock data")


class HistoricalDataResponse(BaseResponse):
    """Response model for historical data endpoint"""
    data: List[HistoricalData] = Field(..., description="Historical stock data")
    symbol: str = Field(..., description="Stock symbol")
    timeframe: TimeFrame = Field(..., description="Data timeframe")
    count: int = Field(..., description="Number of data points returned")


class PredictionResponse(BaseResponse):
    """Response model for prediction endpoint"""
    data: EnsemblePrediction = Field(..., description="Stock prediction data")


class SearchResponse(BaseResponse):
    """Response model for stock search endpoint"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    count: int = Field(..., description="Number of results found")


class HealthResponse(BaseResponse):
    """Response model for health check endpoint"""
    status: str = Field("healthy", description="Service health status")
    version: str = Field("2.0.0", description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    database_status: Optional[str] = Field(None, description="Database connection status")
    external_apis_status: Optional[Dict[str, str]] = Field(None, description="External API status")


# Portfolio Models (for future extension)
class PortfolioPosition(BaseModel):
    """Portfolio position model"""
    symbol: str = Field(..., description="Stock symbol")
    shares: int = Field(..., gt=0, description="Number of shares")
    cost_basis: float = Field(..., gt=0, description="Total cost basis")
    current_value: Optional[float] = Field(None, description="Current market value")
    gain_loss: Optional[float] = Field(None, description="Unrealized gain/loss")
    gain_loss_percent: Optional[float] = Field(None, description="Unrealized gain/loss percentage")


class PortfolioAnalysis(BaseModel):
    """Portfolio analysis model"""
    total_value: float = Field(..., description="Total portfolio value")
    total_cost: float = Field(..., description="Total cost basis")
    total_gain_loss: float = Field(..., description="Total gain/loss")
    total_return_percent: float = Field(..., description="Total return percentage")
    positions: List[PortfolioPosition] = Field(..., description="Portfolio positions")
    diversification_score: float = Field(..., ge=0, le=100, description="Diversification score")
    risk_score: float = Field(..., ge=0, le=100, description="Overall risk score")
