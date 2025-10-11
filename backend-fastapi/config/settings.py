"""
Stock Price Insight Arena - Configuration Settings
Application settings and environment variable management using Pydantic Settings.
"""

from pydantic import BaseSettings, Field
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application settings
    APP_NAME: str = Field("Stock Price Insight Arena API", description="Application name")
    VERSION: str = Field("2.0.0", description="Application version")
    ENVIRONMENT: str = Field("development", description="Environment (development, staging, production)")
    DEBUG: bool = Field(True, description="Debug mode")
    
    # Server settings
    HOST: str = Field("0.0.0.0", description="Server host")
    PORT: int = Field(8000, description="Server port")
    
    # Security settings
    SECRET_KEY: str = Field("your-secret-key-here-change-in-production", description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, description="Access token expiration time in minutes")
    ALLOWED_HOSTS: List[str] = Field(["*"], description="Allowed hosts for TrustedHostMiddleware")
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
        description="CORS allowed origins"
    )
    
    # Database settings
    DATABASE_URL: Optional[str] = Field(None, description="Database connection URL")
    DB_HOST: str = Field("localhost", description="Database host")
    DB_PORT: int = Field(5432, description="Database port")
    DB_NAME: str = Field("stock_prediction", description="Database name")
    DB_USER: str = Field("postgres", description="Database user")
    DB_PASSWORD: str = Field("", description="Database password")
    
    # API Keys
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None, description="Alpha Vantage API key")
    FINNHUB_API_KEY: Optional[str] = Field(None, description="Finnhub API key")
    POLYGON_API_KEY: Optional[str] = Field(None, description="Polygon.io API key")
    YAHOO_FINANCE_API_KEY: Optional[str] = Field(None, description="Yahoo Finance API key")
    
    # Cache settings
    REDIS_URL: Optional[str] = Field(None, description="Redis connection URL")
    CACHE_TTL: int = Field(300, description="Cache time-to-live in seconds")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(60, description="Rate limit per minute")
    RATE_LIMIT_BURST: int = Field(100, description="Rate limit burst allowance")
    
    # External API settings
    API_TIMEOUT: int = Field(30, description="External API timeout in seconds")
    API_RETRY_ATTEMPTS: int = Field(3, description="Number of retry attempts for failed API calls")
    API_RETRY_DELAY: int = Field(1, description="Delay between retry attempts in seconds")
    
    # Logging settings
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    LOG_FORMAT: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # ML Model settings
    ML_MODEL_PATH: str = Field("./models", description="Path to store ML models")
    PREDICTION_CACHE_TTL: int = Field(600, description="Prediction cache TTL in seconds")
    MIN_HISTORICAL_DATA_POINTS: int = Field(50, description="Minimum historical data points for ML training")
    
    # Data source priorities
    PRIMARY_DATA_SOURCE: str = Field("alpha_vantage", description="Primary data source")
    FALLBACK_DATA_SOURCES: List[str] = Field(
        ["yahoo_finance", "finnhub", "polygon"],
        description="Fallback data sources in order of preference"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings
    This function is cached to avoid reloading settings on every request
    """
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


class StagingSettings(Settings):
    """Staging environment settings"""
    ENVIRONMENT: str = "staging"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


class ProductionSettings(Settings):
    """Production environment settings"""
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    ALLOWED_ORIGINS: List[str] = ["https://yourdomain.com"]  # Update with actual domain


def get_environment_settings() -> Settings:
    """
    Get environment-specific settings based on ENVIRONMENT variable
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "staging":
        return StagingSettings()
    else:
        return DevelopmentSettings()


# Example environment file content
ENV_EXAMPLE = """
# Application Settings
APP_NAME=Stock Price Insight Arena API
VERSION=2.0.0
ENVIRONMENT=development
DEBUG=true

# Server Settings
HOST=0.0.0.0
PORT=8000

# Security Settings
SECRET_KEY=your-super-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALLOWED_HOSTS=["*"]
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:3001"]

# Database Settings
DATABASE_URL=postgresql://postgres:password@localhost:5432/stock_prediction
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_prediction
DB_USER=postgres
DB_PASSWORD=your_password_here

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here

# Cache Settings
REDIS_URL=redis://localhost:6379
CACHE_TTL=300

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=100

# External API Settings
API_TIMEOUT=30
API_RETRY_ATTEMPTS=3
API_RETRY_DELAY=1

# Logging Settings
LOG_LEVEL=INFO

# ML Model Settings
ML_MODEL_PATH=./models
PREDICTION_CACHE_TTL=600
MIN_HISTORICAL_DATA_POINTS=50

# Data Source Settings
PRIMARY_DATA_SOURCE=alpha_vantage
FALLBACK_DATA_SOURCES=["yahoo_finance","finnhub","polygon"]
"""
