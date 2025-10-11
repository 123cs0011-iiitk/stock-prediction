"""
Stock Price Insight Arena - FastAPI Backend
Main application entry point with FastAPI setup, CORS, and middleware configuration.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from routes.stock_routes import router as stock_router
from routes.prediction_routes import router as prediction_router
from routes.health_routes import router as health_router
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    - Startup: Initialize services, connect to databases, etc.
    - Shutdown: Cleanup resources, close connections, etc.
    """
    # Startup
    logger.info("🚀 Starting Stock Price Insight Arena API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Stock Price Insight Arena API")


# Create FastAPI application instance
app = FastAPI(
    title="Stock Price Insight Arena API",
    description="""
    ## 🏆 Stock Price Insight Arena API
    
    A comprehensive API for real-time stock data, historical analysis, and ML-powered price predictions.
    
    ### Features:
    - 📈 Real-time stock prices from multiple sources
    - 📊 Historical stock data with technical indicators
    - 🤖 Machine learning price predictions
    - 📰 Stock news and market sentiment
    - 📋 Portfolio analysis and management
    
    ### Data Sources:
    - Alpha Vantage API
    - Yahoo Finance
    - Finnhub API
    - Polygon.io
    
    ### ML Models:
    - Linear Regression
    - Random Forest Regressor
    - ARIMA Time Series
    - Ensemble Methods
    """,
    version="2.0.0",
    contact={
        "name": "Ankit Kumar",
        "url": "https://github.com/123cs0011-iiitk",
        "email": "ankit@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors
    """
    logger.error(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Custom HTTP exception handler
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )


# Include API routers
app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health"]
)

app.include_router(
    stock_router,
    prefix="/api/v1",
    tags=["Stocks"]
)

app.include_router(
    prediction_router,
    prefix="/api/v1",
    tags=["Predictions"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information and available endpoints
    """
    return {
        "message": "🏆 Welcome to Stock Price Insight Arena API",
        "version": "2.0.0",
        "description": "Real-time stock data and ML-powered predictions",
        "documentation": "/docs",
        "health_check": "/api/v1/health",
        "endpoints": {
            "stocks": "/api/v1/stocks",
            "predictions": "/api/v1/predictions",
            "health": "/api/v1/health"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check endpoint (simple)
@app.get("/health", tags=["Health"])
async def simple_health_check():
    """
    Simple health check endpoint for load balancers
    """
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup event
    """
    logger.info("✅ Stock Price Insight Arena API is ready!")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event
    """
    logger.info("👋 Stock Price Insight Arena API is shutting down")


if __name__ == "__main__":
    """
    Run the FastAPI application with Uvicorn
    """
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )
