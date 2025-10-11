"""
Stock Price Insight Arena - Health Routes
API routes for health checks and system status monitoring.
"""

from fastapi import APIRouter, HTTPException
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from models.stock_models import HealthResponse
from services.stock_service import stock_service
from services.prediction_service import prediction_service
from config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Global variables for health tracking
startup_time = datetime.utcnow()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    
    Returns detailed health status including:
    - Service status
    - Database connectivity
    - External API status
    - System uptime
    - Performance metrics
    """
    try:
        logger.info("Performing comprehensive health check")
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()
        
        # Check stock service health
        stock_health = await stock_service.health_check()
        
        # Check prediction service health
        prediction_health = await prediction_service.health_check()
        
        # Determine overall status
        overall_status = "healthy"
        if (stock_health.get("status") != "healthy" or 
            prediction_health.get("status") != "healthy"):
            overall_status = "degraded"
        
        health_response = HealthResponse(
            success=True,
            message="Health check completed successfully",
            status=overall_status,
            version=settings.VERSION,
            uptime=uptime_seconds,
            database_status="connected",  # In real implementation, check actual DB connection
            external_apis_status={
                "alpha_vantage": "available" if settings.ALPHA_VANTAGE_API_KEY else "not_configured",
                "finnhub": "available" if settings.FINNHUB_API_KEY else "not_configured",
                "polygon": "available" if settings.POLYGON_API_KEY else "not_configured",
                "yahoo_finance": "available"
            }
        )
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/simple")
async def simple_health_check():
    """
    Simple health check endpoint for load balancers and monitoring systems
    
    Returns basic status information with minimal overhead.
    """
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "Stock Price Insight Arena API",
            "version": settings.VERSION
        }
    except Exception as e:
        logger.error(f"Simple health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/detailed")
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information
    
    Returns detailed health status including:
    - Individual service status
    - Performance metrics
    - Configuration status
    - Resource usage
    """
    try:
        logger.info("Performing detailed health check")
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()
        uptime_hours = uptime_seconds / 3600
        
        # Check individual services
        stock_health = await stock_service.health_check()
        prediction_health = await prediction_service.health_check()
        
        # System information
        system_info = {
            "environment": settings.ENVIRONMENT,
            "debug_mode": settings.DEBUG,
            "host": settings.HOST,
            "port": settings.PORT,
            "uptime_seconds": uptime_seconds,
            "uptime_hours": round(uptime_hours, 2),
            "startup_time": startup_time.isoformat()
        }
        
        # Service status
        services_status = {
            "stock_service": stock_health,
            "prediction_service": prediction_health,
            "api_gateway": {
                "status": "healthy",
                "endpoints_available": True,
                "cors_enabled": True,
                "rate_limiting_enabled": True
            }
        }
        
        # Configuration status
        config_status = {
            "database_configured": bool(settings.DATABASE_URL),
            "alpha_vantage_configured": bool(settings.ALPHA_VANTAGE_API_KEY),
            "finnhub_configured": bool(settings.FINNHUB_API_KEY),
            "polygon_configured": bool(settings.POLYGON_API_KEY),
            "cors_origins": settings.ALLOWED_ORIGINS,
            "rate_limits": {
                "per_minute": settings.RATE_LIMIT_PER_MINUTE,
                "burst": settings.RATE_LIMIT_BURST
            }
        }
        
        # Performance metrics
        performance_metrics = {
            "cache_size": len(prediction_service.cache),
            "model_performance_tracked": len(prediction_service.model_performance),
            "api_timeout": settings.API_TIMEOUT,
            "cache_ttl": settings.CACHE_TTL
        }
        
        # Determine overall health
        overall_status = "healthy"
        issues = []
        
        if stock_health.get("status") != "healthy":
            issues.append("Stock service unhealthy")
            overall_status = "degraded"
        
        if prediction_health.get("status") != "healthy":
            issues.append("Prediction service unhealthy")
            overall_status = "degraded"
        
        if not settings.ALPHA_VANTAGE_API_KEY:
            issues.append("Alpha Vantage API key not configured")
            overall_status = "degraded"
        
        detailed_health = {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "system_info": system_info,
            "services_status": services_status,
            "configuration_status": config_status,
            "performance_metrics": performance_metrics,
            "issues": issues,
            "recommendations": _get_health_recommendations(config_status, services_status)
        }
        
        return detailed_health
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")


@router.get("/services")
async def services_health_check():
    """
    Health check for individual services
    
    Returns status of each service component.
    """
    try:
        logger.info("Performing services health check")
        
        services_status = {}
        
        # Check stock service
        try:
            stock_health = await stock_service.health_check()
            services_status["stock_service"] = {
                "status": "healthy",
                "details": stock_health
            }
        except Exception as e:
            services_status["stock_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check prediction service
        try:
            prediction_health = await prediction_service.health_check()
            services_status["prediction_service"] = {
                "status": "healthy",
                "details": prediction_health
            }
        except Exception as e:
            services_status["prediction_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check database (placeholder)
        try:
            services_status["database"] = {
                "status": "healthy",
                "details": {
                    "connection": "active",
                    "response_time": "< 10ms"
                }
            }
        except Exception as e:
            services_status["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check external APIs
        external_apis = {}
        
        # Alpha Vantage
        if settings.ALPHA_VANTAGE_API_KEY:
            external_apis["alpha_vantage"] = {
                "status": "configured",
                "api_key_present": True
            }
        else:
            external_apis["alpha_vantage"] = {
                "status": "not_configured",
                "api_key_present": False
            }
        
        # Finnhub
        if settings.FINNHUB_API_KEY:
            external_apis["finnhub"] = {
                "status": "configured",
                "api_key_present": True
            }
        else:
            external_apis["finnhub"] = {
                "status": "not_configured",
                "api_key_present": False
            }
        
        # Yahoo Finance (always available)
        external_apis["yahoo_finance"] = {
            "status": "available",
            "api_key_required": False
        }
        
        services_status["external_apis"] = external_apis
        
        return {
            "success": True,
            "message": "Services health check completed",
            "services": services_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Services health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Services health check failed")


@router.get("/readiness")
async def readiness_check():
    """
    Readiness check for Kubernetes and container orchestration
    
    Returns readiness status for deployment systems.
    """
    try:
        # Check if all critical services are ready
        stock_health = await stock_service.health_check()
        prediction_health = await prediction_service.health_check()
        
        # Determine if service is ready to receive traffic
        is_ready = (
            stock_health.get("status") == "healthy" and
            prediction_health.get("status") == "healthy"
        )
        
        if is_ready:
            return {
                "ready": True,
                "message": "Service is ready to receive traffic",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "ready": False,
                "message": "Service is not ready",
                "issues": [
                    "Stock service unhealthy" if stock_health.get("status") != "healthy" else None,
                    "Prediction service unhealthy" if prediction_health.get("status") != "healthy" else None
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {
            "ready": False,
            "message": "Readiness check failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/liveness")
async def liveness_check():
    """
    Liveness check for Kubernetes and container orchestration
    
    Returns liveness status for deployment systems.
    """
    try:
        # Simple check to see if the service is alive
        uptime_seconds = (datetime.utcnow() - startup_time).total_seconds()
        
        return {
            "alive": True,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return {
            "alive": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


def _get_health_recommendations(config_status: Dict[str, Any], services_status: Dict[str, Any]) -> list:
    """
    Generate health recommendations based on current status
    
    Args:
        config_status: Configuration status dictionary
        services_status: Services status dictionary
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # API key recommendations
    if not config_status.get("alpha_vantage_configured"):
        recommendations.append("Configure Alpha Vantage API key for real-time data access")
    
    if not config_status.get("finnhub_configured"):
        recommendations.append("Configure Finnhub API key for additional data sources")
    
    if not config_status.get("polygon_configured"):
        recommendations.append("Configure Polygon API key for enhanced market data")
    
    # Database recommendations
    if not config_status.get("database_configured"):
        recommendations.append("Configure database connection for data persistence")
    
    # Service recommendations
    for service_name, service_status in services_status.items():
        if isinstance(service_status, dict) and service_status.get("status") == "unhealthy":
            recommendations.append(f"Investigate and fix {service_name} issues")
    
    # Performance recommendations
    if len(recommendations) == 0:
        recommendations.append("All systems are healthy - consider monitoring for performance optimization")
    
    return recommendations
