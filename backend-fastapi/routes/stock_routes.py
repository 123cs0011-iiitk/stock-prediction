"""
Stock Price Insight Arena - Stock Routes
API routes for stock data operations including quotes, historical data, and company information.
"""

from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import List, Optional
import logging
from datetime import date, datetime

from models.stock_models import (
    StockQuoteRequest, StockDataResponse, HistoricalDataRequest,
    HistoricalDataResponse, SearchRequest, SearchResponse,
    StockQuoteResponse, TimeFrame, Currency
)
from services.stock_service import stock_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=dict)
async def stock_service_health():
    """
    Health check endpoint for stock service
    """
    try:
        health_status = await stock_service.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/quote/{symbol}", response_model=StockQuoteResponse)
async def get_stock_quote(
    symbol: str = Path(..., description="Stock symbol (e.g., AAPL, GOOGL)"),
    currency: Optional[Currency] = Query(Currency.USD, description="Currency for the quote"),
    refresh: Optional[bool] = Query(False, description="Force refresh from data source")
):
    """
    Get real-time stock quote for a symbol
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    - **currency**: Desired currency (USD, EUR, GBP, JPY, INR)
    - **refresh**: Force refresh from data source (default: False)
    
    Returns current stock price, change, volume, and other market data.
    """
    try:
        logger.info(f"Fetching quote for {symbol}")
        
        quote = await stock_service.get_stock_quote(
            symbol=symbol,
            currency=currency,
            refresh=refresh
        )
        
        return StockQuoteResponse(
            success=True,
            message=f"Quote retrieved successfully for {symbol}",
            data=quote
        )
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch stock quote")


@router.get("/data/{symbol}", response_model=StockDataResponse)
async def get_complete_stock_data(
    symbol: str = Path(..., description="Stock symbol"),
    refresh: Optional[bool] = Query(False, description="Force refresh from data source"),
    include_indicators: Optional[bool] = Query(True, description="Include technical indicators")
):
    """
    Get complete stock data including quote, company info, historical data, and technical indicators
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    - **refresh**: Force refresh from data source (default: False)
    - **include_indicators**: Include technical indicators (default: True)
    
    Returns comprehensive stock data including:
    - Real-time quote
    - Company information
    - Historical price data
    - Technical indicators (RSI, SMA, etc.)
    """
    try:
        logger.info(f"Fetching complete data for {symbol}")
        
        stock_data = await stock_service.get_complete_stock_data(
            symbol=symbol,
            refresh=refresh
        )
        
        # Remove technical indicators if not requested
        if not include_indicators:
            stock_data.technical_indicators = None
        
        return StockDataResponse(
            success=True,
            message=f"Complete data retrieved successfully for {symbol}",
            data=stock_data
        )
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching complete data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch complete stock data")


@router.get("/historical/{symbol}", response_model=HistoricalDataResponse)
async def get_historical_data(
    symbol: str = Path(..., description="Stock symbol"),
    timeframe: TimeFrame = Query(TimeFrame.DAILY, description="Data timeframe"),
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of data points")
):
    """
    Get historical stock data
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    - **timeframe**: Data timeframe (1min, 5min, 15min, 30min, 1hour, daily, weekly, monthly)
    - **start_date**: Start date for historical data (YYYY-MM-DD)
    - **end_date**: End date for historical data (YYYY-MM-DD)
    - **limit**: Maximum number of data points (1-1000)
    
    Returns historical price data including OHLC (Open, High, Low, Close) and volume.
    """
    try:
        logger.info(f"Fetching historical data for {symbol}, timeframe: {timeframe}")
        
        historical_data = await stock_service.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        if not historical_data:
            raise HTTPException(
                status_code=404,
                detail=f"No historical data found for {symbol} with the specified parameters"
            )
        
        return HistoricalDataResponse(
            success=True,
            message=f"Historical data retrieved successfully for {symbol}",
            data=historical_data,
            symbol=symbol,
            timeframe=timeframe,
            count=len(historical_data)
        )
        
    except ValueError as e:
        logger.warning(f"Invalid parameters for {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical data")


@router.get("/search", response_model=SearchResponse)
async def search_stocks(
    query: str = Query(..., min_length=1, max_length=100, description="Search query"),
    limit: Optional[int] = Query(10, ge=1, le=50, description="Maximum number of results"),
    include_quotes: Optional[bool] = Query(False, description="Include current quotes in results")
):
    """
    Search for stocks by symbol, company name, or sector
    
    - **query**: Search query (symbol, company name, or sector)
    - **limit**: Maximum number of results (1-50)
    - **include_quotes**: Include current quotes in results (default: False)
    
    Returns list of matching stocks with basic information.
    """
    try:
        logger.info(f"Searching stocks with query: {query}")
        
        results = await stock_service.search_stocks(
            query=query,
            limit=limit,
            include_quotes=include_quotes
        )
        
        return SearchResponse(
            success=True,
            message=f"Search completed successfully",
            results=results,
            query=query,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to search stocks")


@router.get("/company/{symbol}")
async def get_company_info(
    symbol: str = Path(..., description="Stock symbol")
):
    """
    Get detailed company information
    
    - **symbol**: Stock symbol (e.g., AAPL, GOOGL, MSFT)
    
    Returns company information including:
    - Company name and description
    - Sector and industry
    - CEO and headquarters
    - Website and employee count
    """
    try:
        logger.info(f"Fetching company info for {symbol}")
        
        company_info = await stock_service.get_company_info(symbol=symbol)
        
        if not company_info:
            raise HTTPException(
                status_code=404,
                detail=f"Company information not found for {symbol}"
            )
        
        return {
            "success": True,
            "message": f"Company information retrieved successfully for {symbol}",
            "data": company_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching company info for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch company information")


@router.get("/batch-quotes")
async def get_batch_quotes(
    symbols: str = Query(..., description="Comma-separated list of stock symbols"),
    currency: Optional[Currency] = Query(Currency.USD, description="Currency for quotes")
):
    """
    Get quotes for multiple stocks in a single request
    
    - **symbols**: Comma-separated list of stock symbols (e.g., AAPL,GOOGL,MSFT)
    - **currency**: Desired currency (USD, EUR, GBP, JPY, INR)
    
    Returns quotes for all requested symbols.
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        if len(symbol_list) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 symbols allowed per request"
            )
        
        logger.info(f"Fetching batch quotes for {len(symbol_list)} symbols")
        
        quotes = []
        errors = []
        
        for symbol in symbol_list:
            try:
                quote = await stock_service.get_stock_quote(
                    symbol=symbol,
                    currency=currency
                )
                quotes.append(quote)
            except Exception as e:
                errors.append({"symbol": symbol, "error": str(e)})
        
        return {
            "success": True,
            "message": f"Batch quotes retrieved for {len(quotes)} symbols",
            "quotes": quotes,
            "errors": errors,
            "requested_count": len(symbol_list),
            "successful_count": len(quotes),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid symbols: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching batch quotes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch batch quotes")


@router.get("/market-overview")
async def get_market_overview():
    """
    Get market overview with popular stocks
    
    Returns overview of major market indices and popular stocks with current quotes.
    """
    try:
        logger.info("Fetching market overview")
        
        # Popular stocks for overview
        popular_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        
        overview_data = {
            "market_summary": {
                "total_stocks": len(popular_symbols),
                "last_updated": datetime.utcnow().isoformat()
            },
            "popular_stocks": []
        }
        
        for symbol in popular_symbols:
            try:
                quote = await stock_service.get_stock_quote(symbol=symbol)
                overview_data["popular_stocks"].append({
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "change": quote.change,
                    "change_percent": quote.change_percent,
                    "volume": quote.volume
                })
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {str(e)}")
                continue
        
        return {
            "success": True,
            "message": "Market overview retrieved successfully",
            "data": overview_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch market overview")


@router.get("/symbols/available")
async def get_available_symbols():
    """
    Get list of available stock symbols
    
    Returns list of stock symbols available in the system.
    """
    try:
        logger.info("Fetching available symbols")
        
        # Get symbols from sample data (in real implementation, this would come from database)
        available_symbols = list(stock_service.sample_data.keys())
        
        return {
            "success": True,
            "message": "Available symbols retrieved successfully",
            "symbols": available_symbols,
            "count": len(available_symbols),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching available symbols: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch available symbols")
