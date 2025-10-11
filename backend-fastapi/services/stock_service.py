"""
Stock Price Insight Arena - Stock Service
Business logic layer for stock data operations, API integrations, and data processing.
"""

import asyncio
import aiohttp
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, date
import json
import random
import math
from dataclasses import dataclass

from models.stock_models import (
    StockQuote, CompanyInfo, HistoricalData, TechnicalIndicators,
    StockData, TimeFrame, Currency, PredictionModel
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class APIConfig:
    """Configuration for external API"""
    base_url: str
    api_key: str
    rate_limit: int
    timeout: int


class StockService:
    """
    Main service class for stock data operations
    Handles API integrations, data processing, and business logic
    """
    
    def __init__(self):
        self.api_configs = {
            "alpha_vantage": APIConfig(
                base_url="https://www.alphavantage.co/query",
                api_key=settings.ALPHA_VANTAGE_API_KEY or "",
                rate_limit=5,  # 5 calls per minute for free tier
                timeout=settings.API_TIMEOUT
            ),
            "finnhub": APIConfig(
                base_url="https://finnhub.io/api/v1",
                api_key=settings.FINNHUB_API_KEY or "",
                rate_limit=60,  # 60 calls per minute for free tier
                timeout=settings.API_TIMEOUT
            ),
            "yahoo_finance": APIConfig(
                base_url="https://query1.finance.yahoo.com/v8/finance/chart",
                api_key="",  # No API key required
                rate_limit=1000,
                timeout=settings.API_TIMEOUT
            )
        }
        
        # Sample data for testing when APIs are not available
        self.sample_data = self._initialize_sample_data()
    
    def _initialize_sample_data(self) -> Dict[str, Any]:
        """Initialize sample data for testing purposes"""
        return {
            "AAPL": {
                "quote": {
                    "symbol": "AAPL",
                    "price": 175.43,
                    "change": 2.15,
                    "change_percent": 1.24,
                    "volume": 45678900,
                    "high": 176.20,
                    "low": 173.80,
                    "open": 174.50,
                    "previous_close": 173.28,
                    "market_cap": 2750000000000,
                    "currency": "USD",
                    "data_source": "Sample Data"
                },
                "company_info": {
                    "symbol": "AAPL",
                    "name": "Apple Inc.",
                    "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "website": "https://www.apple.com",
                    "employees": 164000,
                    "founded": 1976,
                    "ceo": "Tim Cook",
                    "headquarters": "Cupertino, CA",
                    "market_cap": 2750000000000
                },
                "historical_data": self._generate_sample_historical_data("AAPL", 30)
            },
            "GOOGL": {
                "quote": {
                    "symbol": "GOOGL",
                    "price": 142.56,
                    "change": -1.23,
                    "change_percent": -0.86,
                    "volume": 23456700,
                    "high": 144.20,
                    "low": 141.80,
                    "open": 143.50,
                    "previous_close": 143.79,
                    "market_cap": 1800000000000,
                    "currency": "USD",
                    "data_source": "Sample Data"
                },
                "company_info": {
                    "symbol": "GOOGL",
                    "name": "Alphabet Inc.",
                    "description": "Alphabet Inc. provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America.",
                    "sector": "Technology",
                    "industry": "Internet Content & Information",
                    "website": "https://www.alphabet.com",
                    "employees": 190000,
                    "founded": 1998,
                    "ceo": "Sundar Pichai",
                    "headquarters": "Mountain View, CA",
                    "market_cap": 1800000000000
                },
                "historical_data": self._generate_sample_historical_data("GOOGL", 30)
            },
            "MSFT": {
                "quote": {
                    "symbol": "MSFT",
                    "price": 378.85,
                    "change": 5.23,
                    "change_percent": 1.40,
                    "volume": 18923400,
                    "high": 380.50,
                    "low": 375.20,
                    "open": 376.50,
                    "previous_close": 373.62,
                    "market_cap": 2810000000000,
                    "currency": "USD",
                    "data_source": "Sample Data"
                },
                "company_info": {
                    "symbol": "MSFT",
                    "name": "Microsoft Corporation",
                    "description": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.",
                    "sector": "Technology",
                    "industry": "Software—Infrastructure",
                    "website": "https://www.microsoft.com",
                    "employees": 221000,
                    "founded": 1975,
                    "ceo": "Satya Nadella",
                    "headquarters": "Redmond, WA",
                    "market_cap": 2810000000000
                },
                "historical_data": self._generate_sample_historical_data("MSFT", 30)
            }
        }
    
    def _generate_sample_historical_data(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Generate sample historical data for testing"""
        historical_data = []
        base_price = self.sample_data[symbol]["quote"]["price"]
        current_date = datetime.now().date()
        
        for i in range(days):
            date = current_date - timedelta(days=i)
            
            # Generate realistic price movement
            volatility = random.uniform(-0.05, 0.05)  # ±5% daily volatility
            if i == 0:
                price = base_price
            else:
                price = historical_data[-1]["close"] * (1 + volatility)
            
            open_price = price * random.uniform(0.98, 1.02)
            high_price = max(open_price, price) * random.uniform(1.00, 1.03)
            low_price = min(open_price, price) * random.uniform(0.97, 1.00)
            volume = random.randint(1000000, 100000000)
            
            historical_data.append({
                "date": date.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(price, 2),
                "volume": volume,
                "adjusted_close": round(price, 2)
            })
        
        return list(reversed(historical_data))  # Return in chronological order
    
    async def get_stock_quote(self, symbol: str, currency: Currency = Currency.USD, refresh: bool = False) -> StockQuote:
        """
        Get real-time stock quote for a symbol
        
        Args:
            symbol: Stock symbol (e.g., AAPL)
            currency: Desired currency
            refresh: Force refresh from data source
            
        Returns:
            StockQuote object with current price data
        """
        symbol = symbol.upper().strip()
        logger.info(f"Fetching quote for {symbol}")
        
        try:
            # Try Alpha Vantage first
            if settings.ALPHA_VANTAGE_API_KEY:
                quote = await self._fetch_alpha_vantage_quote(symbol)
                if quote:
                    return quote
            
            # Fallback to sample data for testing
            if symbol in self.sample_data:
                logger.info(f"Using sample data for {symbol}")
                quote_data = self.sample_data[symbol]["quote"].copy()
                quote_data["timestamp"] = datetime.utcnow()
                return StockQuote(**quote_data)
            
            raise ValueError(f"Stock symbol {symbol} not found")
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            raise
    
    async def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """
        Get company information for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            CompanyInfo object or None if not found
        """
        symbol = symbol.upper().strip()
        logger.info(f"Fetching company info for {symbol}")
        
        try:
            if symbol in self.sample_data:
                company_data = self.sample_data[symbol]["company_info"]
                return CompanyInfo(**company_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame = TimeFrame.DAILY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[HistoricalData]:
        """
        Get historical stock data
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of data points
            
        Returns:
            List of HistoricalData objects
        """
        symbol = symbol.upper().strip()
        logger.info(f"Fetching historical data for {symbol}, timeframe: {timeframe}")
        
        try:
            if symbol in self.sample_data:
                historical_data = self.sample_data[symbol]["historical_data"]
                
                # Apply date filters if provided
                if start_date or end_date:
                    filtered_data = []
                    for data_point in historical_data:
                        data_date = datetime.fromisoformat(data_point["date"]).date()
                        
                        if start_date and data_date < start_date:
                            continue
                        if end_date and data_date > end_date:
                            continue
                        
                        filtered_data.append(data_point)
                    
                    historical_data = filtered_data
                
                # Apply limit
                historical_data = historical_data[-limit:]
                
                return [HistoricalData(**data) for data in historical_data]
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return []
    
    async def get_complete_stock_data(self, symbol: str, refresh: bool = False) -> StockData:
        """
        Get complete stock data including quote, company info, and historical data
        
        Args:
            symbol: Stock symbol
            refresh: Force refresh from data source
            
        Returns:
            StockData object with all available data
        """
        symbol = symbol.upper().strip()
        logger.info(f"Fetching complete data for {symbol}")
        
        try:
            # Fetch all data concurrently
            quote_task = self.get_stock_quote(symbol, refresh=refresh)
            company_task = self.get_company_info(symbol)
            historical_task = self.get_historical_data(symbol, limit=30)
            
            quote, company_info, historical_data = await asyncio.gather(
                quote_task, company_task, historical_task
            )
            
            # Calculate technical indicators if historical data is available
            technical_indicators = None
            if historical_data:
                technical_indicators = self._calculate_technical_indicators(historical_data)
            
            return StockData(
                symbol=symbol,
                quote=quote,
                company_info=company_info,
                historical_data=historical_data,
                technical_indicators=technical_indicators
            )
            
        except Exception as e:
            logger.error(f"Error fetching complete data for {symbol}: {str(e)}")
            raise
    
    async def search_stocks(self, query: str, limit: int = 10, include_quotes: bool = False) -> List[Dict[str, Any]]:
        """
        Search for stocks by symbol or company name
        
        Args:
            query: Search query
            limit: Maximum number of results
            include_quotes: Include current quotes in results
            
        Returns:
            List of stock search results
        """
        query = query.upper().strip()
        logger.info(f"Searching stocks with query: {query}")
        
        try:
            results = []
            
            # Search in sample data
            for symbol, data in self.sample_data.items():
                if (query in symbol or 
                    query in data["company_info"]["name"].upper() or
                    query in data["company_info"]["sector"].upper()):
                    
                    result = {
                        "symbol": symbol,
                        "name": data["company_info"]["name"],
                        "sector": data["company_info"]["sector"],
                        "industry": data["company_info"]["industry"],
                        "market_cap": data["company_info"]["market_cap"]
                    }
                    
                    if include_quotes:
                        result["quote"] = data["quote"]
                    
                    results.append(result)
            
            # Sort by market cap (largest first)
            results.sort(key=lambda x: x.get("market_cap", 0), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            return []
    
    def _calculate_technical_indicators(self, historical_data: List[HistoricalData]) -> TechnicalIndicators:
        """
        Calculate technical indicators from historical data
        
        Args:
            historical_data: List of historical price data
            
        Returns:
            TechnicalIndicators object with calculated indicators
        """
        if len(historical_data) < 20:
            return TechnicalIndicators()
        
        prices = [data.close for data in historical_data]
        volumes = [data.volume for data in historical_data]
        
        # Calculate moving averages
        sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else prices[-1]
        sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else prices[-1]
        
        # Calculate RSI (simplified)
        rsi = self._calculate_rsi(prices)
        
        # Calculate volatility
        if len(prices) > 1:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = math.sqrt(sum([r**2 for r in returns]) / len(returns)) * math.sqrt(252) * 100
        else:
            volatility = 0
        
        # Calculate volume SMA
        volume_sma = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        
        return TechnicalIndicators(
            sma_20=round(sma_20, 2),
            sma_50=round(sma_50, 2),
            rsi=round(rsi, 2),
            volatility=round(volatility, 2),
            volume_sma=round(volume_sma, 2)
        )
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: List of price values
            period: RSI period (default 14)
            
        Returns:
            RSI value
        """
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _fetch_alpha_vantage_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Fetch stock quote from Alpha Vantage API
        
        Args:
            symbol: Stock symbol
            
        Returns:
            StockQuote object or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": settings.ALPHA_VANTAGE_API_KEY
                }
                
                async with session.get(
                    self.api_configs["alpha_vantage"].base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.api_configs["alpha_vantage"].timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Global Quote" in data:
                            quote_data = data["Global Quote"]
                            
                            return StockQuote(
                                symbol=symbol,
                                price=float(quote_data.get("05. price", 0)),
                                change=float(quote_data.get("09. change", 0)),
                                change_percent=float(quote_data.get("10. change percent", "0%").replace("%", "")),
                                volume=int(quote_data.get("06. volume", 0)),
                                high=float(quote_data.get("03. high", 0)),
                                low=float(quote_data.get("04. low", 0)),
                                open=float(quote_data.get("02. open", 0)),
                                previous_close=float(quote_data.get("08. previous close", 0)),
                                currency=Currency.USD,
                                data_source="Alpha Vantage"
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage quote for {symbol}: {str(e)}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for the service
        
        Returns:
            Dictionary with health status information
        """
        health_status = {
            "service": "StockService",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "data_sources": {}
        }
        
        # Check API configurations
        for source, config in self.api_configs.items():
            if source == "alpha_vantage":
                health_status["data_sources"][source] = {
                    "configured": bool(config.api_key),
                    "available": bool(config.api_key)
                }
            else:
                health_status["data_sources"][source] = {
                    "configured": bool(config.api_key) if config.api_key else True,
                    "available": True
                }
        
        # Check sample data
        health_status["sample_data"] = {
            "available": bool(self.sample_data),
            "symbols": list(self.sample_data.keys())
        }
        
        return health_status


# Create singleton instance
stock_service = StockService()
