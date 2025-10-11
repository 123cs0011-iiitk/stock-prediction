"""
Multi-Source Data Manager - Intelligent data fetching with 4 data sources
Combines Yahoo Finance, Finnhub, Polygon.io, and Alpha Vantage with intelligent fallback
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import logging
from dotenv import load_dotenv

# Import all data sources
from services.yahoo_finance import yahoo_finance
from services.finnhub_api import finnhub_api
from services.polygon_api import polygon_api
from services.alpha_vantage import alpha_vantage
from services.postgresql_database import postgres_db
from services.csv_backup_service import csv_backup

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSourceDataManager:
    """Intelligent data manager with 4 data sources and smart fallback"""
    
    def __init__(self):
        self.cache_duration_minutes = 5  # Cache live quotes for 5 minutes
        self.historical_cache_days = 1   # Cache historical data for 1 day
        self.company_info_cache_days = 7 # Cache company info for 7 days
        
        # Get data source priority from environment
        priority_str = os.getenv('DATA_SOURCE_PRIORITY', 'yahoo,finnhub,polygon,alpha_vantage')
        self.data_sources = [source.strip() for source in priority_str.split(',')]
        
        # API status tracking
        self.api_status = {
            'yahoo': True,
            'finnhub': True,
            'polygon': True,
            'alpha_vantage': True
        }
        
        logger.info(f"Multi-source data manager initialized with priority: {self.data_sources}")
    
    def _get_data_source_client(self, source: str):
        """Get the appropriate data source client"""
        clients = {
            'yahoo': yahoo_finance,
            'finnhub': finnhub_api,
            'polygon': polygon_api,
            'alpha_vantage': alpha_vantage
        }
        return clients.get(source)
    
    def _mark_api_failed(self, source: str):
        """Mark an API as failed for a short period"""
        self.api_status[source] = False
        logger.warning(f"Marked {source} API as failed")
        
        # Reset after 5 minutes
        import threading
        def reset_api():
            import time
            time.sleep(300)  # 5 minutes
            self.api_status[source] = True
            logger.info(f"Reset {source} API status to active")
        
        thread = threading.Thread(target=reset_api, daemon=True)
        thread.start()
    
    def get_live_quote(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get live stock quote with intelligent multi-source fallback
        """
        try:
            symbol = symbol.upper()
            
            # Check PostgreSQL cache first (unless force refresh)
            if not force_refresh:
                try:
                    cached_quote = postgres_db.get_latest_quote(symbol, self.cache_duration_minutes)
                    if cached_quote:
                        logger.info(f"Using PostgreSQL cached quote for {symbol}")
                        return self._format_quote_response(cached_quote)
                except Exception as e:
                    logger.warning(f"PostgreSQL cache check failed for {symbol}: {e}")
                
                # Fallback to CSV backup cache
                try:
                    cached_quote = csv_backup.get_latest_quote(symbol, self.cache_duration_minutes)
                    if cached_quote:
                        logger.info(f"Using CSV backup cached quote for {symbol}")
                        return self._format_quote_response(cached_quote)
                except Exception as e:
                    logger.warning(f"CSV backup cache check failed for {symbol}: {e}")
            
            # Try each data source in priority order
            for source in self.data_sources:
                if not self.api_status.get(source, True):
                    logger.info(f"Skipping {source} API (marked as failed)")
                    continue
                
                try:
                    logger.info(f"Fetching live quote for {symbol} from {source}")
                    client = self._get_data_source_client(source)
                    
                    if source == 'yahoo':
                        quote = client.get_live_quote(symbol)
                    elif source == 'finnhub':
                        quote = client.get_quote(symbol)
                    elif source == 'polygon':
                        quote = client.get_previous_close(symbol)
                    elif source == 'alpha_vantage':
                        quote = client.get_global_quote(symbol)
                    else:
                        continue
                    
                    if quote:
                        # Store in both PostgreSQL and CSV backup
                        try:
                            postgres_db.store_stock_quote(symbol, quote, source)
                            logger.info(f"Stored quote for {symbol} in PostgreSQL")
                        except Exception as e:
                            logger.warning(f"Failed to store quote for {symbol} in PostgreSQL: {e}")
                        
                        try:
                            csv_backup.store_stock_quote(symbol, quote, source)
                            logger.info(f"Stored quote for {symbol} in CSV backup")
                        except Exception as e:
                            logger.warning(f"Failed to store quote for {symbol} in CSV backup: {e}")
                        
                        logger.info(f"Successfully fetched and stored quote for {symbol} from {source}")
                        return self._format_quote_response(quote)
                        
                except Exception as e:
                    logger.warning(f"{source} API failed for {symbol}: {e}")
                    self._mark_api_failed(source)
                    continue
            
            # Return cached data if available (even if expired)
            if not force_refresh:
                # Try PostgreSQL first
                try:
                    cached_quote = postgres_db.get_latest_quote(symbol, 1440)  # 24 hours
                    if cached_quote:
                        logger.info(f"Using expired PostgreSQL cached quote for {symbol}")
                        return self._format_quote_response(cached_quote)
                except Exception as e:
                    logger.warning(f"PostgreSQL expired cache check failed for {symbol}: {e}")
                
                # Fallback to CSV backup
                try:
                    cached_quote = csv_backup.get_latest_quote(symbol, 1440)  # 24 hours
                    if cached_quote:
                        logger.info(f"Using expired CSV backup cached quote for {symbol}")
                        return self._format_quote_response(cached_quote)
                except Exception as e:
                    logger.warning(f"CSV backup expired cache check failed for {symbol}: {e}")
            
            logger.error(f"Failed to fetch quote for {symbol} from all sources")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_live_quote for {symbol}: {e}")
            return None
    
    def get_company_info(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get company information with multi-source fallback
        """
        try:
            symbol = symbol.upper()
            
            # Check database cache first
            if not force_refresh:
                cached_info = postgres_db.get_company_info(symbol)
                if cached_info:
                    # Check if data is fresh enough
                    last_updated = datetime.fromisoformat(cached_info['last_updated'])
                    if datetime.now() - last_updated < timedelta(days=self.company_info_cache_days):
                        logger.info(f"Using cached company info for {symbol}")
                        return cached_info
            
            # Try each data source in priority order
            for source in self.data_sources:
                if not self.api_status.get(source, True):
                    continue
                
                try:
                    logger.info(f"Fetching company info for {symbol} from {source}")
                    client = self._get_data_source_client(source)
                    
                    if source == 'yahoo':
                        company_info = client.get_company_info(symbol)
                    elif source == 'finnhub':
                        company_info = client.get_company_profile(symbol)
                    elif source == 'polygon':
                        company_info = client.get_ticker_details(symbol)
                    elif source == 'alpha_vantage':
                        company_info = client.get_company_overview(symbol)
                    else:
                        continue
                    
                    if company_info:
                        postgres_db.store_company_info(symbol, company_info, source)
                        logger.info(f"Successfully fetched and stored company info for {symbol}")
                        return company_info
                        
                except Exception as e:
                    logger.warning(f"{source} API failed for company info {symbol}: {e}")
                    self._mark_api_failed(source)
                    continue
            
            # Return cached data if available
            if not force_refresh:
                cached_info = postgres_db.get_company_info(symbol)
                if cached_info:
                    logger.info(f"Using cached company info for {symbol}")
                    return cached_info
            
            logger.error(f"Failed to fetch company info for {symbol} from all sources")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_company_info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y", force_refresh: bool = False) -> Optional[List[Dict]]:
        """
        Get historical data with multi-source fallback
        """
        try:
            symbol = symbol.upper()
            
            # Check database cache first
            if not force_refresh:
                cached_historical = postgres_db.get_historical_data(symbol, 365)
                if cached_historical:
                    # Check if we have recent data
                    latest_date = datetime.strptime(cached_historical[0]['date'], '%Y-%m-%d')
                    if datetime.now() - latest_date < timedelta(days=self.historical_cache_days):
                        logger.info(f"Using cached historical data for {symbol}")
                        return self._format_historical_response(cached_historical)
            
            # Try each data source in priority order
            for source in self.data_sources:
                if not self.api_status.get(source, True):
                    continue
                
                try:
                    logger.info(f"Fetching historical data for {symbol} from {source}")
                    client = self._get_data_source_client(source)
                    
                    historical_data = None
                    
                    if source == 'yahoo':
                        historical_data = client.get_historical_data(symbol, period)
                    elif source == 'finnhub':
                        # Convert period to count
                        count = 100 if period == "1y" else 30 if period == "1mo" else 7
                        historical_data = client.get_candles(symbol, 'D', count)
                    elif source == 'polygon':
                        historical_data = client.get_aggregates(symbol, 1, 'day', limit=100)
                    elif source == 'alpha_vantage':
                        alpha_data = client.get_daily_time_series(symbol, 'compact')
                        if alpha_data and 'historical_data' in alpha_data:
                            historical_data = alpha_data['historical_data']
                    else:
                        continue
                    
                    if historical_data:
                        postgres_db.store_historical_data(symbol, historical_data, source)
                        logger.info(f"Successfully fetched and stored historical data for {symbol}")
                        return self._format_historical_response(historical_data)
                        
                except Exception as e:
                    logger.warning(f"{source} API failed for historical data {symbol}: {e}")
                    self._mark_api_failed(source)
                    continue
            
            # Return cached data if available
            if not force_refresh:
                cached_historical = postgres_db.get_historical_data(symbol, 365)
                if cached_historical:
                    logger.info(f"Using cached historical data for {symbol}")
                    return self._format_historical_response(cached_historical)
            
            logger.error(f"Failed to fetch historical data for {symbol} from all sources")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {e}")
            return None
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks using multiple sources
        """
        try:
            all_results = []
            
            # Try each data source
            for source in self.data_sources:
                if not self.api_status.get(source, True):
                    continue
                
                try:
                    client = self._get_data_source_client(source)
                    
                    if source == 'yahoo':
                        results = client.search_stocks(query)
                    elif source == 'finnhub':
                        results = client.search_stocks(query)
                    elif source == 'polygon':
                        results = client.search_tickers(query)
                    elif source == 'alpha_vantage':
                        # Alpha Vantage doesn't have search, skip
                        continue
                    else:
                        continue
                    
                    if results:
                        all_results.extend(results)
                        
                except Exception as e:
                    logger.warning(f"{source} search failed for {query}: {e}")
                    self._mark_api_failed(source)
                    continue
            
            # Remove duplicates and limit results
            seen_symbols = set()
            unique_results = []
            
            for result in all_results:
                symbol = result.get('symbol', '').upper()
                if symbol and symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    unique_results.append(result)
                    if len(unique_results) >= 20:  # Limit to 20 results
                        break
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in search_stocks: {e}")
            return []
    
    def get_comprehensive_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get comprehensive stock data from multiple sources
        """
        try:
            symbol = symbol.upper()
            
            # Get all data
            quote = self.get_live_quote(symbol, force_refresh)
            company_info = self.get_company_info(symbol, force_refresh)
            historical = self.get_historical_data(symbol, "1y", force_refresh)
            
            if not quote:
                return None
            
            # Combine all data
            comprehensive_data = {
                'symbol': symbol,
                'quote': quote,
                'company_info': company_info,
                'historical_data': historical,
                'data_sources': {
                    'quote_source': quote.get('source', 'unknown'),
                    'company_source': company_info.get('source', 'unknown') if company_info else 'unknown',
                    'historical_source': 'cached' if historical else 'unknown'
                },
                'api_status': self.api_status.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error in get_comprehensive_stock_data for {symbol}: {e}")
            return None
    
    def get_news_data(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get news data from multiple sources
        """
        try:
            symbol = symbol.upper()
            all_news = []
            
            # Try each data source for news
            for source in self.data_sources:
                if not self.api_status.get(source, True):
                    continue
                
                try:
                    client = self._get_data_source_client(source)
                    
                    if source == 'finnhub':
                        news = client.get_company_news(symbol, days=7)
                        news_items = [{
                            'headline': item.get('headline', ''),
                            'summary': item.get('summary', ''),
                            'url': item.get('url', ''),
                            'datetime': item.get('datetime', 0),
                            'source': item.get('source', ''),
                            'category': item.get('category', ''),
                            'api_source': 'finnhub'
                        } for item in news[:limit]]
                        all_news.extend(news_items)
                        
                    elif source == 'polygon':
                        news = client.get_ticker_news(symbol, limit)
                        news_items = [{
                            'headline': item.get('title', ''),
                            'summary': item.get('description', ''),
                            'url': item.get('article_url', ''),
                            'datetime': item.get('published_utc', ''),
                            'source': item.get('publisher', ''),
                            'category': 'general',
                            'api_source': 'polygon'
                        } for item in news[:limit]]
                        all_news.extend(news_items)
                        
                except Exception as e:
                    logger.warning(f"{source} news API failed for {symbol}: {e}")
                    continue
            
            # Sort by datetime and limit results
            all_news.sort(key=lambda x: x.get('datetime', 0), reverse=True)
            return all_news[:limit]
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return []
    
    def _format_quote_response(self, quote_data: Dict) -> Dict:
        """Format quote data for consistent API response"""
        return {
            'symbol': quote_data.get('symbol', ''),
            'name': quote_data.get('name', f"{quote_data.get('symbol', '')} Corporation"),
            'price': quote_data.get('price', 0),
            'change': quote_data.get('change', 0),
            'changePercent': quote_data.get('change_percent', 0),
            'volume': quote_data.get('volume', 0),
            'high': quote_data.get('high', 0),
            'low': quote_data.get('low', 0),
            'open': quote_data.get('open', 0),
            'previousClose': quote_data.get('previous_close', 0),
            'marketCap': quote_data.get('market_cap', 'N/A'),
            'sector': quote_data.get('sector', 'N/A'),
            'industry': quote_data.get('industry', 'N/A'),
            'currency': quote_data.get('currency', 'USD'),
            'timestamp': quote_data.get('timestamp', datetime.now().isoformat()),
            'source': quote_data.get('source', 'unknown')
        }
    
    def _format_historical_response(self, historical_data: List[Dict]) -> List[Dict]:
        """Format historical data for consistent API response"""
        return historical_data
    
    def get_data_statistics(self) -> Dict:
        """Get comprehensive statistics about data usage and storage"""
        try:
            db_stats = postgres_db.get_database_stats()
            
            return {
                'database_stats': db_stats,
                'cache_settings': {
                    'quote_cache_minutes': self.cache_duration_minutes,
                    'historical_cache_days': self.historical_cache_days,
                    'company_info_cache_days': self.company_info_cache_days
                },
                'data_sources': {
                    'available_sources': ['Yahoo Finance', 'Finnhub', 'Polygon.io', 'Alpha Vantage'],
                    'priority_order': self.data_sources,
                    'storage': 'PostgreSQL Database'
                },
                'api_status': self.api_status.copy(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
    
    def cleanup_old_data(self):
        """Clean up old data from database"""
        try:
            postgres_db.cleanup_old_data(days_to_keep=30)
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    def health_check(self) -> Dict:
        """Comprehensive health check for all services"""
        try:
            health_status = {
                'database': postgres_db.health_check(),
                'api_sources': {},
                'overall_status': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
            
            # Check each API source
            for source in self.data_sources:
                try:
                    client = self._get_data_source_client(source)
                    if source == 'yahoo':
                        # Test with a simple symbol
                        test_quote = client.get_live_quote('AAPL')
                        health_status['api_sources'][source] = {
                            'status': 'healthy' if test_quote else 'no_data',
                            'api_key_configured': True
                        }
                    elif source == 'finnhub':
                        health_status['api_sources'][source] = {
                            'status': 'healthy' if self.api_status.get(source, True) else 'failed',
                            'api_key_configured': bool(os.getenv('FINNHUB_API_KEY'))
                        }
                    elif source == 'polygon':
                        health_status['api_sources'][source] = {
                            'status': 'healthy' if self.api_status.get(source, True) else 'failed',
                            'api_key_configured': bool(os.getenv('POLYGON_API_KEY'))
                        }
                    elif source == 'alpha_vantage':
                        health_status['api_sources'][source] = {
                            'status': 'healthy' if self.api_status.get(source, True) else 'failed',
                            'api_key_configured': bool(os.getenv('ALPHA_VANTAGE_API_KEY'))
                        }
                        
                except Exception as e:
                    health_status['api_sources'][source] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Determine overall status
            if health_status['database']['status'] != 'healthy':
                health_status['overall_status'] = 'unhealthy'
            elif any(api['status'] == 'error' for api in health_status['api_sources'].values()):
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global multi-source data manager instance
multi_source_data_manager = MultiSourceDataManager()
