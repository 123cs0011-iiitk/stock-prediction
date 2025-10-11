"""
Data Manager - Intelligent data fetching and caching system
Combines Yahoo Finance API with database storage and Alpha Vantage fallback
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from services.yahoo_finance import yahoo_finance
from services.alpha_vantage import alpha_vantage
from services.postgresql_database import postgres_db
from services.csv_backup_service import csv_backup
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Intelligent data manager with caching and fallback strategies"""
    
    def __init__(self):
        self.cache_duration_minutes = 5  # Cache live quotes for 5 minutes
        self.historical_cache_days = 1   # Cache historical data for 1 day
        self.company_info_cache_days = 7 # Cache company info for 7 days
    
    def get_live_quote(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get live stock quote with intelligent caching and fallback
        Priority: PostgreSQL cache -> CSV backup cache -> Yahoo Finance -> Alpha Vantage
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
            
            # Try Yahoo Finance first
            try:
                logger.info(f"Fetching live quote for {symbol} from Yahoo Finance")
                yahoo_quote = yahoo_finance.get_live_quote(symbol)
                
                if yahoo_quote:
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_stock_quote(symbol, yahoo_quote, 'yahoo')
                        logger.info(f"Stored quote for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store quote for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_stock_quote(symbol, yahoo_quote, 'yahoo')
                        logger.info(f"Stored quote for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store quote for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored quote for {symbol} from Yahoo Finance")
                    return self._format_quote_response(yahoo_quote)
                    
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for {symbol}: {e}")
            
            # Fallback to Alpha Vantage
            try:
                logger.info(f"Falling back to Alpha Vantage for {symbol}")
                alpha_quote = alpha_vantage.get_global_quote(symbol)
                
                if alpha_quote:
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_stock_quote(symbol, alpha_quote, 'alpha_vantage')
                        logger.info(f"Stored quote for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store quote for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_stock_quote(symbol, alpha_quote, 'alpha_vantage')
                        logger.info(f"Stored quote for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store quote for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored quote for {symbol} from Alpha Vantage")
                    return self._format_quote_response(alpha_quote)
                    
            except Exception as e:
                logger.warning(f"Alpha Vantage also failed for {symbol}: {e}")
            
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
        Get company information with caching
        """
        try:
            symbol = symbol.upper()
            
            # Check PostgreSQL cache first
            if not force_refresh:
                try:
                    cached_info = postgres_db.get_company_info(symbol)
                    if cached_info:
                        # Check if data is fresh enough
                        last_updated = datetime.fromisoformat(cached_info['last_updated'])
                        if datetime.now() - last_updated < timedelta(days=self.company_info_cache_days):
                            logger.info(f"Using PostgreSQL cached company info for {symbol}")
                            return cached_info
                except Exception as e:
                    logger.warning(f"PostgreSQL company info cache check failed for {symbol}: {e}")
                
                # Fallback to CSV backup cache
                try:
                    cached_info = csv_backup.get_company_info(symbol)
                    if cached_info:
                        logger.info(f"Using CSV backup cached company info for {symbol}")
                        return cached_info
                except Exception as e:
                    logger.warning(f"CSV backup company info cache check failed for {symbol}: {e}")
            
            # Try Yahoo Finance first
            try:
                logger.info(f"Fetching company info for {symbol} from Yahoo Finance")
                yahoo_info = yahoo_finance.get_company_info(symbol)
                
                if yahoo_info:
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_company_info(symbol, yahoo_info, 'yahoo')
                        logger.info(f"Stored company info for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store company info for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_company_info(symbol, yahoo_info, 'yahoo')
                        logger.info(f"Stored company info for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store company info for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored company info for {symbol}")
                    return yahoo_info
                    
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for company info {symbol}: {e}")
            
            # Fallback to Alpha Vantage
            try:
                logger.info(f"Falling back to Alpha Vantage for company info {symbol}")
                alpha_info = alpha_vantage.get_company_overview(symbol)
                
                if alpha_info:
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_company_info(symbol, alpha_info, 'alpha_vantage')
                        logger.info(f"Stored company info for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store company info for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_company_info(symbol, alpha_info, 'alpha_vantage')
                        logger.info(f"Stored company info for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store company info for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored company info for {symbol}")
                    return alpha_info
                    
            except Exception as e:
                logger.warning(f"Alpha Vantage also failed for company info {symbol}: {e}")
            
            # Return cached data if available
            if not force_refresh:
                # Try PostgreSQL first
                try:
                    cached_info = postgres_db.get_company_info(symbol)
                    if cached_info:
                        logger.info(f"Using cached PostgreSQL company info for {symbol}")
                        return cached_info
                except Exception as e:
                    logger.warning(f"PostgreSQL cached company info check failed for {symbol}: {e}")
                
                # Fallback to CSV backup
                try:
                    cached_info = csv_backup.get_company_info(symbol)
                    if cached_info:
                        logger.info(f"Using cached CSV backup company info for {symbol}")
                        return cached_info
                except Exception as e:
                    logger.warning(f"CSV backup cached company info check failed for {symbol}: {e}")
            
            logger.error(f"Failed to fetch company info for {symbol} from all sources")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_company_info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y", force_refresh: bool = False) -> Optional[List[Dict]]:
        """
        Get historical data with caching
        """
        try:
            symbol = symbol.upper()
            
            # Check PostgreSQL cache first
            if not force_refresh:
                try:
                    cached_historical = postgres_db.get_historical_data(symbol, 365)  # Get up to 1 year
                    if cached_historical:
                        # Check if we have recent data
                        latest_date = datetime.strptime(cached_historical[0]['date'], '%Y-%m-%d')
                        if datetime.now() - latest_date < timedelta(days=self.historical_cache_days):
                            logger.info(f"Using PostgreSQL cached historical data for {symbol}")
                            return self._format_historical_response(cached_historical)
                except Exception as e:
                    logger.warning(f"PostgreSQL historical data cache check failed for {symbol}: {e}")
                
                # Fallback to CSV backup cache
                try:
                    cached_historical = csv_backup.get_historical_data(symbol, 365)  # Get up to 1 year
                    if cached_historical:
                        # Check if we have recent data
                        latest_date = datetime.strptime(cached_historical[0]['date'], '%Y-%m-%d')
                        if datetime.now() - latest_date < timedelta(days=self.historical_cache_days):
                            logger.info(f"Using CSV backup cached historical data for {symbol}")
                            return self._format_historical_response(cached_historical)
                except Exception as e:
                    logger.warning(f"CSV backup historical data cache check failed for {symbol}: {e}")
            
            # Try Yahoo Finance first
            try:
                logger.info(f"Fetching historical data for {symbol} from Yahoo Finance")
                yahoo_historical = yahoo_finance.get_historical_data(symbol, period)
                
                if yahoo_historical:
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_historical_data(symbol, yahoo_historical, 'yahoo')
                        logger.info(f"Stored historical data for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store historical data for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_historical_data(symbol, yahoo_historical, 'yahoo')
                        logger.info(f"Stored historical data for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store historical data for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored historical data for {symbol}")
                    return self._format_historical_response(yahoo_historical)
                    
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for historical data {symbol}: {e}")
            
            # Fallback to Alpha Vantage
            try:
                logger.info(f"Falling back to Alpha Vantage for historical data {symbol}")
                alpha_historical = alpha_vantage.get_daily_time_series(symbol, 'compact')
                
                if alpha_historical and 'historical_data' in alpha_historical:
                    historical_list = alpha_historical['historical_data']
                    
                    # Store in both PostgreSQL and CSV backup
                    try:
                        postgres_db.store_historical_data(symbol, historical_list, 'alpha_vantage')
                        logger.info(f"Stored historical data for {symbol} in PostgreSQL")
                    except Exception as e:
                        logger.warning(f"Failed to store historical data for {symbol} in PostgreSQL: {e}")
                    
                    try:
                        csv_backup.store_historical_data(symbol, historical_list, 'alpha_vantage')
                        logger.info(f"Stored historical data for {symbol} in CSV backup")
                    except Exception as e:
                        logger.warning(f"Failed to store historical data for {symbol} in CSV backup: {e}")
                    
                    logger.info(f"Successfully fetched and stored historical data for {symbol}")
                    return self._format_historical_response(historical_list)
                    
            except Exception as e:
                logger.warning(f"Alpha Vantage also failed for historical data {symbol}: {e}")
            
            # Return cached data if available
            if not force_refresh:
                # Try PostgreSQL first
                try:
                    cached_historical = postgres_db.get_historical_data(symbol, 365)
                    if cached_historical:
                        logger.info(f"Using cached PostgreSQL historical data for {symbol}")
                        return self._format_historical_response(cached_historical)
                except Exception as e:
                    logger.warning(f"PostgreSQL cached historical data check failed for {symbol}: {e}")
                
                # Fallback to CSV backup
                try:
                    cached_historical = csv_backup.get_historical_data(symbol, 365)
                    if cached_historical:
                        logger.info(f"Using cached CSV backup historical data for {symbol}")
                        return self._format_historical_response(cached_historical)
                except Exception as e:
                    logger.warning(f"CSV backup cached historical data check failed for {symbol}: {e}")
            
            logger.error(f"Failed to fetch historical data for {symbol} from all sources")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_historical_data for {symbol}: {e}")
            return None
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks
        """
        try:
            # Use Yahoo Finance search
            results = yahoo_finance.search_stocks(query)
            
            # Enhance with cached data if available
            enhanced_results = []
            for stock in results:
                symbol = stock['symbol']
                
                # Try to get additional info from cache
                cached_quote = postgres_db.get_latest_quote(symbol, 60)  # 1 hour cache
                if cached_quote:
                    stock.update({
                        'price': cached_quote['price'],
                        'change': cached_quote['change'],
                        'change_percent': cached_quote['change_percent'],
                        'volume': cached_quote['volume'],
                        'source': 'cached'
                    })
                
                enhanced_results.append(stock)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in search_stocks: {e}")
            return []
    
    def get_comprehensive_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get comprehensive stock data (quote + company info + historical)
        """
        try:
            symbol = symbol.upper()
            
            # Get all data in parallel (simplified - in real app you might use threading)
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
                'timestamp': datetime.now().isoformat()
            }
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error in get_comprehensive_stock_data for {symbol}: {e}")
            return None
    
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
        """Get statistics about data usage and storage"""
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
                    'primary': 'Yahoo Finance (yfinance)',
                    'fallback': 'Alpha Vantage',
                    'storage': 'SQLite Database'
                },
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

# Global data manager instance
data_manager = DataManager()
