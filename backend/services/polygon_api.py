"""
Polygon.io API service for real-time and historical stock data
Documentation: https://polygon.io/docs
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class PolygonAPI:
    """Polygon.io API client for stock data"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.base_url = 'https://api.polygon.io'
        self.rate_limit_delay = 1  # 1 second between requests (free tier: 5 calls/minute)
        self.last_call_time = 0
        
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not found in environment variables")
    
    def _handle_rate_limit(self):
        """Handle rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling and rate limiting"""
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self._handle_rate_limit()
        
        if params is None:
            params = {}
        
        params['apikey'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if data.get('status') != 'OK':
                error_msg = data.get('message', 'Unknown error')
                raise ValueError(f"Polygon API Error: {error_msg}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Polygon request failed: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError("Polygon request timeout")
    
    def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed ticker information"""
        try:
            data = self._make_request(f'v3/reference/tickers/{symbol.upper()}')
            
            if not data or 'results' not in data:
                return None
            
            ticker_data = data['results']
            
            return {
                'symbol': symbol.upper(),
                'name': ticker_data.get('name', ''),
                'description': ticker_data.get('description', ''),
                'market': ticker_data.get('market', ''),
                'locale': ticker_data.get('locale', ''),
                'primary_exchange': ticker_data.get('primary_exchange', ''),
                'type': ticker_data.get('type', ''),
                'currency': ticker_data.get('currency_name', 'USD'),
                'active': ticker_data.get('active', True),
                'cik': ticker_data.get('cik', ''),
                'composite_figi': ticker_data.get('composite_figi', ''),
                'share_class_figi': ticker_data.get('share_class_figi', ''),
                'market_cap': ticker_data.get('market_cap'),
                'phone_number': ticker_data.get('phone_number', ''),
                'address': ticker_data.get('address', {}),
                'description': ticker_data.get('description', ''),
                'sic_code': ticker_data.get('sic_code', ''),
                'sic_description': ticker_data.get('sic_description', ''),
                'ticker_root': ticker_data.get('ticker_root', ''),
                'homepage_url': ticker_data.get('homepage_url', ''),
                'total_employees': ticker_data.get('total_employees'),
                'list_date': ticker_data.get('list_date', ''),
                'branding': ticker_data.get('branding', {}),
                'share_class_shares_outstanding': ticker_data.get('share_class_shares_outstanding'),
                'weighted_shares_outstanding': ticker_data.get('weighted_shares_outstanding'),
                'round_lot': ticker_data.get('round_lot'),
                'source': 'polygon'
            }
            
        except Exception as e:
            logger.error(f"Error getting Polygon ticker details for {symbol}: {e}")
            return None
    
    def get_previous_close(self, symbol: str) -> Optional[Dict]:
        """Get previous day's close data"""
        try:
            data = self._make_request(f'v2/aggs/ticker/{symbol.upper()}/prev')
            
            if not data or 'results' not in data or len(data['results']) == 0:
                return None
            
            result = data['results'][0]
            
            return {
                'symbol': symbol.upper(),
                'price': float(result.get('c', 0)),  # Close price
                'change': float(result.get('c', 0)) - float(result.get('o', 0)),  # Close - Open
                'change_percent': ((float(result.get('c', 0)) - float(result.get('o', 0))) / float(result.get('o', 1))) * 100,
                'high': float(result.get('h', 0)),
                'low': float(result.get('l', 0)),
                'open': float(result.get('o', 0)),
                'previous_close': float(result.get('c', 0)),
                'volume': int(result.get('v', 0)),
                'timestamp': datetime.fromtimestamp(result.get('t', 0) / 1000).isoformat(),
                'source': 'polygon'
            }
            
        except Exception as e:
            logger.error(f"Error getting Polygon previous close for {symbol}: {e}")
            return None
    
    def get_aggregates(self, symbol: str, multiplier: int = 1, timespan: str = 'day', 
                      from_date: str = None, to_date: str = None, limit: int = 100) -> Optional[List[Dict]]:
        """Get aggregate bars for a ticker"""
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=limit)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            
            data = self._make_request(f'v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{from_date}/{to_date}')
            
            if not data or 'results' not in data:
                return None
            
            aggregates = []
            for result in data['results']:
                aggregates.append({
                    'date': datetime.fromtimestamp(result.get('t', 0) / 1000).strftime('%Y-%m-%d'),
                    'open': float(result.get('o', 0)),
                    'high': float(result.get('h', 0)),
                    'low': float(result.get('l', 0)),
                    'close': float(result.get('c', 0)),
                    'volume': int(result.get('v', 0)),
                    'volume_weighted_avg_price': float(result.get('vw', 0)),
                    'transactions': int(result.get('n', 0))
                })
            
            # Sort by date (most recent first)
            aggregates.sort(key=lambda x: x['date'], reverse=True)
            
            return aggregates
            
        except Exception as e:
            logger.error(f"Error getting Polygon aggregates for {symbol}: {e}")
            return None
    
    def get_grouped_daily(self, date: str = None) -> List[Dict]:
        """Get grouped daily bars for all tickers on a specific date"""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            data = self._make_request(f'v2/aggs/grouped/locale/us/market/stocks/{date}')
            
            if not data or 'results' not in data:
                return []
            
            grouped_data = []
            for result in data['results'][:100]:  # Limit to 100 results
                grouped_data.append({
                    'symbol': result.get('T', ''),
                    'price': float(result.get('c', 0)),
                    'change': float(result.get('c', 0)) - float(result.get('o', 0)),
                    'change_percent': ((float(result.get('c', 0)) - float(result.get('o', 0))) / float(result.get('o', 1))) * 100,
                    'high': float(result.get('h', 0)),
                    'low': float(result.get('l', 0)),
                    'open': float(result.get('o', 0)),
                    'volume': int(result.get('v', 0)),
                    'volume_weighted_avg_price': float(result.get('vw', 0)),
                    'transactions': int(result.get('n', 0)),
                    'source': 'polygon'
                })
            
            return grouped_data
            
        except Exception as e:
            logger.error(f"Error getting Polygon grouped daily: {e}")
            return []
    
    def get_ticker_news(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get news for a specific ticker"""
        try:
            data = self._make_request(f'v2/reference/news', {
                'ticker': symbol.upper(),
                'limit': limit
            })
            
            if not data or 'results' not in data:
                return []
            
            news_items = []
            for item in data['results']:
                news_items.append({
                    'id': item.get('id', ''),
                    'publisher': item.get('publisher', {}).get('name', ''),
                    'title': item.get('title', ''),
                    'author': item.get('author', ''),
                    'published_utc': item.get('published_utc', ''),
                    'article_url': item.get('article_url', ''),
                    'tickers': item.get('tickers', []),
                    'image_url': item.get('image_url', ''),
                    'description': item.get('description', ''),
                    'keywords': item.get('keywords', []),
                    'source': 'polygon'
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error getting Polygon news for {symbol}: {e}")
            return []
    
    def search_tickers(self, search: str, limit: int = 10) -> List[Dict]:
        """Search for tickers"""
        try:
            data = self._make_request('v3/reference/tickers', {
                'search': search,
                'active': True,
                'limit': limit
            })
            
            if not data or 'results' not in data:
                return []
            
            results = []
            for ticker in data['results']:
                results.append({
                    'symbol': ticker.get('ticker', ''),
                    'name': ticker.get('name', ''),
                    'market': ticker.get('market', ''),
                    'locale': ticker.get('locale', ''),
                    'primary_exchange': ticker.get('primary_exchange', ''),
                    'type': ticker.get('type', ''),
                    'currency': ticker.get('currency_name', 'USD'),
                    'active': ticker.get('active', True),
                    'description': ticker.get('description', ''),
                    'source': 'polygon'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Polygon tickers: {e}")
            return []
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            data = self._make_request('v1/marketstatus/now')
            
            if not data:
                return {'status': 'unknown'}
            
            return {
                'market': data.get('market', 'unknown'),
                'serverTime': data.get('serverTime', ''),
                'exchanges': data.get('exchanges', {}),
                'currencies': data.get('currencies', {}),
                'source': 'polygon'
            }
            
        except Exception as e:
            logger.error(f"Error getting Polygon market status: {e}")
            return {'status': 'error', 'error': str(e)}

# Global Polygon instance
polygon_api = PolygonAPI()
