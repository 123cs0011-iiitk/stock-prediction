"""
Finnhub API service for real-time stock data
Documentation: https://finnhub.io/docs/api
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

class FinnhubAPI:
    """Finnhub API client for stock data"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        self.base_url = 'https://finnhub.io/api/v1'
        self.rate_limit_delay = 1  # 1 second between requests (free tier: 60 calls/minute)
        self.last_call_time = 0
        
        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not found in environment variables")
    
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
            raise ValueError("Finnhub API key is required")
        
        self._handle_rate_limit()
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if isinstance(data, dict) and 'error' in data:
                raise ValueError(f"Finnhub API Error: {data['error']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Finnhub request failed: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError("Finnhub request timeout")
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time stock quote"""
        try:
            data = self._make_request('quote', {'symbol': symbol.upper()})
            
            if not data or 'c' not in data:  # 'c' is current price
                return None
            
            return {
                'symbol': symbol.upper(),
                'price': float(data.get('c', 0)),
                'change': float(data.get('d', 0)),  # Change
                'change_percent': float(data.get('dp', 0)),  # Change percent
                'high': float(data.get('h', 0)),  # High price of the day
                'low': float(data.get('l', 0)),   # Low price of the day
                'open': float(data.get('o', 0)),  # Open price of the day
                'previous_close': float(data.get('pc', 0)),  # Previous close price
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub'
            }
            
        except Exception as e:
            logger.error(f"Error getting Finnhub quote for {symbol}: {e}")
            return None
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """Get company profile information"""
        try:
            data = self._make_request('stock/profile2', {'symbol': symbol.upper()})
            
            if not data or 'ticker' not in data:
                return None
            
            return {
                'symbol': symbol.upper(),
                'name': data.get('name', ''),
                'description': '',  # Finnhub doesn't provide description in profile2
                'sector': data.get('finnhubIndustry', ''),
                'industry': data.get('finnhubIndustry', ''),
                'market_cap': str(data.get('marketCapitalization', '')),
                'country': data.get('country', ''),
                'currency': data.get('currency', 'USD'),
                'website': data.get('weburl', ''),
                'logo': data.get('logo', ''),
                'exchange': data.get('exchange', ''),
                'ipo_date': data.get('ipo', ''),
                'source': 'finnhub'
            }
            
        except Exception as e:
            logger.error(f"Error getting Finnhub company profile for {symbol}: {e}")
            return None
    
    def get_company_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Get company news"""
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            data = self._make_request('company-news', {
                'symbol': symbol.upper(),
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d')
            })
            
            if not isinstance(data, list):
                return []
            
            # Limit to 50 most recent news items
            news_items = []
            for item in data[:50]:
                news_items.append({
                    'headline': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'image': item.get('image', ''),
                    'datetime': item.get('datetime', 0),
                    'source': item.get('source', ''),
                    'category': item.get('category', ''),
                    'id': item.get('id', 0),
                    'related': item.get('related', '')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error getting Finnhub news for {symbol}: {e}")
            return []
    
    def get_candles(self, symbol: str, resolution: str = 'D', count: int = 100) -> Optional[List[Dict]]:
        """Get historical candle data"""
        try:
            # Calculate timestamps
            to_timestamp = int(time.time())
            from_timestamp = to_timestamp - (count * 86400)  # Approximate days
            
            data = self._make_request('stock/candle', {
                'symbol': symbol.upper(),
                'resolution': resolution,
                'from': from_timestamp,
                'to': to_timestamp
            })
            
            if not data or data.get('s') != 'ok':  # 's' is status
                return None
            
            # Extract candle data
            timestamps = data.get('t', [])
            opens = data.get('o', [])
            highs = data.get('h', [])
            lows = data.get('l', [])
            closes = data.get('c', [])
            volumes = data.get('v', [])
            
            candles = []
            for i in range(len(timestamps)):
                candle_date = datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d')
                candles.append({
                    'date': candle_date,
                    'open': float(opens[i]),
                    'high': float(highs[i]),
                    'low': float(lows[i]),
                    'close': float(closes[i]),
                    'volume': int(volumes[i])
                })
            
            # Sort by date (most recent first)
            candles.sort(key=lambda x: x['date'], reverse=True)
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting Finnhub candles for {symbol}: {e}")
            return None
    
    def get_recommendation_trends(self, symbol: str) -> Optional[Dict]:
        """Get analyst recommendation trends"""
        try:
            data = self._make_request('stock/recommendation', {'symbol': symbol.upper()})
            
            if not isinstance(data, list) or len(data) == 0:
                return None
            
            # Get the most recent recommendation
            latest = data[0]
            
            return {
                'symbol': symbol.upper(),
                'period': latest.get('period', ''),
                'strong_buy': latest.get('strongBuy', 0),
                'buy': latest.get('buy', 0),
                'hold': latest.get('hold', 0),
                'sell': latest.get('sell', 0),
                'strong_sell': latest.get('strongSell', 0),
                'source': 'finnhub'
            }
            
        except Exception as e:
            logger.error(f"Error getting Finnhub recommendations for {symbol}: {e}")
            return None
    
    def get_earnings_calendar(self, symbol: str = None) -> List[Dict]:
        """Get earnings calendar (can be filtered by symbol)"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol.upper()
            
            # Get earnings for next 30 days
            to_date = datetime.now() + timedelta(days=30)
            params['from'] = datetime.now().strftime('%Y-%m-%d')
            params['to'] = to_date.strftime('%Y-%m-%d')
            
            data = self._make_request('calendar/earnings', params)
            
            if not isinstance(data, list):
                return []
            
            earnings = []
            for item in data:
                earnings.append({
                    'symbol': item.get('symbol', ''),
                    'date': item.get('date', ''),
                    'eps_actual': item.get('epsActual'),
                    'eps_estimate': item.get('epsEstimate'),
                    'hour': item.get('hour', ''),
                    'quarter': item.get('quarter'),
                    'revenue_actual': item.get('revenueActual'),
                    'revenue_estimate': item.get('revenueEstimate'),
                    'year': item.get('year')
                })
            
            return earnings
            
        except Exception as e:
            logger.error(f"Error getting Finnhub earnings calendar: {e}")
            return []
    
    def search_stocks(self, query: str) -> List[Dict]:
        """Search for stocks"""
        try:
            data = self._make_request('search', {'q': query})
            
            if not isinstance(data, list):
                return []
            
            results = []
            for item in data[:10]:  # Limit to 10 results
                results.append({
                    'symbol': item.get('symbol', ''),
                    'description': item.get('description', ''),
                    'display_symbol': item.get('displaySymbol', ''),
                    'type': item.get('type', ''),
                    'source': 'finnhub'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Finnhub stocks: {e}")
            return []

# Global Finnhub instance
finnhub_api = FinnhubAPI()
