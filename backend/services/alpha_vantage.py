"""
Alpha Vantage API service for real-time stock data
Documentation: https://www.alphavantage.co/documentation/
"""

import requests
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AlphaVantageAPI:
    """Alpha Vantage API client for stock data"""
    
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = 'https://www.alphavantage.co/query'
        self.rate_limit_delay = 12  # Free tier: 5 calls per minute (12 seconds between calls)
        self.last_call_time = 0
        
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is required")
    
    def _handle_rate_limit(self):
        """Handle rate limiting for free tier (5 calls per minute)"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def _make_request(self, params: Dict) -> Dict:
        """Make API request with error handling and rate limiting"""
        self._handle_rate_limit()
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                # Rate limit exceeded
                raise ValueError(f"Rate limit exceeded: {data['Note']}")
            
            if 'Information' in data:
                # API limit reached
                raise ValueError(f"API limit reached: {data['Information']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")
        except requests.exceptions.Timeout:
            raise ValueError("Request timeout")
    
    def get_global_quote(self, symbol: str) -> Dict:
        """
        Get real-time stock quote using GLOBAL_QUOTE endpoint
        Returns current price and basic info
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol.upper()
        }
        
        data = self._make_request(params)
        
        if 'Global Quote' not in data:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        quote = data['Global Quote']
        
        if not quote.get('05. price'):
            raise ValueError(f"No price data available for symbol: {symbol}")
        
        return {
            'symbol': quote.get('01. symbol', symbol.upper()),
            'price': float(quote.get('05. price', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
            'volume': int(quote.get('06. volume', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'open': float(quote.get('02. open', 0)),
            'previous_close': float(quote.get('08. previous close', 0)),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get company overview information
        Returns detailed company information
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol.upper()
        }
        
        data = self._make_request(params)
        
        if not data or data.get('Symbol') != symbol.upper():
            raise ValueError(f"No company data found for symbol: {symbol}")
        
        return {
            'symbol': data.get('Symbol', symbol.upper()),
            'name': data.get('Name', ''),
            'description': data.get('Description', ''),
            'sector': data.get('Sector', ''),
            'industry': data.get('Industry', ''),
            'market_cap': data.get('MarketCapitalization', '0'),
            'pe_ratio': data.get('PERatio', '0'),
            'dividend_yield': data.get('DividendYield', '0'),
            '52_week_high': data.get('52WeekHigh', '0'),
            '52_week_low': data.get('52WeekLow', '0'),
            'currency': 'USD'
        }
    
    def get_daily_time_series(self, symbol: str, outputsize: str = 'compact') -> Dict:
        """
        Get daily time series data
        outputsize: 'compact' (last 100 days) or 'full' (20+ years)
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol.upper(),
            'outputsize': outputsize
        }
        
        data = self._make_request(params)
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No time series data found for symbol: {symbol}")
        
        time_series = data['Time Series (Daily)']
        
        # Convert to list format for easier processing
        dates = sorted(time_series.keys(), reverse=True)  # Most recent first
        
        historical_data = []
        for date in dates:
            day_data = time_series[date]
            historical_data.append({
                'date': date,
                'open': float(day_data['1. open']),
                'high': float(day_data['2. high']),
                'low': float(day_data['3. low']),
                'close': float(day_data['4. close']),
                'volume': int(day_data['5. volume'])
            })
        
        return {
            'symbol': symbol.upper(),
            'historical_data': historical_data,
            'metadata': data.get('Meta Data', {})
        }

# Global instance
alpha_vantage = AlphaVantageAPI()
