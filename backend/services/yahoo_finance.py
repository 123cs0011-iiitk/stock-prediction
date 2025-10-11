"""
Yahoo Finance API service using yfinance library
Provides free access to real-time and historical stock data
"""

import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

class YahooFinanceAPI:
    """Yahoo Finance API client using yfinance library"""
    
    def __init__(self):
        self.rate_limit_delay = 1  # 1 second between requests to be respectful
        self.last_call_time = 0
        self.session_timeout = 300  # 5 minutes
    
    def _handle_rate_limit(self):
        """Handle rate limiting to be respectful to Yahoo Finance"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def get_stock_info(self, symbol: str) -> Optional[yf.Ticker]:
        """Get stock ticker object"""
        try:
            self._handle_rate_limit()
            ticker = yf.Ticker(symbol.upper())
            
            # Test if ticker is valid by getting basic info
            info = ticker.info
            if not info or 'symbol' not in info:
                return None
            
            return ticker
            
        except Exception as e:
            print(f"Error getting ticker for {symbol}: {e}")
            return None
    
    def get_live_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time stock quote"""
        try:
            ticker = self.get_stock_info(symbol)
            if not ticker:
                return None
            
            # Get current price and basic info
            info = ticker.info
            
            # Get recent history for current price (more reliable than info['currentPrice'])
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close else 0
            
            return {
                'symbol': symbol.upper(),
                'price': round(float(current_price), 2),
                'change': round(float(change), 2),
                'change_percent': round(float(change_percent), 2),
                'volume': int(info.get('volume', hist['Volume'].iloc[-1] if not hist.empty else 0)),
                'high': round(float(info.get('dayHigh', hist['High'].iloc[-1] if not hist.empty else current_price)), 2),
                'low': round(float(info.get('dayLow', hist['Low'].iloc[-1] if not hist.empty else current_price)), 2),
                'open': round(float(info.get('open', hist['Open'].iloc[-1] if not hist.empty else current_price)), 2),
                'previous_close': round(float(previous_close), 2),
                'market_cap': str(info.get('marketCap', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error getting live quote for {symbol}: {e}")
            return None
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed company information"""
        try:
            ticker = self.get_stock_info(symbol)
            if not ticker:
                return None
            
            info = ticker.info
            
            return {
                'symbol': symbol.upper(),
                'name': info.get('longName', info.get('shortName', f'{symbol} Corporation')),
                'description': info.get('longBusinessSummary', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': str(info.get('marketCap', '')),
                'pe_ratio': str(info.get('trailingPE', '')),
                'dividend_yield': str(info.get('dividendYield', '')),
                '52_week_high': float(info.get('fiftyTwoWeekHigh', 0)),
                '52_week_low': float(info.get('fiftyTwoWeekLow', 0)),
                'currency': info.get('currency', 'USD'),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', ''),
                'city': info.get('city', ''),
                'state': info.get('state', ''),
                'country': info.get('country', '')
            }
            
        except Exception as e:
            print(f"Error getting company info for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[List[Dict]]:
        """Get historical stock data"""
        try:
            ticker = self.get_stock_info(symbol)
            if not ticker:
                return None
            
            # Get historical data
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return None
            
            # Convert to list format
            historical_data = []
            for date, row in hist.iterrows():
                historical_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': round(float(row['Open']), 2),
                    'high': round(float(row['High']), 2),
                    'low': round(float(row['Low']), 2),
                    'close': round(float(row['Close']), 2),
                    'volume': int(row['Volume'])
                })
            
            # Sort by date (most recent first)
            historical_data.sort(key=lambda x: x['date'], reverse=True)
            
            return historical_data
            
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_dividend_history(self, symbol: str) -> Optional[List[Dict]]:
        """Get dividend history"""
        try:
            ticker = self.get_stock_info(symbol)
            if not ticker:
                return None
            
            dividends = ticker.dividends
            
            if dividends.empty:
                return []
            
            dividend_data = []
            for date, dividend in dividends.items():
                dividend_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'dividend': float(dividend)
                })
            
            # Sort by date (most recent first)
            dividend_data.sort(key=lambda x: x['date'], reverse=True)
            
            return dividend_data
            
        except Exception as e:
            print(f"Error getting dividend history for {symbol}: {e}")
            return None
    
    def get_stock_splits(self, symbol: str) -> Optional[List[Dict]]:
        """Get stock split history"""
        try:
            ticker = self.get_stock_info(symbol)
            if not ticker:
                return None
            
            splits = ticker.splits
            
            if splits.empty:
                return []
            
            split_data = []
            for date, split in splits.items():
                split_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'split_ratio': str(split)
                })
            
            # Sort by date (most recent first)
            split_data.sort(key=lambda x: x['date'], reverse=True)
            
            return split_data
            
        except Exception as e:
            print(f"Error getting stock splits for {symbol}: {e}")
            return None
    
    def search_stocks(self, query: str) -> List[Dict]:
        """Search for stocks by symbol or name"""
        try:
            # For now, we'll implement a simple search
            # In a real implementation, you might want to use a stock symbol API
            # or maintain a local database of stock symbols
            
            # Common stock symbols for testing
            common_stocks = [
                {'symbol': 'AAPL', 'name': 'Apple Inc.'},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
                {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
                {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
                {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
                {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
                {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
                {'symbol': 'AMD', 'name': 'Advanced Micro Devices Inc.'},
                {'symbol': 'INTC', 'name': 'Intel Corporation'}
            ]
            
            query_upper = query.upper()
            results = []
            
            for stock in common_stocks:
                if (query_upper in stock['symbol'] or 
                    query_upper in stock['name'].upper()):
                    results.append(stock)
            
            return results[:10]  # Limit to 10 results
            
        except Exception as e:
            print(f"Error searching stocks: {e}")
            return []
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        try:
            # Check if US market is open (simplified)
            now = datetime.now()
            weekday = now.weekday()
            hour = now.hour
            
            # US market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            is_market_open = (
                weekday < 5 and  # Monday-Friday
                9 <= hour <= 16  # Rough approximation of market hours
            )
            
            return {
                'is_open': is_market_open,
                'timestamp': now.isoformat(),
                'timezone': 'ET'
            }
            
        except Exception as e:
            print(f"Error getting market status: {e}")
            return {'is_open': False, 'timestamp': datetime.now().isoformat()}

# Global Yahoo Finance instance
yahoo_finance = YahooFinanceAPI()
