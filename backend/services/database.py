"""
Database service for persistent stock data storage
Uses SQLite for simplicity and reliability
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path

class StockDatabase:
    """SQLite database for storing stock data"""
    
    def __init__(self, db_path: str = "stock_data.db"):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Stock quotes table (real-time data)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_quotes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    change REAL NOT NULL,
                    change_percent REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    open_price REAL NOT NULL,
                    previous_close REAL NOT NULL,
                    market_cap TEXT,
                    sector TEXT,
                    industry TEXT,
                    currency TEXT DEFAULT 'USD',
                    timestamp DATETIME NOT NULL,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Historical data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Company info table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS company_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap TEXT,
                    pe_ratio TEXT,
                    dividend_yield TEXT,
                    week_52_high REAL,
                    week_52_low REAL,
                    currency TEXT DEFAULT 'USD',
                    source TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_quotes_symbol ON stock_quotes(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_quotes_timestamp ON stock_quotes(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_historical_symbol_date ON historical_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_company_symbol ON company_info(symbol)')
            
            conn.commit()
    
    def store_stock_quote(self, symbol: str, quote_data: Dict, source: str = 'yahoo') -> bool:
        """Store real-time stock quote data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_quotes 
                    (symbol, price, change, change_percent, volume, high, low, open_price, 
                     previous_close, market_cap, sector, industry, currency, timestamp, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol.upper(),
                    quote_data.get('price', 0),
                    quote_data.get('change', 0),
                    quote_data.get('change_percent', 0),
                    quote_data.get('volume', 0),
                    quote_data.get('high', 0),
                    quote_data.get('low', 0),
                    quote_data.get('open', 0),
                    quote_data.get('previous_close', 0),
                    quote_data.get('market_cap', 'N/A'),
                    quote_data.get('sector', 'N/A'),
                    quote_data.get('industry', 'N/A'),
                    quote_data.get('currency', 'USD'),
                    quote_data.get('timestamp', datetime.now().isoformat()),
                    source
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error storing stock quote for {symbol}: {e}")
            return False
    
    def store_historical_data(self, symbol: str, historical_data: List[Dict], source: str = 'yahoo') -> bool:
        """Store historical stock data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for day_data in historical_data:
                    cursor.execute('''
                        INSERT OR REPLACE INTO historical_data 
                        (symbol, date, open_price, high, low, close_price, volume, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol.upper(),
                        day_data['date'],
                        day_data['open'],
                        day_data['high'],
                        day_data['low'],
                        day_data['close'],
                        day_data['volume'],
                        source
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error storing historical data for {symbol}: {e}")
            return False
    
    def store_company_info(self, symbol: str, company_data: Dict, source: str = 'yahoo') -> bool:
        """Store company information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO company_info 
                    (symbol, name, description, sector, industry, market_cap, pe_ratio, 
                     dividend_yield, week_52_high, week_52_low, currency, source, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol.upper(),
                    company_data.get('name', ''),
                    company_data.get('description', ''),
                    company_data.get('sector', ''),
                    company_data.get('industry', ''),
                    company_data.get('market_cap', ''),
                    company_data.get('pe_ratio', ''),
                    company_data.get('dividend_yield', ''),
                    company_data.get('52_week_high', 0),
                    company_data.get('52_week_low', 0),
                    company_data.get('currency', 'USD'),
                    source,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error storing company info for {symbol}: {e}")
            return False
    
    def get_latest_quote(self, symbol: str, max_age_minutes: int = 5) -> Optional[Dict]:
        """Get the latest stock quote if it's not too old"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
                
                cursor.execute('''
                    SELECT * FROM stock_quotes 
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (symbol.upper(), cutoff_time.isoformat()))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            print(f"Error getting latest quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 100) -> List[Dict]:
        """Get historical data for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM historical_data 
                    WHERE symbol = ?
                    ORDER BY date DESC LIMIT ?
                ''', (symbol.upper(), days))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Optional[Dict]:
        """Get company information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM company_info WHERE symbol = ?
                ''', (symbol.upper(),))
                
                row = cursor.fetchone()
                return dict(row) if row else None
                
        except Exception as e:
            print(f"Error getting company info for {symbol}: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to keep database size manageable"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Clean old quotes (keep only recent ones)
                cursor.execute('''
                    DELETE FROM stock_quotes 
                    WHERE timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                # Clean old historical data (keep more)
                cutoff_historical = datetime.now() - timedelta(days=365)  # Keep 1 year
                cursor.execute('''
                    DELETE FROM historical_data 
                    WHERE date < ?
                ''', (cutoff_historical.strftime('%Y-%m-%d'),))
                
                conn.commit()
                print(f"Cleaned up data older than {days_to_keep} days")
                
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                cursor.execute('SELECT COUNT(*) FROM stock_quotes')
                stats['quotes_count'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM historical_data')
                stats['historical_count'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM company_info')
                stats['companies_count'] = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor.execute('SELECT COUNT(DISTINCT symbol) FROM stock_quotes')
                stats['unique_symbols'] = cursor.fetchone()[0]
                
                # Database file size
                if os.path.exists(self.db_path):
                    stats['db_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
                
                return stats
                
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}

# Global database instance
stock_db = StockDatabase()
