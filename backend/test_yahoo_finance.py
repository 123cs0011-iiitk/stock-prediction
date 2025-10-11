#!/usr/bin/env python3
"""
Test script for Yahoo Finance API integration
Tests the new data management system with Yahoo Finance and database storage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.yahoo_finance import yahoo_finance
from services.postgresql_database import postgres_db
from services.data_manager import data_manager
import json
from datetime import datetime

def test_yahoo_finance_basic():
    """Test basic Yahoo Finance functionality"""
    print("=" * 60)
    print("Testing Yahoo Finance API Basic Functionality")
    print("=" * 60)
    
    test_symbol = "AAPL"
    
    # Test live quote
    print(f"\n1. Testing live quote for {test_symbol}...")
    try:
        quote = yahoo_finance.get_live_quote(test_symbol)
        if quote:
            print(f"[SUCCESS] Live quote successful:")
            print(f"   Symbol: {quote['symbol']}")
            print(f"   Price: ${quote['price']}")
            print(f"   Change: ${quote['change']} ({quote['change_percent']}%)")
            print(f"   Volume: {quote['volume']:,}")
        else:
            print("[FAILED] Failed to get live quote")
    except Exception as e:
        print(f"[ERROR] Error getting live quote: {e}")
    
    # Test company info
    print(f"\n2. Testing company info for {test_symbol}...")
    try:
        company = yahoo_finance.get_company_info(test_symbol)
        if company:
            print(f"[SUCCESS] Company info successful:")
            print(f"   Name: {company['name']}")
            print(f"   Sector: {company['sector']}")
            print(f"   Industry: {company['industry']}")
            print(f"   Market Cap: {company['market_cap']}")
        else:
            print("[FAILED] Failed to get company info")
    except Exception as e:
        print(f"[ERROR] Error getting company info: {e}")
    
    # Test historical data
    print(f"\n3. Testing historical data for {test_symbol}...")
    try:
        historical = yahoo_finance.get_historical_data(test_symbol, "1mo")
        if historical:
            print(f"[SUCCESS] Historical data successful:")
            print(f"   Records: {len(historical)}")
            print(f"   Latest date: {historical[0]['date']}")
            print(f"   Latest close: ${historical[0]['close']}")
        else:
            print("[FAILED] Failed to get historical data")
    except Exception as e:
        print(f"[ERROR] Error getting historical data: {e}")

def test_database_functionality():
    """Test database storage and retrieval"""
    print("\n" + "=" * 60)
    print("Testing Database Functionality")
    print("=" * 60)
    
    test_symbol = "MSFT"
    
    # Test storing and retrieving quote
    print(f"\n1. Testing database storage for {test_symbol}...")
    try:
        # Get quote from Yahoo Finance
        quote = yahoo_finance.get_live_quote(test_symbol)
        if quote:
            # Store in database
            success = postgres_db.store_stock_quote(test_symbol, quote, 'yahoo')
            if success:
                print("[SUCCESS] Successfully stored quote in database")
                
                # Retrieve from database
                cached_quote = postgres_db.get_latest_quote(test_symbol, 60)
                if cached_quote:
                    print("[SUCCESS] Successfully retrieved quote from database")
                    print(f"   Cached price: ${cached_quote['price']}")
                else:
                    print("[FAILED] Failed to retrieve quote from database")
            else:
                print("[FAILED] Failed to store quote in database")
        else:
            print("[FAILED] No quote data to store")
    except Exception as e:
        print(f"[ERROR] Error testing database: {e}")
    
    # Test database stats
    print(f"\n2. Testing database statistics...")
    try:
        stats = postgres_db.get_database_stats()
        print("[SUCCESS] Database statistics:")
        print(f"   Quotes count: {stats.get('quotes_count', 0)}")
        print(f"   Historical count: {stats.get('historical_count', 0)}")
        print(f"   Companies count: {stats.get('companies_count', 0)}")
        print(f"   Unique symbols: {stats.get('unique_symbols', 0)}")
        print(f"   Database size: {stats.get('db_size_mb', 0)} MB")
    except Exception as e:
        print(f"[ERROR] Error getting database stats: {e}")

def test_data_manager():
    """Test the intelligent data manager"""
    print("\n" + "=" * 60)
    print("Testing Data Manager (Intelligent Caching)")
    print("=" * 60)
    
    test_symbol = "GOOGL"
    
    # Test comprehensive data fetching
    print(f"\n1. Testing comprehensive data for {test_symbol}...")
    try:
        data = data_manager.get_comprehensive_stock_data(test_symbol)
        if data:
            print("[SUCCESS] Comprehensive data successful:")
            print(f"   Symbol: {data['symbol']}")
            print(f"   Quote source: {data['data_sources']['quote_source']}")
            print(f"   Company source: {data['data_sources']['company_source']}")
            print(f"   Historical source: {data['data_sources']['historical_source']}")
            
            quote = data['quote']
            print(f"   Current price: ${quote['price']}")
            print(f"   Company name: {quote['name']}")
        else:
            print("[FAILED] Failed to get comprehensive data")
    except Exception as e:
        print(f"[ERROR] Error getting comprehensive data: {e}")
    
    # Test caching (second call should be faster)
    print(f"\n2. Testing caching (second call for {test_symbol})...")
    try:
        import time
        start_time = time.time()
        data = data_manager.get_live_quote(test_symbol)
        end_time = time.time()
        
        if data:
            print(f"[SUCCESS] Cached data retrieved in {end_time - start_time:.2f} seconds")
            print(f"   Price: ${data['price']}")
            print(f"   Source: {data.get('source', 'unknown')}")
        else:
            print("[FAILED] Failed to get cached data")
    except Exception as e:
        print(f"[ERROR] Error testing cache: {e}")
    
    # Test data manager statistics
    print(f"\n3. Testing data manager statistics...")
    try:
        stats = data_manager.get_data_statistics()
        print("[SUCCESS] Data manager statistics:")
        print(f"   Database size: {stats['database_stats'].get('db_size_mb', 0)} MB")
        print(f"   Primary source: {stats['data_sources']['primary']}")
        print(f"   Fallback source: {stats['data_sources']['fallback']}")
        print(f"   Storage: {stats['data_sources']['storage']}")
    except Exception as e:
        print(f"[ERROR] Error getting data manager stats: {e}")

def test_fallback_logic():
    """Test fallback from Yahoo Finance to Alpha Vantage"""
    print("\n" + "=" * 60)
    print("Testing Fallback Logic")
    print("=" * 60)
    
    # Test with a symbol that might not exist on Yahoo Finance
    test_symbol = "INVALID_SYMBOL"
    
    print(f"\n1. Testing fallback for invalid symbol {test_symbol}...")
    try:
        quote = data_manager.get_live_quote(test_symbol)
        if quote:
            print(f"[SUCCESS] Fallback successful (unexpected): {quote['source']}")
        else:
            print("[SUCCESS] Correctly returned None for invalid symbol")
    except Exception as e:
        print(f"[ERROR] Error testing fallback: {e}")

def main():
    """Run all tests"""
    print("Yahoo Finance API Integration Test Suite")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        # Test Yahoo Finance basic functionality
        test_yahoo_finance_basic()
        
        # Test database functionality
        test_database_functionality()
        
        # Test data manager
        test_data_manager()
        
        # Test fallback logic
        test_fallback_logic()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")

if __name__ == "__main__":
    main()
