#!/usr/bin/env python3
"""
Test script for Multi-Source Data Manager
Tests all 4 data sources: Yahoo Finance, Finnhub, Polygon.io, and Alpha Vantage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.multi_source_data_manager import multi_source_data_manager
from services.finnhub_api import finnhub_api
from services.polygon_api import polygon_api
from services.yahoo_finance import yahoo_finance
from services.alpha_vantage import alpha_vantage
import json
from datetime import datetime

def test_individual_apis():
    """Test individual API services"""
    print("=" * 80)
    print("Testing Individual API Services")
    print("=" * 80)
    
    test_symbol = "AAPL"
    
    # Test Yahoo Finance
    print(f"\n1. Testing Yahoo Finance API...")
    try:
        quote = yahoo_finance.get_live_quote(test_symbol)
        if quote:
            print(f"[SUCCESS] Yahoo Finance - Price: ${quote['price']}")
        else:
            print("[FAILED] Yahoo Finance - No data returned")
    except Exception as e:
        print(f"[ERROR] Yahoo Finance: {e}")
    
    # Test Finnhub
    print(f"\n2. Testing Finnhub API...")
    try:
        quote = finnhub_api.get_quote(test_symbol)
        if quote:
            print(f"[SUCCESS] Finnhub - Price: ${quote['price']}")
        else:
            print("[FAILED] Finnhub - No data returned")
    except Exception as e:
        print(f"[ERROR] Finnhub: {e}")
    
    # Test Polygon.io
    print(f"\n3. Testing Polygon.io API...")
    try:
        quote = polygon_api.get_previous_close(test_symbol)
        if quote:
            print(f"[SUCCESS] Polygon.io - Price: ${quote['price']}")
        else:
            print("[FAILED] Polygon.io - No data returned")
    except Exception as e:
        print(f"[ERROR] Polygon.io: {e}")
    
    # Test Alpha Vantage
    print(f"\n4. Testing Alpha Vantage API...")
    try:
        quote = alpha_vantage.get_global_quote(test_symbol)
        if quote:
            print(f"[SUCCESS] Alpha Vantage - Price: ${quote['price']}")
        else:
            print("[FAILED] Alpha Vantage - No data returned")
    except Exception as e:
        print(f"[ERROR] Alpha Vantage: {e}")

def test_multi_source_manager():
    """Test the multi-source data manager"""
    print("\n" + "=" * 80)
    print("Testing Multi-Source Data Manager")
    print("=" * 80)
    
    test_symbol = "MSFT"
    
    # Test comprehensive data fetching
    print(f"\n1. Testing comprehensive data for {test_symbol}...")
    try:
        data = multi_source_data_manager.get_comprehensive_stock_data(test_symbol)
        if data:
            print("[SUCCESS] Comprehensive data retrieved:")
            print(f"   Symbol: {data['symbol']}")
            print(f"   Quote source: {data['data_sources']['quote_source']}")
            print(f"   Company source: {data['data_sources']['company_source']}")
            print(f"   Historical source: {data['data_sources']['historical_source']}")
            print(f"   API Status: {data['api_status']}")
            
            quote = data['quote']
            print(f"   Current price: ${quote['price']}")
            print(f"   Company name: {quote['name']}")
        else:
            print("[FAILED] No comprehensive data retrieved")
    except Exception as e:
        print(f"[ERROR] Comprehensive data test: {e}")
    
    # Test search functionality
    print(f"\n2. Testing stock search...")
    try:
        results = multi_source_data_manager.search_stocks("Apple")
        if results:
            print(f"[SUCCESS] Search returned {len(results)} results:")
            for i, result in enumerate(results[:3]):
                print(f"   {i+1}. {result.get('symbol', 'N/A')} - {result.get('name', 'N/A')}")
        else:
            print("[FAILED] No search results")
    except Exception as e:
        print(f"[ERROR] Search test: {e}")
    
    # Test news functionality
    print(f"\n3. Testing news data for {test_symbol}...")
    try:
        news = multi_source_data_manager.get_news_data(test_symbol, limit=5)
        if news:
            print(f"[SUCCESS] Retrieved {len(news)} news items:")
            for i, item in enumerate(news[:3]):
                print(f"   {i+1}. {item.get('headline', 'No headline')[:60]}...")
        else:
            print("[FAILED] No news data")
    except Exception as e:
        print(f"[ERROR] News test: {e}")

def test_fallback_logic():
    """Test fallback logic between APIs"""
    print("\n" + "=" * 80)
    print("Testing Fallback Logic")
    print("=" * 80)
    
    # Test with valid symbol
    print(f"\n1. Testing with valid symbol (GOOGL)...")
    try:
        quote = multi_source_data_manager.get_live_quote("GOOGL")
        if quote:
            print(f"[SUCCESS] Quote retrieved from {quote.get('source', 'unknown')}: ${quote['price']}")
        else:
            print("[FAILED] No quote retrieved")
    except Exception as e:
        print(f"[ERROR] Valid symbol test: {e}")
    
    # Test with invalid symbol
    print(f"\n2. Testing with invalid symbol (INVALID)...")
    try:
        quote = multi_source_data_manager.get_live_quote("INVALID")
        if quote:
            print(f"[SUCCESS] Unexpected quote retrieved: {quote.get('source', 'unknown')}")
        else:
            print("[SUCCESS] Correctly returned None for invalid symbol")
    except Exception as e:
        print(f"[ERROR] Invalid symbol test: {e}")

def test_statistics_and_health():
    """Test statistics and health check"""
    print("\n" + "=" * 80)
    print("Testing Statistics and Health Check")
    print("=" * 80)
    
    # Test data statistics
    print(f"\n1. Testing data statistics...")
    try:
        stats = multi_source_data_manager.get_data_statistics()
        if stats:
            print("[SUCCESS] Data statistics retrieved:")
            print(f"   Database size: {stats['database_stats'].get('db_size', 'Unknown')}")
            print(f"   Quotes count: {stats['database_stats'].get('quotes_count', 0)}")
            print(f"   Historical count: {stats['database_stats'].get('historical_count', 0)}")
            print(f"   Companies count: {stats['database_stats'].get('companies_count', 0)}")
            print(f"   API Status: {stats['api_status']}")
            print(f"   Priority Order: {stats['data_sources']['priority_order']}")
        else:
            print("[FAILED] No statistics retrieved")
    except Exception as e:
        print(f"[ERROR] Statistics test: {e}")
    
    # Test health check
    print(f"\n2. Testing health check...")
    try:
        health = multi_source_data_manager.health_check()
        if health:
            print("[SUCCESS] Health check completed:")
            print(f"   Overall status: {health['overall_status']}")
            print(f"   Database status: {health['database']['status']}")
            print(f"   API sources:")
            for source, status in health['api_sources'].items():
                print(f"     {source}: {status['status']}")
        else:
            print("[FAILED] No health data retrieved")
    except Exception as e:
        print(f"[ERROR] Health check test: {e}")

def test_api_comparison():
    """Test and compare different APIs"""
    print("\n" + "=" * 80)
    print("API Comparison Test")
    print("=" * 80)
    
    test_symbol = "TSLA"
    
    print(f"\nComparing APIs for {test_symbol}:")
    print("-" * 60)
    
    apis = [
        ("Yahoo Finance", yahoo_finance.get_live_quote),
        ("Finnhub", finnhub_api.get_quote),
        ("Polygon.io", polygon_api.get_previous_close),
        ("Alpha Vantage", alpha_vantage.get_global_quote)
    ]
    
    results = []
    
    for api_name, api_func in apis:
        try:
            start_time = datetime.now()
            quote = api_func(test_symbol)
            end_time = datetime.now()
            
            if quote:
                response_time = (end_time - start_time).total_seconds()
                results.append({
                    'api': api_name,
                    'price': quote.get('price', 0),
                    'response_time': round(response_time, 2),
                    'status': 'SUCCESS'
                })
                print(f"{api_name:15} | Price: ${quote.get('price', 0):8.2f} | Time: {response_time:.2f}s")
            else:
                results.append({
                    'api': api_name,
                    'price': 0,
                    'response_time': 0,
                    'status': 'NO_DATA'
                })
                print(f"{api_name:15} | No data returned")
                
        except Exception as e:
            results.append({
                'api': api_name,
                'price': 0,
                'response_time': 0,
                'status': f'ERROR: {str(e)[:30]}'
            })
            print(f"{api_name:15} | ERROR: {str(e)[:40]}")
    
    # Summary
    print("\n" + "-" * 60)
    successful_apis = [r for r in results if r['status'] == 'SUCCESS']
    if successful_apis:
        fastest_api = min(successful_apis, key=lambda x: x['response_time'])
        print(f"Fastest API: {fastest_api['api']} ({fastest_api['response_time']}s)")
        
        # Check price consistency
        prices = [r['price'] for r in successful_apis if r['price'] > 0]
        if len(prices) > 1:
            price_variance = max(prices) - min(prices)
            print(f"Price variance: ${price_variance:.2f}")

def main():
    """Run all tests"""
    print("Multi-Source Data Manager Test Suite")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        # Test individual APIs
        test_individual_apis()
        
        # Test multi-source manager
        test_multi_source_manager()
        
        # Test fallback logic
        test_fallback_logic()
        
        # Test statistics and health
        test_statistics_and_health()
        
        # Test API comparison
        test_api_comparison()
        
        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")

if __name__ == "__main__":
    main()
