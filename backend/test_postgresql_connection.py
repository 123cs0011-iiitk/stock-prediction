#!/usr/bin/env python3
"""
Test script to verify PostgreSQL database connection and functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.postgresql_database import postgres_db
from datetime import datetime
import json

def test_postgresql_connection():
    """Test PostgreSQL database connection and basic operations"""
    print("=" * 60)
    print("Testing PostgreSQL Database Connection")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing database health check...")
    try:
        health = postgres_db.health_check()
        print(f"[SUCCESS] Database health check: {health['status']}")
        print(f"   Connection pool size: {health.get('connection_pool_size', 'N/A')}")
        print(f"   Checked out connections: {health.get('checked_out_connections', 'N/A')}")
    except Exception as e:
        print(f"[FAILED] Health check failed: {e}")
        return False
    
    # Test 2: Store test stock quote
    print("\n2. Testing stock quote storage...")
    try:
        test_quote = {
            'price': 150.25,
            'change': 2.50,
            'change_percent': 1.69,
            'volume': 1000000,
            'high': 152.00,
            'low': 148.00,
            'open': 149.50,
            'previous_close': 147.75,
            'market_cap': '2.5T',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'currency': 'USD',
            'timestamp': datetime.now().isoformat()
        }
        
        success = postgres_db.store_stock_quote('TEST', test_quote, 'test')
        if success:
            print("[SUCCESS] Stock quote stored successfully")
        else:
            print("[FAILED] Failed to store stock quote")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error storing stock quote: {e}")
        return False
    
    # Test 3: Retrieve test stock quote
    print("\n3. Testing stock quote retrieval...")
    try:
        retrieved_quote = postgres_db.get_latest_quote('TEST', 60)
        if retrieved_quote:
            print("[SUCCESS] Stock quote retrieved successfully")
            print(f"   Symbol: {retrieved_quote['symbol']}")
            print(f"   Price: ${retrieved_quote['price']}")
            print(f"   Change: {retrieved_quote['change_percent']}%")
        else:
            print("[FAILED] Failed to retrieve stock quote")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error retrieving stock quote: {e}")
        return False
    
    # Test 4: Store test company info
    print("\n4. Testing company info storage...")
    try:
        test_company = {
            'name': 'Test Company Inc.',
            'description': 'A test company for database testing',
            'sector': 'Technology',
            'industry': 'Software',
            'market_cap': '100B',
            'website': 'https://testcompany.com',
            'city': 'San Francisco',
            'state': 'CA',
            'country': 'USA'
        }
        
        success = postgres_db.store_company_info('TEST', test_company, 'test')
        if success:
            print("[SUCCESS] Company info stored successfully")
        else:
            print("[FAILED] Failed to store company info")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error storing company info: {e}")
        return False
    
    # Test 5: Retrieve test company info
    print("\n5. Testing company info retrieval...")
    try:
        retrieved_company = postgres_db.get_company_info('TEST')
        if retrieved_company:
            print("[SUCCESS] Company info retrieved successfully")
            print(f"   Name: {retrieved_company['name']}")
            print(f"   Sector: {retrieved_company['sector']}")
            print(f"   Industry: {retrieved_company['industry']}")
        else:
            print("[FAILED] Failed to retrieve company info")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error retrieving company info: {e}")
        return False
    
    # Test 6: Database statistics
    print("\n6. Testing database statistics...")
    try:
        stats = postgres_db.get_database_stats()
        print("[SUCCESS] Database statistics retrieved")
        print(f"   Quotes count: {stats.get('quotes_count', 0)}")
        print(f"   Historical data count: {stats.get('historical_count', 0)}")
        print(f"   Companies count: {stats.get('companies_count', 0)}")
        print(f"   Unique symbols: {stats.get('unique_symbols', 0)}")
        print(f"   Database size: {stats.get('db_size', 'Unknown')}")
    except Exception as e:
        print(f"[FAILED] Error getting database statistics: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All PostgreSQL tests completed successfully!")
    print("=" * 60)
    return True

def test_environment_configuration():
    """Test environment configuration for PostgreSQL"""
    print("\n" + "=" * 60)
    print("Testing Environment Configuration")
    print("=" * 60)
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check required environment variables
    required_vars = [
        'DATABASE_URL',
        'DB_HOST', 
        'DB_PORT',
        'DB_NAME',
        'DB_USER',
        'DB_PASSWORD'
    ]
    
    print("\nChecking environment variables...")
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if 'PASSWORD' in var:
                print(f"[OK] {var}: {'*' * len(value)}")
            else:
                print(f"[OK] {var}: {value}")
        else:
            print(f"[MISSING] {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n[WARNING] Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        return False
    else:
        print("\n[SUCCESS] All required environment variables are set")
        return True

if __name__ == "__main__":
    print("PostgreSQL Database Test Suite")
    print("=" * 60)
    
    # Test environment configuration first
    env_ok = test_environment_configuration()
    
    if env_ok:
        # Test database connection and functionality
        db_ok = test_postgresql_connection()
        
        if db_ok:
            print("\n🎉 All tests passed! PostgreSQL is properly configured.")
            sys.exit(0)
        else:
            print("\n❌ Database tests failed. Check your PostgreSQL setup.")
            sys.exit(1)
    else:
        print("\n❌ Environment configuration is incomplete.")
        print("Please update your .env file with the required PostgreSQL settings.")
        sys.exit(1)
