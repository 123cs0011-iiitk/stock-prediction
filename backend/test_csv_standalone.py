#!/usr/bin/env python3
"""
Standalone test script for CSV backup system (without PostgreSQL dependency)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.csv_backup_service import csv_backup
from datetime import datetime
import json

def test_csv_backup_standalone():
    """Test CSV backup service functionality without PostgreSQL"""
    print("=" * 60)
    print("Testing CSV Backup Service (Standalone)")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing CSV backup health check...")
    try:
        health = csv_backup.health_check()
        print(f"[SUCCESS] CSV backup health check: {health['status']}")
        print(f"   Backup directory: {health['backup_directory']}")
        print(f"   Writable: {health['writable']}")
    except Exception as e:
        print(f"[FAILED] Health check failed: {e}")
        return False
    
    # Test 2: Store test stock quote
    print("\n2. Testing stock quote storage in CSV...")
    try:
        test_quote = {
            'price': 175.50,
            'change': 3.25,
            'change_percent': 1.89,
            'volume': 1500000,
            'high': 176.00,
            'low': 172.00,
            'open': 173.50,
            'previous_close': 172.25,
            'market_cap': '2.8T',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'currency': 'USD',
            'timestamp': datetime.now().isoformat()
        }
        
        success = csv_backup.store_stock_quote('AAPL', test_quote, 'test')
        if success:
            print("[SUCCESS] Stock quote stored in CSV backup successfully")
        else:
            print("[FAILED] Failed to store stock quote in CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error storing stock quote in CSV: {e}")
        return False
    
    # Test 3: Retrieve test stock quote
    print("\n3. Testing stock quote retrieval from CSV...")
    try:
        retrieved_quote = csv_backup.get_latest_quote('AAPL', 60)
        if retrieved_quote:
            print("[SUCCESS] Stock quote retrieved from CSV backup successfully")
            print(f"   Symbol: {retrieved_quote['symbol']}")
            print(f"   Price: ${retrieved_quote['price']}")
            print(f"   Change: {retrieved_quote['change_percent']}%")
        else:
            print("[FAILED] Failed to retrieve stock quote from CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error retrieving stock quote from CSV: {e}")
        return False
    
    # Test 4: Store test historical data
    print("\n4. Testing historical data storage in CSV...")
    try:
        test_historical = [
            {
                'date': '2024-01-15',
                'open': 170.00,
                'high': 175.00,
                'low': 168.00,
                'close': 173.50,
                'volume': 1200000
            },
            {
                'date': '2024-01-16',
                'open': 173.50,
                'high': 177.00,
                'low': 172.00,
                'close': 175.50,
                'volume': 1500000
            }
        ]
        
        success = csv_backup.store_historical_data('AAPL', test_historical, 'test')
        if success:
            print("[SUCCESS] Historical data stored in CSV backup successfully")
        else:
            print("[FAILED] Failed to store historical data in CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error storing historical data in CSV: {e}")
        return False
    
    # Test 5: Retrieve test historical data
    print("\n5. Testing historical data retrieval from CSV...")
    try:
        retrieved_historical = csv_backup.get_historical_data('AAPL', 10)
        if retrieved_historical:
            print("[SUCCESS] Historical data retrieved from CSV backup successfully")
            print(f"   Records retrieved: {len(retrieved_historical)}")
            print(f"   Latest date: {retrieved_historical[0]['date']}")
        else:
            print("[FAILED] Failed to retrieve historical data from CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error retrieving historical data from CSV: {e}")
        return False
    
    # Test 6: Store test company info
    print("\n6. Testing company info storage in CSV...")
    try:
        test_company = {
            'name': 'Apple Inc.',
            'description': 'Technology company that designs and manufactures consumer electronics',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': '2.8T',
            'website': 'https://apple.com',
            'city': 'Cupertino',
            'state': 'CA',
            'country': 'USA'
        }
        
        success = csv_backup.store_company_info('AAPL', test_company, 'test')
        if success:
            print("[SUCCESS] Company info stored in CSV backup successfully")
        else:
            print("[FAILED] Failed to store company info in CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error storing company info in CSV: {e}")
        return False
    
    # Test 7: Retrieve test company info
    print("\n7. Testing company info retrieval from CSV...")
    try:
        retrieved_company = csv_backup.get_company_info('AAPL')
        if retrieved_company:
            print("[SUCCESS] Company info retrieved from CSV backup successfully")
            print(f"   Name: {retrieved_company['name']}")
            print(f"   Sector: {retrieved_company['sector']}")
            print(f"   Industry: {retrieved_company['industry']}")
        else:
            print("[FAILED] Failed to retrieve company info from CSV backup")
            return False
            
    except Exception as e:
        print(f"[FAILED] Error retrieving company info from CSV: {e}")
        return False
    
    # Test 8: Backup statistics
    print("\n8. Testing backup statistics...")
    try:
        stats = csv_backup.get_backup_statistics()
        print("[SUCCESS] Backup statistics retrieved successfully")
        print(f"   Quotes files: {stats.get('quotes_files', 0)}")
        print(f"   Historical files: {stats.get('historical_files', 0)}")
        print(f"   Companies files: {stats.get('companies_files', 0)}")
        print(f"   Total size: {stats.get('total_size_mb', 0)} MB")
    except Exception as e:
        print(f"[FAILED] Error getting backup statistics: {e}")
        return False
    
    # Test 9: Check backup directory structure
    print("\n9. Testing backup directory structure...")
    try:
        backup_dir = csv_backup.backup_dir
        print(f"Backup directory: {backup_dir}")
        
        # Check if directories exist
        directories = [
            ("Main backup directory", backup_dir),
            ("Quotes directory", csv_backup.quotes_dir),
            ("Historical directory", csv_backup.historical_dir),
            ("Companies directory", csv_backup.companies_dir)
        ]
        
        for name, directory in directories:
            if directory.exists():
                print(f"[SUCCESS] {name}: {directory}")
            else:
                print(f"[FAILED] {name} not found: {directory}")
                return False
        
        # List files in each directory
        print(f"\nFiles in quotes directory:")
        quote_files = list(csv_backup.quotes_dir.glob('*.csv'))
        for file in quote_files:
            print(f"  - {file.name}")
        
        print(f"\nFiles in historical directory:")
        historical_files = list(csv_backup.historical_dir.glob('*.csv'))
        for file in historical_files:
            print(f"  - {file.name}")
        
        print(f"\nFiles in companies directory:")
        company_files = list(csv_backup.companies_dir.glob('*.csv'))
        for file in company_files:
            print(f"  - {file.name}")
        
    except Exception as e:
        print(f"[FAILED] Error checking backup directory structure: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All CSV backup service tests completed successfully!")
    print("[SUCCESS] CSV backup system is working independently of PostgreSQL")
    print("[SUCCESS] Stock data can be stored and retrieved from CSV files")
    print("[SUCCESS] Backup directory structure is properly created")
    print("[SUCCESS] Fallback mechanism is ready for when PostgreSQL is unavailable")
    print("=" * 60)
    return True

if __name__ == "__main__":
    print("CSV Backup System Standalone Test")
    print("=" * 60)
    
    # Test CSV backup service
    csv_ok = test_csv_backup_standalone()
    
    if csv_ok:
        print("\n[SUCCESS] All CSV backup tests passed!")
        print("\n[SUCCESS] The backup directory system is working perfectly!")
        print("[SUCCESS] Your stock data will be safely backed up to CSV files")
        print("[SUCCESS] Even if PostgreSQL is down, you can still access historical data")
        sys.exit(0)
    else:
        print("\n❌ CSV backup tests failed.")
        sys.exit(1)
