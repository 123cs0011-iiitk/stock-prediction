#!/usr/bin/env python3
"""
Test script for Alpha Vantage API integration
Run this script to verify the Alpha Vantage API is working correctly
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_alpha_vantage_integration():
    """Test the Alpha Vantage API integration"""
    
    # Check if API key is set
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        print("❌ ALPHA_VANTAGE_API_KEY not set in environment variables")
        print("Please set your Alpha Vantage API key in the .env file")
        print("Get your free API key at: https://www.alphavantage.co/support/#api-key")
        return False
    
    try:
        from services.alpha_vantage import alpha_vantage
        print("✅ Alpha Vantage service imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Alpha Vantage service: {e}")
        return False
    
    # Test with a popular stock symbol
    test_symbol = "AAPL"
    print(f"\n🔍 Testing with symbol: {test_symbol}")
    
    try:
        # Test global quote
        print("📊 Fetching real-time quote...")
        quote_data = alpha_vantage.get_global_quote(test_symbol)
        print("✅ Quote data retrieved successfully:")
        print(f"   Symbol: {quote_data['symbol']}")
        print(f"   Price: ${quote_data['price']}")
        print(f"   Change: ${quote_data['change']} ({quote_data['change_percent']}%)")
        print(f"   Volume: {quote_data['volume']:,}")
        print(f"   High: ${quote_data['high']}")
        print(f"   Low: ${quote_data['low']}")
        
    except Exception as e:
        print(f"❌ Failed to get quote data: {e}")
        return False
    
    try:
        # Test company overview
        print("\n🏢 Fetching company overview...")
        company_data = alpha_vantage.get_company_overview(test_symbol)
        print("✅ Company data retrieved successfully:")
        print(f"   Name: {company_data['name']}")
        print(f"   Sector: {company_data['sector']}")
        print(f"   Industry: {company_data['industry']}")
        print(f"   Market Cap: ${company_data['market_cap']}")
        
    except Exception as e:
        print(f"⚠️  Company overview failed (this is optional): {e}")
    
    print("\n🎉 Alpha Vantage integration test completed successfully!")
    print("\n📝 Next steps:")
    print("1. Start the backend server: python backend/app.py")
    print("2. Test the API endpoint: http://localhost:5000/api/stock/price/AAPL")
    print("3. Start the frontend: npm run start (in frontend directory)")
    
    return True

def test_backend_endpoint():
    """Test the backend endpoint locally"""
    print("\n🧪 Testing backend endpoint...")
    
    try:
        import requests
        import time
        
        # Wait a moment for rate limiting
        print("⏳ Waiting for rate limit...")
        time.sleep(2)
        
        response = requests.get("http://localhost:5000/api/stock/price/AAPL", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend endpoint working:")
            print(f"   Symbol: {data['symbol']}")
            print(f"   Price: ${data['price']}")
            print(f"   Source: {data['source']}")
            return True
        else:
            print(f"❌ Backend endpoint error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Backend server not running. Start it with: python backend/app.py")
        return False
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Alpha Vantage Integration Test")
    print("=" * 40)
    
    # Test Alpha Vantage integration
    if test_alpha_vantage_integration():
        # Test backend endpoint if server is running
        test_backend_endpoint()
    
    print("\n" + "=" * 40)
    print("✅ Test completed!")
