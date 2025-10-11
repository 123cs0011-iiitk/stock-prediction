#!/usr/bin/env python3
"""
Simple test script for Stock Prediction API
Run this after starting the Flask server to test the endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:9000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"❌ Health Check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check error: {e}")
        return False

def test_stock_search():
    """Test the stock search endpoint"""
    print("\n🔍 Testing Stock Search...")
    try:
        response = requests.get(f"{BASE_URL}/api/search?q=AAPL")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stock Search: Found {data['symbol']} - {data['name']}")
            print(f"   Price: ${data['price']}")
            print(f"   Change: {data['change']} ({data['changePercent']}%)")
            return True
        else:
            print(f"❌ Stock Search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stock Search error: {e}")
        return False

def test_available_stocks():
    """Test multiple available stocks"""
    print("\n🔍 Testing Available Stocks...")
    available_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    
    for symbol in available_stocks:
        try:
            response = requests.get(f"{BASE_URL}/api/search?q={symbol}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ {symbol}: ${data['price']} ({data['name']})")
            else:
                print(f"❌ {symbol}: Failed")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    return True

def test_stock_data():
    """Test the stock data endpoint"""
    print("\n🔍 Testing Stock Data...")
    try:
        response = requests.get(f"{BASE_URL}/api/stock/AAPL")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stock Data: {data['symbol']} - {data['info']['name']}")
            print(f"   Current Price: ${data['info']['price']}")
            print(f"   Market Cap: ${data['info']['marketCap']:,}")
            print(f"   Historical Data Points: {len(data['historical']['dates'])}")
            print(f"   Technical Indicators: RSI={data['technical']['rsi']:.1f}, Volatility={data['technical']['volatility']:.1f}%")
            return True
        else:
            print(f"❌ Stock Data failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stock Data error: {e}")
        return False

def test_predictions():
    """Test the predictions endpoint"""
    print("\n🔍 Testing Predictions...")
    try:
        response = requests.get(f"{BASE_URL}/api/predict/AAPL")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Predictions: {data['symbol']}")
            print(f"   Trend: {data['prediction']['trend'].upper()}")
            print(f"   Confidence: {data['prediction']['confidence']}%")
            print(f"   Support: ${data['prediction']['support_level']}")
            print(f"   Resistance: ${data['prediction']['resistance_level']}")
            return True
        else:
            print(f"❌ Predictions failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Predictions error: {e}")
        return False

def test_portfolio_analysis():
    """Test the portfolio analysis endpoint"""
    print("\n🔍 Testing Portfolio Analysis...")
    try:
        portfolio_data = {
            "portfolio": [
                {
                    "symbol": "AAPL",
                    "shares": 100,
                    "costBasis": 15000.00
                },
                {
                    "symbol": "GOOGL",
                    "shares": 50,
                    "costBasis": 7500.00
                }
            ]
        }
        
        response = requests.post(
            f"{BASE_URL}/api/portfolio/analyze",
            json=portfolio_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Portfolio Analysis:")
            print(f"   Total Cost: ${data['summary']['totalCost']:,.2f}")
            print(f"   Total Value: ${data['summary']['totalValue']:,.2f}")
            print(f"   Total Return: {data['summary']['totalReturnPercent']}%")
            print(f"   Diversification Score: {data['risk_metrics']['diversification_score']}/100")
            return True
        else:
            print(f"❌ Portfolio Analysis failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Portfolio Analysis error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Stock Prediction API Test Suite")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_stock_search,
        test_available_stocks,
        test_stock_data,
        test_predictions,
        test_portfolio_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the server logs for details.")
    
    print("\n💡 Next steps:")
    print("   1. Open index.html in your browser")
    print("   2. Search for a stock symbol (e.g., AAPL, GOOGL, MSFT)")
    print("   3. View real-time data, charts, and predictions")
    print("   4. Add stocks to your portfolio and analyze performance")

if __name__ == "__main__":
    main()
