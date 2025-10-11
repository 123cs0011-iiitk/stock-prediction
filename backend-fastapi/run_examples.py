#!/usr/bin/env python3
"""
Stock Price Insight Arena - API Testing Examples
Example scripts demonstrating how to use the FastAPI endpoints.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any
import time


class APITester:
    """API testing class with example usage"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.base_url}{endpoint}"
        try:
            async with self.session.request(method, url, **kwargs) as response:
                data = await response.json()
                return {
                    "status_code": response.status,
                    "data": data,
                    "success": response.status < 400
                }
        except Exception as e:
            return {
                "status_code": 0,
                "data": {"error": str(e)},
                "success": False
            }
    
    async def test_health_checks(self):
        """Test all health check endpoints"""
        print("🏥 Testing Health Check Endpoints")
        print("=" * 50)
        
        endpoints = [
            "/health",
            "/api/v1/health",
            "/api/v1/health/services",
            "/api/v1/health/detailed"
        ]
        
        for endpoint in endpoints:
            result = await self.make_request("GET", endpoint)
            status = "✅" if result["success"] else "❌"
            print(f"{status} {endpoint} - Status: {result['status_code']}")
            if not result["success"]:
                print(f"   Error: {result['data']}")
        
        print()
    
    async def test_stock_endpoints(self):
        """Test stock data endpoints"""
        print("📈 Testing Stock Data Endpoints")
        print("=" * 50)
        
        # Test stock quote
        print("Testing stock quote...")
        result = await self.make_request("GET", "/api/v1/stocks/quote/AAPL")
        if result["success"]:
            quote = result["data"]["data"]
            print(f"✅ AAPL Quote: ${quote['price']} ({quote['change']:+.2f})")
        else:
            print(f"❌ Quote failed: {result['data']}")
        
        # Test complete stock data
        print("Testing complete stock data...")
        result = await self.make_request("GET", "/api/v1/stocks/data/AAPL")
        if result["success"]:
            data = result["data"]["data"]
            print(f"✅ AAPL Data: ${data['quote']['price']}, {len(data['historical_data'])} historical points")
        else:
            print(f"❌ Complete data failed: {result['data']}")
        
        # Test historical data
        print("Testing historical data...")
        result = await self.make_request("GET", "/api/v1/stocks/historical/AAPL?limit=10")
        if result["success"]:
            historical = result["data"]["data"]
            print(f"✅ Historical Data: {len(historical)} data points")
        else:
            print(f"❌ Historical data failed: {result['data']}")
        
        # Test stock search
        print("Testing stock search...")
        result = await self.make_request("GET", "/api/v1/stocks/search?query=Apple")
        if result["success"]:
            search_results = result["data"]["results"]
            print(f"✅ Search Results: {len(search_results)} stocks found")
        else:
            print(f"❌ Search failed: {result['data']}")
        
        # Test batch quotes
        print("Testing batch quotes...")
        result = await self.make_request("GET", "/api/v1/stocks/batch-quotes?symbols=AAPL,GOOGL,MSFT")
        if result["success"]:
            quotes = result["data"]["quotes"]
            print(f"✅ Batch Quotes: {len(quotes)} quotes retrieved")
        else:
            print(f"❌ Batch quotes failed: {result['data']}")
        
        print()
    
    async def test_prediction_endpoints(self):
        """Test prediction endpoints"""
        print("🤖 Testing Prediction Endpoints")
        print("=" * 50)
        
        # Test prediction
        print("Testing stock prediction...")
        result = await self.make_request("GET", "/api/v1/predictions/predict/AAPL")
        if result["success"]:
            prediction = result["data"]["data"]
            print(f"✅ AAPL Prediction: ${prediction['final_prediction']:.2f}")
            print(f"   Confidence: {prediction['confidence']:.1f}%")
            print(f"   Trend: {prediction['trend']}")
            print(f"   Risk Score: {prediction['risk_score']:.1f}")
        else:
            print(f"❌ Prediction failed: {result['data']}")
        
        # Test model performance
        print("Testing model performance...")
        result = await self.make_request("GET", "/api/v1/predictions/models/performance/AAPL")
        if result["success"]:
            performance = result["data"]["data"]
            print(f"✅ Model Performance: {performance['data_points']} data points")
        else:
            print(f"❌ Model performance failed: {result['data']}")
        
        # Test available models
        print("Testing available models...")
        result = await self.make_request("GET", "/api/v1/predictions/models/available")
        if result["success"]:
            models = result["data"]["models"]
            print(f"✅ Available Models: {list(models.keys())}")
        else:
            print(f"❌ Available models failed: {result['data']}")
        
        # Test batch predictions
        print("Testing batch predictions...")
        result = await self.make_request("GET", "/api/v1/predictions/batch-predict?symbols=AAPL,GOOGL")
        if result["success"]:
            predictions = result["data"]["predictions"]
            print(f"✅ Batch Predictions: {len(predictions)} predictions generated")
        else:
            print(f"❌ Batch predictions failed: {result['data']}")
        
        print()
    
    async def test_error_handling(self):
        """Test error handling"""
        print("⚠️ Testing Error Handling")
        print("=" * 50)
        
        # Test invalid symbol
        print("Testing invalid symbol...")
        result = await self.make_request("GET", "/api/v1/stocks/quote/INVALID")
        if not result["success"]:
            print(f"✅ Invalid symbol handled correctly: {result['status_code']}")
        else:
            print(f"❌ Invalid symbol should have failed")
        
        # Test invalid endpoint
        print("Testing invalid endpoint...")
        result = await self.make_request("GET", "/api/v1/invalid/endpoint")
        if not result["success"]:
            print(f"✅ Invalid endpoint handled correctly: {result['status_code']}")
        else:
            print(f"❌ Invalid endpoint should have failed")
        
        print()
    
    async def performance_test(self):
        """Test API performance"""
        print("⚡ Testing API Performance")
        print("=" * 50)
        
        # Test multiple concurrent requests
        endpoints = [
            "/api/v1/stocks/quote/AAPL",
            "/api/v1/stocks/quote/GOOGL",
            "/api/v1/stocks/quote/MSFT",
            "/api/v1/predictions/predict/AAPL",
            "/api/v1/health"
        ]
        
        start_time = time.time()
        
        tasks = []
        for endpoint in endpoints:
            task = self.make_request("GET", endpoint)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful_requests = sum(1 for r in results if r["success"])
        
        print(f"✅ Concurrent Requests: {len(endpoints)} requests in {duration:.2f}s")
        print(f"   Successful: {successful_requests}/{len(endpoints)}")
        print(f"   Average response time: {duration/len(endpoints):.2f}s")
        
        print()


async def main():
    """Main testing function"""
    print("🚀 Stock Price Insight Arena - API Testing")
    print("=" * 60)
    print()
    
    async with APITester() as tester:
        # Run all tests
        await tester.test_health_checks()
        await tester.test_stock_endpoints()
        await tester.test_prediction_endpoints()
        await tester.test_error_handling()
        await tester.performance_test()
    
    print("🎉 API Testing Complete!")
    print()
    print("📚 Next Steps:")
    print("1. Check the interactive documentation at: http://localhost:8000/docs")
    print("2. Try the API endpoints with your own data")
    print("3. Integrate with your frontend application")
    print("4. Configure additional API keys for enhanced functionality")


if __name__ == "__main__":
    asyncio.run(main())
