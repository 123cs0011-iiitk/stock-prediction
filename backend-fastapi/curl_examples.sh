#!/bin/bash

# Stock Price Insight Arena - cURL API Examples
# This script demonstrates how to use the FastAPI endpoints with cURL

BASE_URL="http://localhost:8000"

echo "🚀 Stock Price Insight Arena - API Examples with cURL"
echo "======================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to make API call and display result
make_api_call() {
    local description="$1"
    local endpoint="$2"
    local method="${3:-GET}"
    
    echo -e "${BLUE}📡 $description${NC}"
    echo -e "${YELLOW}curl -X $method \"$BASE_URL$endpoint\"${NC}"
    echo ""
    
    response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X "$method" "$BASE_URL$endpoint")
    http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
    body=$(echo "$response" | sed '/HTTP_STATUS:/d')
    
    if [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
        echo -e "${GREEN}✅ Success (HTTP $http_status)${NC}"
        echo "$body" | python -m json.tool 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ Error (HTTP $http_status)${NC}"
        echo "$body" | python -m json.tool 2>/dev/null || echo "$body"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
}

echo "🏥 1. HEALTH CHECK ENDPOINTS"
echo "=============================="

make_api_call "Simple Health Check" "/health"
make_api_call "Comprehensive Health Check" "/api/v1/health"
make_api_call "Services Health Check" "/api/v1/health/services"
make_api_call "Detailed Health Check" "/api/v1/health/detailed"

echo "📈 2. STOCK DATA ENDPOINTS"
echo "==========================="

make_api_call "Get AAPL Stock Quote" "/api/v1/stocks/quote/AAPL"
make_api_call "Get GOOGL Stock Quote with Currency" "/api/v1/stocks/quote/GOOGL?currency=USD"
make_api_call "Get Complete Stock Data for AAPL" "/api/v1/stocks/data/AAPL"
make_api_call "Get Historical Data for AAPL" "/api/v1/stocks/historical/AAPL?limit=10"
make_api_call "Search for Apple Stocks" "/api/v1/stocks/search?query=Apple&limit=5"
make_api_call "Get Company Info for AAPL" "/api/v1/stocks/company/AAPL"
make_api_call "Batch Quotes for Multiple Stocks" "/api/v1/stocks/batch-quotes?symbols=AAPL,GOOGL,MSFT"
make_api_call "Market Overview" "/api/v1/stocks/market-overview"
make_api_call "Available Symbols" "/api/v1/stocks/symbols/available"

echo "🤖 3. PREDICTION ENDPOINTS"
echo "==========================="

make_api_call "Generate Prediction for AAPL" "/api/v1/predictions/predict/AAPL"
make_api_call "Generate Prediction with Linear Regression" "/api/v1/predictions/predict/AAPL?models=linear_regression"
make_api_call "Generate Prediction with Random Forest" "/api/v1/predictions/predict/AAPL?models=random_forest"
make_api_call "Generate Ensemble Prediction" "/api/v1/predictions/predict/AAPL?models=ensemble"
make_api_call "Get Model Performance for AAPL" "/api/v1/predictions/models/performance/AAPL"
make_api_call "Get Available Models" "/api/v1/predictions/models/available"
make_api_call "Batch Predictions" "/api/v1/predictions/batch-predict?symbols=AAPL,GOOGL"
make_api_call "Trend Analysis for AAPL" "/api/v1/predictions/trend-analysis/AAPL"
make_api_call "Prediction History for AAPL" "/api/v1/predictions/prediction-history/AAPL"

echo "📊 4. POST REQUEST EXAMPLES"
echo "============================"

echo -e "${BLUE}📡 Generate Prediction via POST${NC}"
echo -e "${YELLOW}curl -X POST \"$BASE_URL/api/v1/predictions/predict\" \\${NC}"
echo -e "${YELLOW}     -H \"Content-Type: application/json\" \\${NC}"
echo -e "${YELLOW}     -d '{\"symbol\": \"AAPL\", \"models\": [\"ensemble\"], \"time_horizon\": \"1 day\"}'${NC}"
echo ""

response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$BASE_URL/api/v1/predictions/predict" \
    -H "Content-Type: application/json" \
    -d '{"symbol": "AAPL", "models": ["ensemble"], "time_horizon": "1 day"}')

http_status=$(echo "$response" | grep "HTTP_STATUS:" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_STATUS:/d')

if [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
    echo -e "${GREEN}✅ Success (HTTP $http_status)${NC}"
    echo "$body" | python -m json.tool 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ Error (HTTP $http_status)${NC}"
    echo "$body" | python -m json.tool 2>/dev/null || echo "$body"
fi

echo ""
echo "----------------------------------------"
echo ""

echo "⚠️  5. ERROR HANDLING EXAMPLES"
echo "==============================="

make_api_call "Invalid Stock Symbol" "/api/v1/stocks/quote/INVALID"
make_api_call "Invalid Endpoint" "/api/v1/invalid/endpoint"
make_api_call "Missing Query Parameter" "/api/v1/stocks/search"

echo "🔍 6. QUERY PARAMETER EXAMPLES"
echo "==============================="

make_api_call "Historical Data with Date Range" "/api/v1/stocks/historical/AAPL?start_date=2023-01-01&end_date=2023-12-31&limit=5"
make_api_call "Search with Quotes" "/api/v1/stocks/search?query=Microsoft&include_quotes=true"
make_api_call "Prediction with Custom Parameters" "/api/v1/predictions/predict/AAPL?models=ensemble&time_horizon=1%20week&include_confidence=true"

echo "📱 7. FRONTEND INTEGRATION EXAMPLES"
echo "===================================="

echo -e "${BLUE}📡 JavaScript/Fetch Example${NC}"
echo -e "${YELLOW}fetch('$BASE_URL/api/v1/stocks/quote/AAPL')${NC}"
echo -e "${YELLOW}  .then(response => response.json())${NC}"
echo -e "${YELLOW}  .then(data => console.log(data));${NC}"
echo ""

echo -e "${BLUE}📡 Python Requests Example${NC}"
echo -e "${YELLOW}import requests${NC}"
echo -e "${YELLOW}response = requests.get('$BASE_URL/api/v1/stocks/quote/AAPL')${NC}"
echo -e "${YELLOW}print(response.json())${NC}"
echo ""

echo -e "${BLUE}📡 React/Axios Example${NC}"
echo -e "${YELLOW}const response = await axios.get('$BASE_URL/api/v1/stocks/quote/AAPL');${NC}"
echo -e "${YELLOW}console.log(response.data);${NC}"
echo ""

echo "🎉 API Testing Complete!"
echo ""
echo "📚 Next Steps:"
echo "1. Open interactive documentation: $BASE_URL/docs"
echo "2. Try the ReDoc interface: $BASE_URL/redoc"
echo "3. View OpenAPI specification: $BASE_URL/openapi.json"
echo "4. Integrate with your frontend application"
echo "5. Configure additional API keys for enhanced functionality"
echo ""
echo "💡 Tips:"
echo "- Use the interactive documentation at /docs for easy testing"
echo "- All endpoints return JSON responses"
echo "- Error responses include detailed error messages"
echo "- Rate limiting is enabled for API protection"
echo "- CORS is configured for frontend integration"
echo ""
