# Alpha Vantage API Integration Setup

This document explains how to set up and use the Alpha Vantage API integration for real-time stock prices in the Stock Prediction project.

## Overview

The project now uses Alpha Vantage API to fetch real-time stock prices, replacing the previous Yahoo Finance implementation. The integration includes:

- ✅ Real-time stock price fetching
- ✅ Company information and overview
- ✅ Rate limiting and error handling
- ✅ Fallback to mock data when API is unavailable
- ✅ Frontend integration with live price display

## Setup Instructions

### 1. Get Alpha Vantage API Key

1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. Get your API key (free tier: 500 calls/day, 5 calls/minute)

### 2. Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp backend/env.example backend/.env
   ```

2. Edit `backend/.env` and add your API key:
   ```env
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   ```

### 3. Install Dependencies

```bash
# Backend dependencies (should already be installed)
cd backend
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install
```

### 4. Test the Integration

Run the test script to verify everything is working:

```bash
cd backend
python test_alpha_vantage.py
```

This will test:
- API key configuration
- Alpha Vantage API connectivity
- Backend endpoint functionality

## Usage

### Backend API Endpoints

#### Get Live Stock Price
```http
GET /api/stock/price/{symbol}
```

**Example:**
```bash
curl http://localhost:5000/api/stock/price/AAPL
```

**Response:**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "price": 173.20,
  "change": 2.15,
  "changePercent": 1.26,
  "volume": 45678900,
  "high": 174.50,
  "low": 171.30,
  "open": 172.10,
  "previousClose": 171.05,
  "marketCap": "2800000000000",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "currency": "USD",
  "timestamp": "2025-01-11T14:30:00",
  "source": "Alpha Vantage"
}
```

### Frontend Integration

The frontend automatically uses live prices when available. The `stockService.getLiveStockPrice()` method:

1. Tries to fetch live price from Alpha Vantage
2. Falls back to mock data if API is unavailable
3. Handles rate limiting gracefully

**Usage in components:**
```typescript
import { stockService } from '../services/stockService';

// Get live stock price
const stockData = await stockService.getLiveStockPrice('AAPL');
console.log(`Current price: $${stockData.price}`);
```

## Rate Limiting

The free Alpha Vantage tier has limits:
- **500 calls per day**
- **5 calls per minute**

The integration handles this by:
- Adding 12-second delays between API calls
- Returning appropriate error messages when limits are exceeded
- Falling back to mock data when needed

## Error Handling

The system handles various error scenarios:

### API Key Issues
```json
{
  "error": "ALPHA_VANTAGE_API_KEY environment variable is required"
}
```

### Rate Limit Exceeded
```json
{
  "error": "API rate limit exceeded. Please try again in a few minutes.",
  "retry_after": 300
}
```

### Invalid Stock Symbol
```json
{
  "error": "Stock symbol \"INVALID\" not found or no data available"
}
```

### Network Issues
- Automatic retry with exponential backoff
- Graceful fallback to mock data
- User-friendly error messages

## Testing

### Backend Testing
```bash
# Test Alpha Vantage integration
cd backend
python test_alpha_vantage.py

# Test specific endpoint
curl http://localhost:5000/api/stock/price/AAPL
```

### Frontend Testing
1. Start the frontend: `npm run start`
2. Open browser console
3. Load the test script: `frontend/test-live-price.js`
4. Run: `testLiveStockPrice()`

### Manual Testing
1. Start backend: `python backend/app.py`
2. Start frontend: `npm run start`
3. Open http://localhost:3000
4. Search for any stock symbol (e.g., AAPL, GOOGL, MSFT)
5. Verify live prices are displayed

## Troubleshooting

### Common Issues

**1. "API key not set" error**
- Ensure `.env` file exists in `backend/` directory
- Verify `ALPHA_VANTAGE_API_KEY` is set correctly
- Restart the backend server after changing `.env`

**2. "Rate limit exceeded" error**
- Wait 12 seconds between requests
- Check your daily API usage at Alpha Vantage dashboard
- Consider upgrading to a paid plan for higher limits

**3. "Stock not found" error**
- Verify the stock symbol is correct (e.g., AAPL, not apple)
- Check if the stock is traded on major exchanges
- Some symbols might not be available in Alpha Vantage

**4. Frontend not showing live prices**
- Check browser console for errors
- Verify backend is running on port 5000
- Test the API endpoint directly with curl

### Debug Mode

Enable debug logging by setting:
```env
FLASK_DEBUG=true
```

This will show detailed error messages and API responses.

## File Structure

```
backend/
├── services/
│   └── alpha_vantage.py      # Alpha Vantage API client
├── app.py                    # Main Flask app with live price endpoint
├── test_alpha_vantage.py     # Integration test script
└── .env                      # Environment variables

frontend/
├── src/
│   ├── services/
│   │   └── stockService.ts   # Updated with live price methods
│   └── App.tsx              # Updated to use live prices
└── test-live-price.js       # Frontend test script
```

## API Reference

### Alpha Vantage Service Methods

```python
from services.alpha_vantage import alpha_vantage

# Get real-time quote
quote = alpha_vantage.get_global_quote('AAPL')

# Get company overview
company = alpha_vantage.get_company_overview('AAPL')

# Get daily time series
history = alpha_vantage.get_daily_time_series('AAPL')
```

### Stock Service Methods

```typescript
import { stockService } from '../services/stockService';

// Get live stock price
const liveData = await stockService.getLiveStockPrice('AAPL');

// Search with live prices
const results = await stockService.searchStocksWithLivePrice('AAPL');
```

## Migration from Yahoo Finance

The following changes were made to replace Yahoo Finance:

1. **Removed Yahoo Finance references** from requirements.txt and comments
2. **Added Alpha Vantage service** with proper rate limiting
3. **Created new endpoint** `/api/stock/price/{symbol}` for live prices
4. **Updated frontend service** to use live prices with fallback
5. **Enhanced error handling** for API limits and network issues

## Support

For issues related to:
- **Alpha Vantage API**: Check their [documentation](https://www.alphavantage.co/documentation/)
- **Rate limits**: Consider upgrading your Alpha Vantage plan
- **Integration issues**: Check the test scripts and error logs
- **Frontend problems**: Verify backend connectivity and API responses

## Next Steps

Future enhancements could include:
- Historical data integration
- Multiple data source fallbacks
- Caching for better performance
- Real-time price updates
- WebSocket integration for live feeds
