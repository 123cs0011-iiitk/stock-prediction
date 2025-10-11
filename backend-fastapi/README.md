# 🏆 Stock Price Insight Arena - FastAPI Backend

A comprehensive FastAPI backend for real-time stock data, historical analysis, and ML-powered price predictions.

## 🚀 Features

- **Real-time Stock Data**: Live prices from multiple data sources
- **Historical Data**: Comprehensive historical price data with technical indicators
- **ML Predictions**: Machine learning models for price forecasting
- **RESTful API**: Clean, documented API endpoints
- **Data Validation**: Pydantic models for request/response validation
- **Error Handling**: Comprehensive error handling and logging
- **Health Checks**: Multiple health check endpoints for monitoring
- **CORS Support**: Frontend integration ready
- **Rate Limiting**: Built-in rate limiting for API protection

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL (optional, for data persistence)
- Redis (optional, for caching)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd backend-fastapi
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python main.py
   # Or with uvicorn directly:
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```env
# Required
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Optional
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
```

### API Keys Setup

1. **Alpha Vantage**: Get free API key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)
2. **Finnhub**: Get free API key at [finnhub.io](https://finnhub.io/register)
3. **Polygon**: Get free API key at [polygon.io](https://polygon.io/dashboard)

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Core Endpoints

#### Health Checks
```bash
# Simple health check
GET /health

# Comprehensive health check
GET /api/v1/health

# Services health check
GET /api/v1/health/services
```

#### Stock Data
```bash
# Get real-time quote
GET /api/v1/stocks/quote/{symbol}

# Get complete stock data
GET /api/v1/stocks/data/{symbol}

# Get historical data
GET /api/v1/stocks/historical/{symbol}

# Search stocks
GET /api/v1/stocks/search?query=AAPL
```

#### Predictions
```bash
# Generate prediction
GET /api/v1/predictions/predict/{symbol}

# Get model performance
GET /api/v1/predictions/models/performance/{symbol}

# Available models
GET /api/v1/predictions/models/available
```

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Get stock quote
curl "http://localhost:8000/api/v1/stocks/quote/AAPL"

# Get prediction
curl "http://localhost:8000/api/v1/predictions/predict/AAPL"

# Search stocks
curl "http://localhost:8000/api/v1/stocks/search?query=Apple"
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get stock quote
response = requests.get("http://localhost:8000/api/v1/stocks/quote/AAPL")
print(response.json())

# Generate prediction
response = requests.get("http://localhost:8000/api/v1/predictions/predict/AAPL")
print(response.json())
```

### Using JavaScript/Fetch

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Get stock quote
fetch('http://localhost:8000/api/v1/stocks/quote/AAPL')
  .then(response => response.json())
  .then(data => console.log(data));

// Generate prediction
fetch('http://localhost:8000/api/v1/predictions/predict/AAPL')
  .then(response => response.json())
  .then(data => console.log(data));
```

## 🏗️ Project Structure

```
backend-fastapi/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── README.md              # This file
├── config/
│   └── settings.py        # Application configuration
├── models/
│   └── stock_models.py    # Pydantic data models
├── routes/
│   ├── stock_routes.py    # Stock data endpoints
│   ├── prediction_routes.py # Prediction endpoints
│   └── health_routes.py   # Health check endpoints
└── services/
    ├── stock_service.py   # Stock data business logic
    └── prediction_service.py # ML prediction logic
```

## 🔍 Key Components

### 1. Data Models (`models/stock_models.py`)

Pydantic models for data validation:
- `StockQuote`: Real-time stock quote
- `HistoricalData`: Historical price data
- `CompanyInfo`: Company information
- `PredictionResult`: ML prediction results
- `EnsemblePrediction`: Combined predictions

### 2. Services (`services/`)

Business logic layer:
- `StockService`: Handles stock data operations
- `PredictionService`: Manages ML predictions

### 3. Routes (`routes/`)

API endpoints:
- `stock_routes.py`: Stock data endpoints
- `prediction_routes.py`: Prediction endpoints
- `health_routes.py`: Health monitoring

### 4. Configuration (`config/settings.py`)

Application settings using Pydantic Settings:
- Environment variables
- API configurations
- Database settings
- Security settings

## 🤖 Machine Learning Models

### Available Models

1. **Linear Regression**: Fast, interpretable trend analysis
2. **Random Forest**: Complex pattern recognition
3. **ARIMA**: Time series forecasting
4. **Ensemble**: Combined model for best accuracy

### Model Features

- **Feature Engineering**: 25+ technical indicators
- **Confidence Scoring**: Prediction reliability metrics
- **Risk Assessment**: Investment risk analysis
- **Trend Analysis**: Bullish/bearish/neutral classification

## 🔒 Security Features

- **CORS Configuration**: Controlled cross-origin requests
- **Rate Limiting**: API request throttling
- **Input Validation**: Pydantic model validation
- **Error Handling**: Secure error responses
- **Environment-based Config**: Secure configuration management

## 📊 Monitoring and Health Checks

### Health Check Endpoints

- `/health`: Simple health check
- `/api/v1/health`: Comprehensive health status
- `/api/v1/health/services`: Individual service status
- `/api/v1/health/readiness`: Kubernetes readiness probe
- `/api/v1/health/liveness`: Kubernetes liveness probe

### Health Check Features

- Service status monitoring
- Database connectivity checks
- External API status
- Performance metrics
- Configuration validation

## 🚀 Deployment

### Development
```bash
python main.py
```

### Production with Gunicorn
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (example)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest

# Run with coverage
pytest --cov=.
```

### Test Examples
```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_stock_quote():
    response = client.get("/api/v1/stocks/quote/AAPL")
    assert response.status_code == 200
    assert "data" in response.json()
```

## 📈 Performance Optimization

### Caching
- In-memory prediction caching
- Redis integration for distributed caching
- Configurable cache TTL

### Rate Limiting
- Per-minute request limits
- Burst allowance
- Configurable limits per endpoint

### Async Operations
- Async/await for I/O operations
- Concurrent API calls
- Background task processing

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Alpha Vantage API key not configured
   Solution: Add ALPHA_VANTAGE_API_KEY to .env file
   ```

2. **Import Errors**
   ```
   Error: No module named 'fastapi'
   Solution: Install dependencies: pip install -r requirements.txt
   ```

3. **Port Already in Use**
   ```
   Error: Address already in use
   Solution: Change PORT in .env or kill existing process
   ```

4. **CORS Issues**
   ```
   Error: CORS policy blocks request
   Solution: Update ALLOWED_ORIGINS in .env
   ```

### Debug Mode

Enable debug mode in `.env`:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black .
isort .

# Run linting
flake8 .
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FastAPI** for the excellent web framework
- **Pydantic** for data validation
- **scikit-learn** for machine learning capabilities
- **Alpha Vantage** for stock data API
- **All contributors** and supporters

## 📞 Support

- **Documentation**: Check the `/docs` endpoint
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the maintainers

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Stock predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.
