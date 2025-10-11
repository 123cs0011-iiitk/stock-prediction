# Backend Documentation - Stock Price Insight Arena

A comprehensive Python Flask backend providing real-time stock data analysis, machine learning predictions, and portfolio management through RESTful APIs.

## 🚀 Features

### Core Functionality
- ✅ **Real-time Stock Data**: Alpha Vantage API integration for live market data
- ✅ **Advanced ML Predictions**: Ensemble models with Linear Regression and Random Forest
- ✅ **ARIMA Time Series**: AutoRegressive Integrated Moving Average forecasting
- ✅ **Technical Analysis**: 25+ technical indicators (RSI, SMA, Bollinger Bands, etc.)
- ✅ **Portfolio Management**: Real-time portfolio analysis and risk assessment
- ✅ **RESTful APIs**: Comprehensive API endpoints for frontend integration
- ✅ **Error Handling**: Robust error handling with API rate limiting
- ✅ **Caching**: Smart caching system for improved performance

### ML Algorithms
- ✅ **Ensemble Stock Predictor**: Combined Linear Regression (30%) and Random Forest (70%)
- ✅ **ARIMA Time Series**: Multi-step forecasting with confidence scoring
- ✅ **Feature Engineering**: Advanced technical indicator calculations
- ✅ **Performance Metrics**: R² score, RMSE, MAE for model evaluation

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.8+**: Core programming language
- **Flask**: Web framework with CORS support
- **scikit-learn**: Machine learning algorithms and tools
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **requests**: HTTP client for API calls

### ML Libraries
- **scikit-learn**: Linear Regression, Random Forest, StandardScaler
- **pandas**: Data preprocessing and feature engineering
- **numpy**: Mathematical operations and array handling

### External Services
- **Alpha Vantage API**: Real-time stock data provider
- **python-dotenv**: Environment variable management

## 📁 Backend Project Structure

```
backend/
├── app.py                    # Main Flask application with API endpoints
├── requirements.txt          # Python dependencies
├── env.example              # Environment variables template
├── models/                  # ML model implementations
│   ├── __init__.py         # Package initialization
│   └── ml_models.py        # Ensemble & ARIMA models
├── services/               # External service integrations
│   ├── __init__.py         # Package initialization
│   └── alpha_vantage.py    # Alpha Vantage API client
├── test_alpha_vantage.py   # Alpha Vantage integration tests
├── test_api.py            # API endpoint tests
├── test_ml_refactor.py    # ML model tests
└── venv/                  # Python virtual environment
```

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** - Core runtime environment
- **pip** - Python package manager
- **Git** - Version control (optional)
- **Alpha Vantage API Key** - Free account at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### Installation & Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   py -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**:
   ```bash
   cp env.example .env
   # Edit .env and add your ALPHA_VANTAGE_API_KEY
   ```

5. **Start the backend server**:
   ```bash
   py app.py
   ```

6. **Access the API**:
   - Backend API: `http://localhost:5000`
   - Health Check: `http://localhost:5000/api/health`

## 🎯 API Usage

### Stock Data Endpoints

#### Get Real-time Stock Price
```bash
GET /api/stock/price/{symbol}
curl http://localhost:5000/api/stock/price/AAPL
```

#### Get Detailed Stock Data
```bash
GET /api/stock/{symbol}
curl http://localhost:5000/api/stock/AAPL
```

#### Search Stocks
```bash
GET /api/search?q={query}
curl http://localhost:5000/api/search?q=AAPL
```

### ML Prediction Endpoints

#### Get Stock Prediction
```bash
GET /api/predict/{symbol}
curl http://localhost:5000/api/predict/AAPL
```

#### Train ML Models
```bash
POST /api/models/train/{symbol}
curl -X POST http://localhost:5000/api/models/train/AAPL
```

#### Get Model Status
```bash
GET /api/models/status
curl http://localhost:5000/api/models/status
```

### Portfolio Management

#### Analyze Portfolio
```bash
POST /api/portfolio/analyze
curl -X POST http://localhost:5000/api/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{"portfolio": [{"symbol": "AAPL", "shares": 10, "costBasis": 15000}]}'
```

## 🧪 Testing

### Running Tests

#### Test Alpha Vantage Integration
```bash
py test_alpha_vantage.py
```

#### Test API Endpoints
```bash
py test_api.py
```

#### Test ML Models
```bash
py test_ml_refactor.py
```

### Test Coverage
- **API Integration**: Alpha Vantage connectivity and error handling
- **ML Models**: Ensemble and ARIMA model functionality
- **Endpoints**: All REST API endpoints with various scenarios
- **Error Handling**: Rate limiting, invalid symbols, network issues

## 🔧 Configuration

### Environment Variables
Create `backend/.env` file with:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
PORT=5000
FLASK_ENV=development
```

### ML Model Configuration
- **Ensemble Weights**: Linear Regression (30%), Random Forest (70%)
- **ARIMA Order**: (1, 1, 1) - configurable in ml_models.py
- **Feature Engineering**: 25+ technical indicators
- **Training Threshold**: Minimum 100 data points for robust training

## 📊 Performance

### Caching Strategy
- **Stock Data**: 5-minute cache to avoid API rate limits
- **ML Models**: Trained models cached per symbol
- **API Responses**: Smart caching for frequently accessed data

### Rate Limiting
- **Alpha Vantage**: 5 calls/minute (free tier)
- **Backend**: 12-second delays between API calls
- **Error Handling**: Graceful fallbacks when limits exceeded

## 🚧 Current Limitations

- **API Rate Limits**: Alpha Vantage free tier has daily/minute limits
- **Data Availability**: Some symbols may not be available
- **Model Training**: Requires sufficient historical data (100+ points)
- **Real-time Updates**: No WebSocket integration yet

## 🔮 Future Enhancements

### Planned Features
1. **Database Integration**: PostgreSQL for data persistence
2. **Advanced ML Models**: LSTM, GRU for time series
3. **WebSocket Support**: Real-time price updates
4. **User Authentication**: JWT-based user management
5. **Caching Layer**: Redis for improved performance

### Performance Improvements
1. **Load Balancing**: Multiple backend instances
2. **Background Tasks**: Celery for ML model training
3. **API Optimization**: GraphQL for efficient data fetching
4. **Monitoring**: Application performance monitoring

## 🤝 Contributing

We welcome contributions to improve the backend! Please follow these guidelines:

### Development Guidelines
- Follow **PEP 8** for Python code style
- Write comprehensive **docstrings** for all functions
- Add **tests** for new features
- Update **documentation** for API changes
- Use **conventional commit messages**

### Code Structure
- **Modular Design**: Separate concerns (models, services, API)
- **Error Handling**: Comprehensive exception handling
- **Type Hints**: Use Python type annotations
- **Documentation**: Clear docstrings and comments

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For backend-related issues:
- Check the [API Documentation](#-api-usage) above
- Review the [ML Refactor Summary](../../ML_REFACTOR_SUMMARY.md)
- Check existing GitHub issues
- Create a new issue with detailed error information

## 🔗 Related Documentation

- [Frontend Documentation](../Frontend-README.md)
- [Alpha Vantage Setup](../../ALPHA_VANTAGE_SETUP.md)
- [ML Implementation Details](../../ML_REFACTOR_SUMMARY.md)
- [Quick Start Guide](../../QUICK_START.md)

---

**Happy Coding! 🚀📈**
