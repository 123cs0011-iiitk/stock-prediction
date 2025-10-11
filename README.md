# Stock Price Insight Arena

## 📌 Overview

Stock Price Insight Arena is a comprehensive machine learning-powered application designed to analyze real-time stock data and predict future price movements. The system leverages advanced ensemble algorithms including Linear Regression, Random Forest Regressor, and ARIMA time series models to provide accurate predictions through a modern React-based web interface.

## 🎯 Problem Statement

Stock market volatility and complexity make it challenging for investors to predict future price movements, leading to potential financial losses and missed opportunities in trading decisions. This project addresses these challenges by providing data-driven insights and predictions using real market data.

## 💡 Solution

Our system combines multiple machine learning algorithms to analyze:
- Real-time stock data via Alpha Vantage API
- Historical price patterns and technical indicators
- Advanced feature engineering with 25+ technical indicators
- Ensemble prediction methods for improved accuracy
- Comprehensive risk analysis and portfolio management

The predictions are delivered through an intuitive React-based web interface with real-time data visualization and interactive charts.

## 🚀 Features

### Core Functionality
- **Real-time Stock Data**: Live prices and market data from Alpha Vantage API
- **Advanced ML Predictions**: Ensemble models combining Linear Regression and Random Forest
- **Time Series Analysis**: ARIMA models for trend prediction
- **Technical Analysis**: 25+ technical indicators (RSI, SMA, Bollinger Bands, etc.)
- **Interactive Dashboard**: Modern React UI with real-time updates
- **Portfolio Management**: Track and analyze investment portfolios
- **Currency Support**: USD/INR conversion with live rates

### ML Algorithms
- **Linear Regression**: Linear relationship modeling (30% weight)
- **Random Forest Regressor**: Non-linear pattern recognition (70% weight)
- **ARIMA Time Series**: AutoRegressive Integrated Moving Average
- **Ensemble Methods**: Weighted combination for improved accuracy

### Technical Features
- **Feature Engineering**: 25+ technical indicators from price and volume data
- **Performance Metrics**: R² score, RMSE, MAE for each algorithm
- **Confidence Scoring**: Prediction confidence based on model agreement
- **Error Handling**: Comprehensive API rate limiting and fallback systems
- **Caching**: Smart caching for performance optimization

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework with CORS support
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **requests** - HTTP client for API calls
- **python-dotenv** - Environment variable management

### Frontend
- **React 18** - Modern UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Radix UI** - Accessible component primitives
- **Recharts** - Data visualization library
- **Lucide React** - Icon library

### Data & APIs
- **Alpha Vantage API** - Real-time stock data provider
- **RESTful APIs** - Backend communication
- **JSON** - Data exchange format

## 📂 Project Structure

```
stock-prediction/
├── backend/                    # Python Flask backend
│   ├── app.py                 # Main Flask application with API endpoints
│   ├── requirements.txt       # Python dependencies
│   ├── env.example           # Environment variables template
│   ├── models/               # ML model implementations
│   │   └── ml_models.py      # Ensemble & ARIMA models
│   ├── services/             # External service integrations
│   │   └── alpha_vantage.py  # Alpha Vantage API client
│   ├── test_*.py            # Backend test files
│   └── venv/                # Python virtual environment
├── frontend/                   # React TypeScript frontend
│   ├── src/                  # Source code
│   │   ├── App.tsx          # Main React application
│   │   ├── components/      # React components
│   │   ├── services/        # API service layer
│   │   ├── utils/           # Utility functions
│   │   └── styles/          # CSS and styling
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.ts       # Vite configuration
│   └── index.html           # HTML entry point
├── documentation/            # Project documentation
│   ├── Backend-README.md    # Backend documentation
│   ├── Frontend-README.md   # Frontend documentation
│   ├── GitHub-Repository.md # Repository information
│   └── Guidelines.md        # Development guidelines
├── README.md                 # Main project documentation
├── QUICK_START.md           # Quick start guide
├── ALPHA_VANTAGE_SETUP.md   # API setup instructions
├── ML_REFACTOR_SUMMARY.md   # ML implementation details
└── LICENSE                   # MIT License
```

## 🚦 Getting Started

### Prerequisites
- **Python 3.8+** - For backend development
- **Node.js 16+** - For frontend development
- **Git** - Version control
- **Alpha Vantage API Key** - Free account at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/123cs0011-iiitk/stock-prediction.git
   cd stock-prediction
   ```

2. **Setup Backend**
   ```bash
   cd backend
   
   # Create virtual environment
   py -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Setup environment variables
   cp env.example .env
   # Edit .env and add your ALPHA_VANTAGE_API_KEY
   
   # Start backend server
   py app.py
   ```

3. **Setup Frontend** (in a new terminal)
   ```bash
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

4. **Access the application**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:5000
   - **API Health Check**: http://localhost:5000/api/health

### Environment Configuration

Create `backend/.env` file with:
```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
PORT=5000
FLASK_ENV=development
```

## 📊 Usage

### Stock Analysis
1. **Search for Stocks**: Enter a stock symbol (e.g., AAPL, GOOGL, TSLA) in the search bar
2. **View Real-time Data**: See current price, volume, market cap, and company information
3. **Analyze Charts**: Interactive price charts with different time periods (week, month, year)
4. **Get Predictions**: AI-powered price predictions with confidence scores

### ML Predictions
1. **Ensemble Models**: Combined Linear Regression and Random Forest predictions
2. **ARIMA Analysis**: Time series forecasting for trend analysis
3. **Technical Indicators**: 25+ indicators including RSI, SMA, Bollinger Bands
4. **Confidence Scoring**: Prediction reliability based on model agreement

### Portfolio Management
1. **Track Positions**: Add stocks to your portfolio with purchase details
2. **Performance Analysis**: Real-time portfolio value and P&L calculations
3. **Risk Assessment**: Diversification and volatility analysis

### Currency Support
- Toggle between USD and INR with real-time conversion
- All prices and predictions display in selected currency

## 🧪 Testing

### Backend Testing
```bash
cd backend

# Test Alpha Vantage integration
py test_alpha_vantage.py

# Test API endpoints
py test_api.py

# Test ML models
py test_ml_refactor.py
```

### Frontend Testing
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Test live price integration
# Open browser console and run test-live-price.js
```

### Manual Testing
1. Start both backend and frontend servers
2. Open http://localhost:3000
3. Search for popular stocks (AAPL, GOOGL, MSFT, etc.)
4. Verify real-time data and predictions are working

## 🔗 Project Repository

**GitHub Repository:** [https://github.com/123cs0011-iiitk/stock-prediction](https://github.com/123cs0011-iiitk/stock-prediction)

**Project Lead GitHub:** [https://github.com/123cs0011-iiitk](https://github.com/123cs0011-iiitk)

📋 **For detailed repository information, contributing guidelines, and development workflow, see:** [`documentation/GitHub-Repository.md`](documentation/GitHub-Repository.md)

## 👥 Contributors

| Name | Role | GitHub |
|------|------|--------|
| **Ankit Kumar** | Project Lead & ML Engineer | [123cs0011-iiitk](https://github.com/123cs0011-iiitk) |
| **Utkarsh Chaudhary** | Backend Developer | utkarsh |
| **Debapriya Sarkar** | Frontend Developer & Data Analyst | debapriya |

## 📚 API Documentation

### Backend Endpoints

#### Stock Data
- `GET /api/health` - Health check
- `GET /api/stock/price/{symbol}` - Real-time stock price
- `GET /api/stock/{symbol}` - Detailed stock data with historical info
- `GET /api/search?q={query}` - Search stocks by symbol

#### Predictions
- `GET /api/predict/{symbol}` - ML predictions with ensemble models
- `POST /api/models/train/{symbol}` - Train ML models for specific symbol
- `GET /api/models/status` - Get ML model status and performance

#### Portfolio
- `POST /api/portfolio/analyze` - Analyze portfolio performance

### Example API Calls
```bash
# Get live stock price
curl http://localhost:5000/api/stock/price/AAPL

# Get ML prediction
curl http://localhost:5000/api/predict/AAPL

# Health check
curl http://localhost:5000/api/health
```

## 🤝 Contributing

We welcome contributions from the community! Please follow these steps:

1. **Fork the repository** from [https://github.com/123cs0011-iiitk/stock-prediction](https://github.com/123cs0011-iiitk/stock-prediction)
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/stock-prediction.git
   cd stock-prediction
   ```
3. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
4. **Make your changes** and commit them (`git commit -m 'Add some AmazingFeature'`)
5. **Push to your fork** (`git push origin feature/AmazingFeature`)
6. **Open a Pull Request** to the main repository

**Repository Information:**
- **Main Repository:** [https://github.com/123cs0011-iiitk/stock-prediction](https://github.com/123cs0011-iiitk/stock-prediction)
- **Project Lead:** [Ankit Kumar (@123cs0011-iiitk)](https://github.com/123cs0011-iiitk)

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Write comprehensive tests for new features
- Update documentation for API changes
- Follow conventional commit messages

## 🔧 Troubleshooting

### Common Issues

**Backend Issues:**
- **API Key Error**: Ensure `ALPHA_VANTAGE_API_KEY` is set in `backend/.env`
- **Port Already in Use**: Change port in `.env` or kill existing process
- **Import Errors**: Activate virtual environment and reinstall dependencies

**Frontend Issues:**
- **Build Errors**: Run `npm install` to update dependencies
- **API Connection**: Ensure backend is running on port 5000
- **CORS Issues**: Backend has CORS enabled for localhost:3000

**API Issues:**
- **Rate Limit Exceeded**: Alpha Vantage free tier has 5 calls/minute limit
- **Invalid Symbol**: Use valid stock symbols (AAPL, GOOGL, etc.)
- **No Data**: Some symbols may not be available in Alpha Vantage

### Getting Help
1. Check the [Alpha Vantage Setup Guide](ALPHA_VANTAGE_SETUP.md)
2. Review the [ML Refactor Summary](ML_REFACTOR_SUMMARY.md)
3. Check existing GitHub issues
4. Create a new issue with detailed error information

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Alpha Vantage** for providing comprehensive stock data API
- **scikit-learn** community for excellent machine learning tools
- **React** and **TypeScript** communities for modern frontend frameworks
- **Tailwind CSS** and **Radix UI** for beautiful UI components
- All contributors and supporters of this project

## 📞 Support

If you encounter any issues or have questions:
- **Create an issue** on GitHub Issues with detailed error information
- **Check documentation** in the `/documentation` folder
- **Review troubleshooting** section above
- **Contact the team** via GitHub

## ⭐ Show Your Support

If you find this project helpful, please give it a star! ⭐

## 🚀 Future Enhancements

### Planned Features
- **WebSocket Integration**: Real-time price updates
- **Advanced ML Models**: LSTM, GRU for time series prediction
- **Sentiment Analysis**: News and social media sentiment integration
- **Mobile App**: React Native mobile application
- **User Authentication**: User accounts and portfolio persistence
- **Advanced Analytics**: More technical indicators and risk metrics

### Performance Improvements
- **Database Integration**: PostgreSQL for data persistence
- **Caching Layer**: Redis for improved performance
- **Load Balancing**: Multiple backend instances
- **CDN Integration**: Faster asset delivery

---

**⚠️ Disclaimer:** This tool is for educational and research purposes only. Stock predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with financial advisors before making investment decisions. The authors are not responsible for any financial losses incurred through the use of this application.
