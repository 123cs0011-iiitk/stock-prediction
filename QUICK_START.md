# 🚀 Quick Start Guide - Stock Price Insight Arena

## ✅ Project Structure

```
stock-prediction/
├── backend/                    # Python Flask backend
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   ├── env.example           # Environment variables template
│   ├── models/               # ML model implementations
│   ├── services/             # Alpha Vantage API integration
│   ├── test_*.py            # Backend test files
│   └── venv/                # Python virtual environment
├── frontend/                   # React TypeScript frontend
│   ├── src/                  # Source code
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.ts       # Vite configuration
│   └── index.html           # HTML entry point
├── documentation/            # Project documentation
├── README.md                 # Main project documentation
├── ALPHA_VANTAGE_SETUP.md   # API setup guide
└── LICENSE                   # MIT License
```

## 🎯 How to Run the Application

### Prerequisites
- **Python 3.8+** - For backend
- **Node.js 16+** - For frontend
- **Alpha Vantage API Key** - Get free key at [alphavantage.co](https://www.alphavantage.co/support/#api-key)

### Step 1: Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
py -m venv venv

# Activate virtual environment
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

### Step 2: Setup Frontend (New Terminal)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## ✅ Current Features

1. **Real-time Stock Data**: Alpha Vantage API integration
2. **Advanced ML Models**: Ensemble (Linear Regression + Random Forest) and ARIMA
3. **Modern Frontend**: React TypeScript with Tailwind CSS
4. **Interactive Charts**: Price visualization with multiple time periods
5. **Portfolio Management**: Track and analyze investments
6. **Currency Support**: USD/INR conversion
7. **Comprehensive Testing**: Backend and frontend test suites

## 🔧 Environment Variables

Create `backend/.env` file with:
```env
ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
PORT=5000
FLASK_ENV=development
```

**Important**: Replace `your_actual_api_key_here` with your real Alpha Vantage API key.

## 🌐 Access Points

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/health
- **Stock Price API**: http://localhost:5000/api/stock/price/AAPL
- **Prediction API**: http://localhost:5000/api/predict/AAPL

## ✨ Features Working

- ✅ **Real-time Stock Data**: Alpha Vantage API integration
- ✅ **ML Predictions**: Ensemble models (Linear Regression + Random Forest) and ARIMA
- ✅ **Interactive Frontend**: React TypeScript with modern UI components
- ✅ **Price Charts**: Interactive charts with multiple time periods
- ✅ **Technical Analysis**: 25+ technical indicators
- ✅ **Portfolio Management**: Track and analyze investments
- ✅ **Currency Support**: USD/INR conversion with live rates
- ✅ **Error Handling**: Comprehensive error handling and rate limiting
- ✅ **Responsive Design**: Mobile-first approach

## 🧪 Testing

### Test Backend
```bash
cd backend
py test_alpha_vantage.py  # Test API integration
py test_api.py           # Test API endpoints
py test_ml_refactor.py   # Test ML models
```

### Test Frontend
1. Open http://localhost:3000
2. Search for popular stocks (AAPL, GOOGL, MSFT, TSLA)
3. Verify real-time data and predictions
4. Test currency conversion
5. Check responsive design

## 🚨 Troubleshooting

### Common Issues
- **Backend won't start**: Check if port 5000 is available
- **API errors**: Verify Alpha Vantage API key in `.env` file
- **Frontend won't connect**: Ensure backend is running on port 5000
- **Rate limit errors**: Alpha Vantage free tier has 5 calls/minute limit

### Getting Help
- Check [Alpha Vantage Setup Guide](ALPHA_VANTAGE_SETUP.md)
- Review [Backend Documentation](documentation/Backend-README.md)
- Check [Frontend Documentation](documentation/Frontend-README.md)
- Create GitHub issue for bugs

---

**Ready to use!** Follow the setup instructions above and open http://localhost:3000 in your browser.
