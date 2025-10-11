# 🚀 Quick Start Guide - Stock Price Insight Arena

## ✅ Project Structure (Cleaned Up)

```
stock-prediction/
├── .env                    # Environment variables (root level)
├── backend/
│   ├── venv/              # Python virtual environment (ONLY ONE)
│   ├── app.py             # Flask API server
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── src/               # React application
│   └── package.json       # Node.js dependencies
```

## 🎯 How to Run the Application

### Manual Start (Recommended)

**Backend:**
```bash
cd backend
venv\Scripts\activate
set ALPHA_VANTAGE_API_KEY=VI0EJZBQAD1JHE8E
py app.py
```

**Frontend (in a new terminal):**
```bash
cd frontend
npm run dev
```

## ✅ What's Fixed

1. **Single Virtual Environment**: Removed duplicate `.venv`, kept only `backend/venv/`
2. **Environment Variables**: Moved `.env` to root directory for easy access
3. **Clean Structure**: Organized project files properly

## 🔧 Environment Variables

The `.env` file contains:
- `ALPHA_VANTAGE_API_KEY=VI0EJZBQAD1JHE8E` (Valid API key)
- `PORT=5000` (Backend port)
- `FLASK_ENV=development` (Development mode)

## 🌐 Access Points

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/health
- **Stock Price API**: http://localhost:5000/api/stock/price/AAPL
- **Prediction API**: http://localhost:5000/api/predict/AAPL

## ✨ Features Working

- ✅ Real-time stock data from Alpha Vantage API
- ✅ ML predictions with ARIMA and Ensemble models
- ✅ Interactive React frontend with charts
- ✅ Full-stack integration
- ✅ Error handling and rate limiting
- ✅ Currency conversion (USD/INR)

---

**Ready to use!** Follow the manual start instructions above and open http://localhost:3000 in your browser.
