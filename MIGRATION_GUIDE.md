# Migration Guide: Flask to Express.js + FastAPI Hybrid

## 🎯 Architecture Overview

You now have a **hybrid backend architecture**:

- **Express.js** (Port 5000): Main business logic, data fetching, API endpoints
- **FastAPI** (Port 8000): Dedicated ML service with all machine learning algorithms
- **Frontend** (Port 3000): Next.js/React app (unchanged)

## 🚀 How to Start the Services

### 1. Start FastAPI ML Service (Port 8000)

```bash
cd backend-fastapi

# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Express.js Backend (Port 5000)

```bash
cd backend-node

# Install dependencies
npm install

# Start Express.js service
npm run dev
```

### 3. Start Frontend (Port 3000)

```bash
cd frontend-nextjs

# Install dependencies (if needed)
npm install

# Start Next.js app
npm run dev
```

## 📁 What Changed

### New Express.js Backend Structure
```
backend-node/
├── routes/
│   ├── stockRoutes.js          # Stock data endpoints
│   ├── predictionRoutes.js     # ML prediction endpoints
│   ├── modelRoutes.js          # ML algorithm management
│   ├── healthRoutes.js         # Health checks
│   ├── portfolioRoutes.js      # Portfolio analysis
│   └── newsRoutes.js           # Stock news
├── services/
│   ├── mlService.js            # Communication with FastAPI ML service
│   └── dataService.js          # Stock data fetching (Yahoo, Alpha Vantage)
├── middleware/
│   └── errorHandler.js         # Error handling
├── utils/
│   └── logger.js               # Logging utility
└── config/
    └── database.js             # PostgreSQL configuration
```

### Key Features

1. **Express.js handles:**
   - Stock data fetching from multiple APIs
   - Portfolio analysis
   - News aggregation
   - API routing and middleware
   - Database operations

2. **FastAPI ML Service handles:**
   - All machine learning algorithms
   - Stock predictions
   - Model training
   - Technical analysis

## 🔧 Configuration

### Environment Variables

Create `.env` files in both backends:

**backend-node/.env:**
```env
NODE_ENV=development
PORT=5000
ML_SERVICE_URL=http://localhost:8000
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_prediction
DB_USER=postgres
DB_PASSWORD=your_password
```

**backend-fastapi/.env:**
```env
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

## 🌐 API Endpoints

### Express.js Backend (Port 5000)
- `GET /api/health` - Health check
- `GET /api/stock/price/:symbol` - Stock price
- `GET /api/stock/:symbol` - Comprehensive stock data
- `GET /api/stock/search?q=query` - Search stocks
- `GET /api/predict/:symbol` - Get ML prediction
- `POST /api/portfolio/analyze` - Portfolio analysis
- `GET /api/news/:symbol` - Stock news

### FastAPI ML Service (Port 8000)
- `GET /api/v1/algorithms/available` - Available ML algorithms
- `POST /api/v1/predictions/predict` - ML predictions
- `POST /api/v1/algorithms/train` - Train ML models
- `GET /docs` - Interactive API documentation

## 🔄 Migration Benefits

1. **Better Separation of Concerns:**
   - Express.js: Business logic, data management
   - FastAPI: ML algorithms, predictions

2. **Language Optimization:**
   - JavaScript: Fast API development, async operations
   - Python: Superior ML libraries, scientific computing

3. **Scalability:**
   - Can scale ML service independently
   - Multiple ML service instances possible

4. **Development Experience:**
   - FastAPI auto-generated docs
   - Express.js familiar ecosystem
   - TypeScript support in frontend

## 🧪 Testing the Migration

1. **Test Express.js Backend:**
   ```bash
   curl http://localhost:5000/api/health
   ```

2. **Test FastAPI ML Service:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Test Stock Data:**
   ```bash
   curl http://localhost:5000/api/stock/price/AAPL
   ```

4. **Test ML Prediction:**
   ```bash
   curl http://localhost:5000/api/predict/AAPL
   ```

## 🎉 You're All Set!

Your Flask backend has been successfully migrated to a hybrid Express.js + FastAPI architecture. The frontend will continue to work exactly as before, but now with better performance and separation of concerns.

**Next Steps:**
1. Start both services using the commands above
2. Test the endpoints
3. Update any API keys in the environment files
4. Enjoy your new hybrid architecture!
