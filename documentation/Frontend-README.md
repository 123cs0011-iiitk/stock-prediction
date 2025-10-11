# Frontend Documentation - Stock Price Insight Arena

A modern React TypeScript frontend providing an intuitive interface for stock analysis, real-time data visualization, and ML-powered predictions.

## 🚀 Features

### Core Functionality
- ✅ **Real-time Stock Data**: Live prices and market data integration
- ✅ **Interactive Charts**: Price charts with multiple time periods
- ✅ **ML Predictions**: AI-powered price predictions with confidence scores
- ✅ **Portfolio Management**: Track and analyze investment portfolios
- ✅ **Currency Support**: USD/INR conversion with live rates
- ✅ **Responsive Design**: Mobile-first approach with modern UI
- ✅ **Error Handling**: Comprehensive error states and loading indicators

### UI Components
- ✅ **Stock Search**: Search and select stocks by symbol
- ✅ **Stock Information**: Company details and market data
- ✅ **Price Charts**: Interactive charts with Recharts
- ✅ **Prediction Display**: ML predictions with technical indicators
- ✅ **Currency Toggle**: Real-time currency conversion
- ✅ **Portfolio Tracker**: Investment portfolio management

## 🛠️ Technology Stack

### Core Technologies
- **React 18**: Modern UI framework with hooks
- **TypeScript**: Type-safe JavaScript development
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework

### UI Libraries
- **Radix UI**: Accessible component primitives
- **Recharts**: Data visualization and charting
- **Lucide React**: Icon library
- **class-variance-authority**: Component styling utilities

### Development Tools
- **Vite**: Build tool and dev server
- **TypeScript**: Type checking and development
- **ESLint**: Code linting (if configured)
- **Prettier**: Code formatting (if configured)

## 📁 Frontend Project Structure

```
frontend/
├── src/                    # Source code
│   ├── App.tsx            # Main React application
│   ├── main.tsx           # Application entry point
│   ├── index.css          # Global styles
│   ├── components/        # React components
│   │   ├── StockSearch.tsx       # Stock search component
│   │   ├── StockInfo.tsx         # Stock information display
│   │   ├── StockChart.tsx        # Price charts
│   │   ├── StockPrediction.tsx   # ML predictions display
│   │   ├── CurrencyToggle.tsx    # Currency conversion
│   │   └── ui/                   # Reusable UI components
│   ├── services/          # API service layer
│   │   └── stockService.ts       # Backend API communication
│   ├── utils/             # Utility functions
│   │   └── currency.ts           # Currency conversion utilities
│   └── styles/            # CSS and styling
│       └── globals.css           # Global styles
├── public/                # Static assets
├── package.json          # Node.js dependencies
├── vite.config.ts        # Vite configuration
├── tsconfig.json         # TypeScript configuration
└── index.html            # HTML entry point
```

## 🚀 Getting Started

### Prerequisites
- **Node.js 16+** - JavaScript runtime
- **npm** - Package manager (comes with Node.js)
- **Backend Running** - Ensure backend is running on port 5000

### Installation & Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Access the application**:
   - Frontend: `http://localhost:3000`
   - Ensure backend is running on `http://localhost:5000`

### Available Scripts

```bash
# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Start development server (alias)
npm run start
```

## 🎯 Usage Guide

### Stock Analysis Workflow

1. **Search for Stocks**:
   - Enter stock symbol in search bar (e.g., AAPL, GOOGL, TSLA)
   - Select from popular stocks or search results
   - Real-time data will be fetched automatically

2. **View Stock Information**:
   - Current price, volume, and market cap
   - Company details and sector information
   - Price change and percentage change
   - 52-week high/low data

3. **Analyze Price Charts**:
   - Interactive price charts with multiple time periods
   - Toggle between week, month, and year views
   - Volume data visualization
   - Responsive chart design

4. **Review ML Predictions**:
   - AI-powered price predictions with confidence scores
   - Technical indicators (RSI, SMA, Bollinger Bands)
   - Trend analysis and support/resistance levels
   - Ensemble model predictions

5. **Currency Conversion**:
   - Toggle between USD and INR
   - Real-time conversion rates
   - All prices and predictions update accordingly

### Component Architecture

#### StockSearch Component
- Handles stock symbol input and validation
- Integrates with backend search API
- Provides popular stocks for quick access

#### StockInfo Component
- Displays comprehensive stock information
- Shows real-time price data and market metrics
- Handles loading states and error conditions

#### StockChart Component
- Interactive price charts using Recharts
- Multiple time period support
- Responsive design for mobile and desktop

#### StockPrediction Component
- ML prediction display with confidence scores
- Technical indicator visualization
- Trend analysis and risk assessment

#### CurrencyToggle Component
- Currency conversion between USD and INR
- Real-time rate updates
- Consistent formatting across the app

## 🔧 Configuration

### API Configuration
The frontend communicates with the backend through the `stockService.ts`:

```typescript
// Base URL for API calls
const BASE_URL = 'http://localhost:5000/api';

// Available endpoints
- /api/stock/price/{symbol}    // Real-time price
- /api/stock/{symbol}          // Detailed data
- /api/predict/{symbol}        // ML predictions
- /api/search?q={query}        // Stock search
```

### Environment Variables
For production deployment, configure:
```env
VITE_API_BASE_URL=https://your-backend-url.com/api
```

## 🧪 Testing

### Manual Testing
1. Start both backend and frontend servers
2. Open http://localhost:3000
3. Test stock search functionality
4. Verify real-time data updates
5. Test currency conversion
6. Check responsive design on mobile

### Browser Testing
- **Chrome**: Primary development browser
- **Firefox**: Cross-browser compatibility
- **Safari**: macOS compatibility
- **Edge**: Windows compatibility
- **Mobile**: iOS Safari, Chrome Mobile

## 📱 Responsive Design

### Breakpoints
- **Desktop**: 1024px and above
- **Tablet**: 768px - 1023px
- **Mobile**: Below 768px

### Mobile Features
- Touch-friendly interface
- Responsive charts and tables
- Optimized loading states
- Mobile navigation patterns

## 🚧 Current Limitations

- **Real-time Updates**: No WebSocket integration yet
- **Offline Support**: Requires internet connection
- **Data Persistence**: No local storage for user preferences
- **Advanced Charts**: Limited chart customization options

## 🔮 Future Enhancements

### Planned Features
1. **Real-time Updates**: WebSocket integration for live prices
2. **User Authentication**: Login and user accounts
3. **Portfolio Persistence**: Save portfolios to user accounts
4. **Advanced Charts**: More chart types and indicators
5. **Mobile App**: React Native mobile application

### Performance Improvements
1. **Code Splitting**: Lazy loading for better performance
2. **Caching**: Service worker for offline functionality
3. **Bundle Optimization**: Smaller bundle sizes
4. **Image Optimization**: Optimized asset loading

## 🤝 Contributing

### Development Guidelines
- Use **TypeScript** for type safety
- Follow **React best practices** and hooks patterns
- Use **Tailwind CSS** for styling
- Write **accessible components** with Radix UI
- Add **error boundaries** for better error handling

### Code Structure
- **Component-based architecture**: Reusable UI components
- **Service layer**: Centralized API communication
- **Utility functions**: Shared helper functions
- **Type definitions**: Comprehensive TypeScript interfaces

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For frontend-related issues:
- Check the [Component Architecture](#component-architecture) section
- Review the [Usage Guide](#-usage-guide) above
- Check existing GitHub issues
- Create a new issue with detailed error information

## 🔗 Related Documentation

- [Backend Documentation](./Backend-README.md)
- [Main README](../README.md)
- [Quick Start Guide](../QUICK_START.md)
- [Alpha Vantage Setup](../ALPHA_VANTAGE_SETUP.md)

---

**Happy Coding! 🚀📈**
