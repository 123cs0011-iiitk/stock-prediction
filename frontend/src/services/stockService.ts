// Stock data interfaces
export interface StockData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  currency: string;
  pe?: number;
  dividend?: number;
  sector?: string;
  industry?: string;
  open?: number;
  high?: number;
  low?: number;
  previousClose?: number;
  timestamp?: string;
  source?: string;
}

export interface PricePoint {
  date: string;
  price: number;
  volume: number;
}

export interface PredictionResult {
  symbol: string;
  current_price: number;
  prediction: {
    trend: string;
    confidence: number;
    support_level: number;
    resistance_level: number;
    next_day_range: {
      low: number;
      high: number;
    };
  };
  technical_indicators: {
    sma_20: number;
    sma_50: number;
    trend_strength: number;
  };
  ml_predictions?: any;
  arima_predictions?: any;
  model_status: {
    ml_models_trained: boolean;
    arima_trained: boolean;
  };
}

// Base URL for API calls - using local backend
const BASE_URL = 'http://localhost:5000/api';

// API call helper with error handling
async function apiCall<T>(endpoint: string): Promise<T> {
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API call failed for ${endpoint}:`, error);
    throw error instanceof Error ? error : new Error('Unknown API error');
  }
}

export const stockService = {
  // Get live stock price data using Alpha Vantage API
  getLiveStockPrice: async (symbol: string): Promise<StockData> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    try {
      const response = await apiCall<any>(`/stock/price/${symbol.toUpperCase()}`);
      // Transform backend response to match frontend interface
      return {
        symbol: response.symbol,
        name: response.name,
        price: response.price,
        change: response.change,
        changePercent: response.changePercent,
        volume: response.volume,
        marketCap: response.marketCap,
        currency: response.currency,
        pe: response.pe,
        dividend: response.dividend,
        sector: response.sector,
        industry: response.industry,
        open: response.open,
        high: response.high,
        low: response.low,
        previousClose: response.previousClose,
        timestamp: response.timestamp,
        source: response.source
      };
    } catch (error) {
      console.error(`Failed to fetch live stock price for ${symbol}:`, error);
      throw new Error(`Unable to fetch live price for ${symbol}. Please try again later.`);
    }
  },

  // Get current stock data (mock data - kept for compatibility)
  getStockData: async (symbol: string): Promise<StockData> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    try {
      const response = await apiCall<any>(`/stock/${symbol.toUpperCase()}`);
      // Transform backend response to match frontend interface
      return {
        symbol: response.symbol,
        name: response.info.name,
        price: response.info.price,
        change: response.info.change,
        changePercent: response.info.changePercent,
        volume: response.info.volume,
        marketCap: response.info.marketCap,
        currency: response.info.currency,
        pe: response.info.pe,
        dividend: response.info.dividend,
        sector: response.info.sector,
        industry: response.info.industry
      };
    } catch (error) {
      console.error(`Failed to fetch stock data for ${symbol}:`, error);
      throw new Error(`Unable to fetch data for ${symbol}. Please try again later.`);
    }
  },

  // Get historical data
  getHistoricalData: async (symbol: string, period: 'week' | 'month' | 'year'): Promise<PricePoint[]> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    if (!['week', 'month', 'year'].includes(period)) {
      throw new Error('Invalid period. Must be week, month, or year.');
    }
    
    try {
      const response = await apiCall<any>(`/stock/${symbol.toUpperCase()}`);
      // Transform backend historical data to match frontend interface
      if (response.historical && response.historical.dates && response.historical.prices) {
        return response.historical.dates.map((date: string, index: number) => ({
          date,
          price: response.historical.prices[index],
          volume: response.historical.volumes ? response.historical.volumes[index] : 0
        }));
      }
      return [];
    } catch (error) {
      console.error(`Failed to fetch historical data for ${symbol} (${period}):`, error);
      throw new Error(`Unable to fetch historical data for ${symbol}. Please try again later.`);
    }
  },

  // Get stock prediction
  getPrediction: async (symbol: string): Promise<PredictionResult> => {
    if (!symbol) {
      throw new Error('Stock symbol is required');
    }
    
    try {
      return await apiCall<PredictionResult>(`/predict/${symbol.toUpperCase()}`);
    } catch (error) {
      console.error(`Failed to get prediction for ${symbol}:`, error);
      throw new Error(`Unable to generate prediction for ${symbol}. Please try again later.`);
    }
  },

  // Search stocks
  searchStocks: async (query: string): Promise<{ symbol: string; name: string }[]> => {
    try {
      const encodedQuery = encodeURIComponent(query.trim());
      const response = await apiCall<any>(`/search?q=${encodedQuery}`);
      return [{
        symbol: response.symbol,
        name: response.name
      }];
    } catch (error) {
      console.error(`Failed to search stocks with query "${query}":`, error);
      // Return empty array instead of throwing to avoid breaking the UI
      return [];
    }
  },

  // Search stocks with live price data
  searchStocksWithLivePrice: async (query: string): Promise<StockData[]> => {
    try {
      // First try to get live price directly for the symbol
      const symbol = query.trim().toUpperCase();
      try {
        const liveData = await stockService.getLiveStockPrice(symbol);
        return [liveData];
      } catch (liveError) {
        // If live price fails, fall back to mock search
        console.log(`Live price not available for ${symbol}, falling back to mock data`);
        const mockResults = await stockService.searchStocks(query);
        return mockResults.map(stock => ({
          symbol: stock.symbol,
          name: stock.name,
          price: 0, // Will be filled by mock data
          change: 0,
          changePercent: 0,
          volume: 0,
          marketCap: 0,
          currency: 'USD'
        }));
      }
    } catch (error) {
      console.error(`Failed to search stocks with live price for "${query}":`, error);
      return [];
    }
  },

  // Get popular stocks (static list for UI display)
  getPopularStocks: async (): Promise<{ symbol: string; name: string }[]> => {
    // Popular stocks for quick access - all use live Alpha Vantage data
    return [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'NFLX', name: 'Netflix Inc.' }
    ];
  },

  // Health check for server status
  checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
    try {
      return await apiCall<{ status: string; timestamp: string }>('/health');
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error('Server is unavailable');
    }
  }
};