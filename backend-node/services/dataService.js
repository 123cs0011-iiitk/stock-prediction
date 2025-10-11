/**
 * Data Service - Stock data fetching and management
 * Handles stock data from multiple sources with caching
 */

const axios = require('axios');
const logger = require('../utils/logger');

class DataService {
  constructor() {
    // Data source configurations
    this.sources = {
      yahoo: {
        baseUrl: 'https://query1.finance.yahoo.com/v8/finance/chart',
        timeout: 10000
      },
      alphaVantage: {
        baseUrl: 'https://www.alphavantage.co/query',
        apiKey: process.env.ALPHA_VANTAGE_API_KEY,
        timeout: 15000,
        rateLimit: 5 // calls per minute for free tier
      },
      finnhub: {
        baseUrl: 'https://finnhub.io/api/v1',
        apiKey: process.env.FINNHUB_API_KEY,
        timeout: 10000,
        rateLimit: 60 // calls per minute for free tier
      }
    };

    // Cache for stock data
    this.cache = new Map();
    this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
  }

  /**
   * Get cached data if still valid
   */
  getCachedData(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }
    this.cache.delete(key);
    return null;
  }

  /**
   * Cache data with timestamp
   */
  setCachedData(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  /**
   * Get stock quote from Yahoo Finance
   */
  async getYahooQuote(symbol) {
    try {
      const cached = this.getCachedData(`yahoo_quote_${symbol}`);
      if (cached) return cached;

      const response = await axios.get(`${this.sources.yahoo.baseUrl}/${symbol}`, {
        timeout: this.sources.yahoo.timeout,
        params: {
          interval: '1d',
          range: '1d'
        }
      });

      const data = response.data;
      if (!data.chart || !data.chart.result || !data.chart.result[0]) {
        throw new Error('No data available');
      }

      const result = data.chart.result[0];
      const meta = result.meta;
      const quote = result.indicators.quote[0];

      const quoteData = {
        symbol: symbol.toUpperCase(),
        price: meta.regularMarketPrice || quote.close[quote.close.length - 1],
        change: meta.regularMarketChange || 0,
        changePercent: meta.regularMarketChangePercent * 100 || 0,
        volume: meta.regularMarketVolume || quote.volume[quote.volume.length - 1],
        high: meta.regularMarketDayHigh || Math.max(...quote.high),
        low: meta.regularMarketDayLow || Math.min(...quote.low),
        open: meta.regularMarketOpen || quote.open[quote.open.length - 1],
        previousClose: meta.previousClose || quote.close[quote.close.length - 2],
        marketCap: meta.marketCap || null,
        currency: meta.currency || 'USD',
        timestamp: new Date().toISOString(),
        source: 'yahoo'
      };

      this.setCachedData(`yahoo_quote_${symbol}`, quoteData);
      return quoteData;

    } catch (error) {
      logger.error(`Yahoo Finance error for ${symbol}:`, error.message);
      throw error;
    }
  }

  /**
   * Get stock quote from Alpha Vantage
   */
  async getAlphaVantageQuote(symbol) {
    try {
      if (!this.sources.alphaVantage.apiKey) {
        throw new Error('Alpha Vantage API key not configured');
      }

      const cached = this.getCachedData(`alpha_quote_${symbol}`);
      if (cached) return cached;

      const response = await axios.get(this.sources.alphaVantage.baseUrl, {
        timeout: this.sources.alphaVantage.timeout,
        params: {
          function: 'GLOBAL_QUOTE',
          symbol: symbol.toUpperCase(),
          apikey: this.sources.alphaVantage.apiKey
        }
      });

      const data = response.data;
      if (data['Error Message']) {
        throw new Error(data['Error Message']);
      }

      if (data['Note']) {
        throw new Error(data['Note']);
      }

      if (!data['Global Quote']) {
        throw new Error('No quote data available');
      }

      const quote = data['Global Quote'];
      const quoteData = {
        symbol: quote['01. symbol'],
        price: parseFloat(quote['05. price']),
        change: parseFloat(quote['09. change']),
        changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
        volume: parseInt(quote['06. volume']),
        high: parseFloat(quote['03. high']),
        low: parseFloat(quote['04. low']),
        open: parseFloat(quote['02. open']),
        previousClose: parseFloat(quote['08. previous close']),
        timestamp: new Date().toISOString(),
        source: 'alpha_vantage'
      };

      this.setCachedData(`alpha_quote_${symbol}`, quoteData);
      return quoteData;

    } catch (error) {
      logger.error(`Alpha Vantage error for ${symbol}:`, error.message);
      throw error;
    }
  }

  /**
   * Get stock quote with fallback between sources
   */
  async getStockQuote(symbol) {
    const sources = [
      () => this.getYahooQuote(symbol),
      () => this.getAlphaVantageQuote(symbol)
    ];

    for (const source of sources) {
      try {
        return await source();
      } catch (error) {
        logger.warn(`Source failed for ${symbol}:`, error.message);
        continue;
      }
    }

    throw new Error(`Failed to fetch quote for ${symbol} from all sources`);
  }

  /**
   * Search for stocks
   */
  async searchStocks(query) {
    // Simple search implementation - in production, you'd use a proper search API
    const commonStocks = [
      { symbol: 'AAPL', name: 'Apple Inc.' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.' },
      { symbol: 'MSFT', name: 'Microsoft Corporation' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.' },
      { symbol: 'TSLA', name: 'Tesla Inc.' },
      { symbol: 'META', name: 'Meta Platforms Inc.' },
      { symbol: 'NVDA', name: 'NVIDIA Corporation' },
      { symbol: 'NFLX', name: 'Netflix Inc.' },
      { symbol: 'AMD', name: 'Advanced Micro Devices Inc.' },
      { symbol: 'INTC', name: 'Intel Corporation' }
    ];

    const queryUpper = query.toUpperCase();
    return commonStocks.filter(stock => 
      stock.symbol.includes(queryUpper) || 
      stock.name.toUpperCase().includes(queryUpper)
    ).slice(0, 10);
  }

  /**
   * Get historical data (simplified)
   */
  async getHistoricalData(symbol, period = '1y') {
    try {
      const cached = this.getCachedData(`historical_${symbol}_${period}`);
      if (cached) return cached;

      // This is a simplified implementation
      // In production, you'd fetch real historical data
      const historicalData = {
        symbol: symbol.toUpperCase(),
        period,
        data: [], // Would contain actual historical data
        timestamp: new Date().toISOString()
      };

      this.setCachedData(`historical_${symbol}_${period}`, historicalData);
      return historicalData;

    } catch (error) {
      logger.error(`Historical data error for ${symbol}:`, error.message);
      throw error;
    }
  }
}

module.exports = new DataService();
