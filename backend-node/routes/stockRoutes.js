/**
 * Stock Routes - Stock data endpoints
 * Handles stock data fetching and management
 */

const express = require('express');
const router = express.Router();
const dataService = require('../services/dataService');
const mlService = require('../services/mlService');
const logger = require('../utils/logger');

/**
 * GET /api/stock/price/:symbol
 * Get real-time stock price
 */
router.get('/price/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { refresh = 'false' } = req.query;

    logger.info(`Stock price request for ${symbol}`);

    // Force refresh if requested
    if (refresh === 'true') {
      // Clear cache for this symbol
      dataService.cache.delete(`yahoo_quote_${symbol}`);
      dataService.cache.delete(`alpha_quote_${symbol}`);
    }

    const quote = await dataService.getStockQuote(symbol);

    res.json({
      success: true,
      data: quote,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Stock price error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch stock price',
      details: error.message
    });
  }
});

/**
 * GET /api/stock/:symbol
 * Get comprehensive stock data including historical data and predictions
 */
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { refresh = 'false', includePrediction = 'true' } = req.query;

    logger.info(`Comprehensive stock data request for ${symbol}`);

    // Get stock quote
    const quote = await dataService.getStockQuote(symbol);

    // Get historical data
    const historical = await dataService.getHistoricalData(symbol);

    // Get ML prediction if requested
    let prediction = null;
    if (includePrediction === 'true') {
      try {
        prediction = await mlService.getStockPrediction(symbol);
      } catch (mlError) {
        logger.warn(`ML prediction failed for ${symbol}:`, mlError.message);
        // Continue without prediction rather than failing the entire request
      }
    }

    const stockData = {
      symbol: symbol.toUpperCase(),
      quote,
      historical,
      prediction,
      timestamp: new Date().toISOString()
    };

    res.json({
      success: true,
      data: stockData,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Stock data error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch stock data',
      details: error.message
    });
  }
});

/**
 * GET /api/stock/search
 * Search for stocks
 */
router.get('/search', async (req, res) => {
  try {
    const { q: query } = req.query;

    if (!query || query.trim().length < 1) {
      return res.status(400).json({
        success: false,
        error: 'Query parameter is required'
      });
    }

    logger.info(`Stock search request for: ${query}`);

    const results = await dataService.searchStocks(query.trim());

    res.json({
      success: true,
      data: results,
      query,
      count: results.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Stock search error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to search stocks',
      details: error.message
    });
  }
});

/**
 * GET /api/stock/:symbol/historical
 * Get historical data for a symbol
 */
router.get('/:symbol/historical', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { period = '1y', refresh = 'false' } = req.query;

    logger.info(`Historical data request for ${symbol}, period: ${period}`);

    if (refresh === 'true') {
      dataService.cache.delete(`historical_${symbol}_${period}`);
    }

    const historical = await dataService.getHistoricalData(symbol, period);

    res.json({
      success: true,
      data: historical,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Historical data error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch historical data',
      details: error.message
    });
  }
});

/**
 * POST /api/stock/batch
 * Get data for multiple symbols at once
 */
router.post('/batch', async (req, res) => {
  try {
    const { symbols, includePrediction = false } = req.body;

    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Symbols array is required'
      });
    }

    if (symbols.length > 20) {
      return res.status(400).json({
        success: false,
        error: 'Maximum 20 symbols allowed per batch request'
      });
    }

    logger.info(`Batch stock data request for ${symbols.length} symbols`);

    const results = await Promise.allSettled(
      symbols.map(async (symbol) => {
        try {
          const quote = await dataService.getStockQuote(symbol);
          let prediction = null;

          if (includePrediction) {
            try {
              prediction = await mlService.getStockPrediction(symbol);
            } catch (mlError) {
              logger.warn(`ML prediction failed for ${symbol}:`, mlError.message);
            }
          }

          return {
            symbol: symbol.toUpperCase(),
            success: true,
            quote,
            prediction
          };
        } catch (error) {
          return {
            symbol: symbol.toUpperCase(),
            success: false,
            error: error.message
          };
        }
      })
    );

    const data = results.map(result => result.value);

    res.json({
      success: true,
      data,
      requested: symbols.length,
      successful: data.filter(item => item.success).length,
      failed: data.filter(item => !item.success).length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Batch stock data error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch batch stock data',
      details: error.message
    });
  }
});

module.exports = router;
