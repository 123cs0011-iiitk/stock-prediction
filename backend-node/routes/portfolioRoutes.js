/**
 * Portfolio Routes - Portfolio management endpoints
 */

const express = require('express');
const router = express.Router();
const dataService = require('../services/dataService');
const logger = require('../utils/logger');

/**
 * POST /api/portfolio/analyze
 * Analyze portfolio performance
 */
router.post('/analyze', async (req, res) => {
  try {
    const { portfolio } = req.body;

    if (!portfolio || !Array.isArray(portfolio) || portfolio.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Portfolio array is required'
      });
    }

    logger.info(`Portfolio analysis request for ${portfolio.length} positions`);

    let totalCost = 0;
    let totalValue = 0;
    const positions = [];

    // Process each position
    for (const position of portfolio) {
      const { symbol, shares, costBasis } = position;

      if (!symbol || !shares || !costBasis) {
        positions.push({
          symbol: symbol || 'UNKNOWN',
          error: 'Missing required fields: symbol, shares, or costBasis',
          shares: shares || 0,
          costBasis: costBasis || 0,
          currentValue: costBasis || 0,
          gainLoss: 0,
          gainLossPercent: 0
        });
        totalCost += costBasis || 0;
        continue;
      }

      try {
        // Get current price
        const quote = await dataService.getStockQuote(symbol);
        const currentPrice = quote.price;
        const currentValue = currentPrice * shares;
        
        totalValue += currentValue;
        totalCost += costBasis;

        const gainLoss = currentValue - costBasis;
        const gainLossPercent = (gainLoss / costBasis) * 100;

        positions.push({
          symbol: symbol.toUpperCase(),
          shares,
          costBasis,
          currentPrice,
          currentValue,
          gainLoss,
          gainLossPercent,
          change: quote.change,
          changePercent: quote.changePercent,
          dataSource: quote.source
        });

      } catch (error) {
        logger.warn(`Failed to get price for ${symbol}:`, error.message);
        
        // Use cost basis if current price unavailable
        totalCost += costBasis;
        positions.push({
          symbol: symbol.toUpperCase(),
          shares,
          costBasis,
          currentPrice: 0,
          currentValue: costBasis,
          gainLoss: 0,
          gainLossPercent: 0,
          error: `Price unavailable: ${error.message}`,
          dataSource: 'cost_basis'
        });
      }
    }

    // Calculate portfolio metrics
    const totalGainLoss = totalValue - totalCost;
    const totalReturnPercent = totalCost > 0 ? (totalGainLoss / totalCost) * 100 : 0;

    // Calculate risk metrics
    const diversificationScore = Math.min(portfolio.length * 10, 100);
    const volatilityWarning = portfolio.length < 5 ? 'High' : 
                             portfolio.length < 10 ? 'Medium' : 'Low';

    const analysis = {
      summary: {
        totalCost,
        totalValue,
        totalGainLoss,
        totalReturnPercent: Math.round(totalReturnPercent * 100) / 100
      },
      positions,
      riskMetrics: {
        diversificationScore,
        volatilityWarning,
        positionCount: portfolio.length,
        successfulQuotes: positions.filter(p => !p.error).length,
        failedQuotes: positions.filter(p => p.error).length
      },
      timestamp: new Date().toISOString()
    };

    res.json({
      success: true,
      data: analysis,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Portfolio analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to analyze portfolio',
      details: error.message
    });
  }
});

/**
 * POST /api/portfolio/summary
 * Get portfolio summary without detailed analysis
 */
router.post('/summary', async (req, res) => {
  try {
    const { portfolio } = req.body;

    if (!portfolio || !Array.isArray(portfolio)) {
      return res.status(400).json({
        success: false,
        error: 'Portfolio array is required'
      });
    }

    let totalCost = 0;
    let totalValue = 0;
    let successfulQuotes = 0;

    // Quick calculation
    for (const position of portfolio) {
      const { shares, costBasis, symbol } = position;
      
      if (!symbol || !shares || !costBasis) continue;

      totalCost += costBasis;

      try {
        const quote = await dataService.getStockQuote(symbol);
        totalValue += quote.price * shares;
        successfulQuotes++;
      } catch (error) {
        totalValue += costBasis; // Use cost basis as fallback
      }
    }

    const summary = {
      totalCost,
      totalValue,
      totalGainLoss: totalValue - totalCost,
      totalReturnPercent: totalCost > 0 ? ((totalValue - totalCost) / totalCost) * 100 : 0,
      positionCount: portfolio.length,
      successfulQuotes,
      timestamp: new Date().toISOString()
    };

    res.json({
      success: true,
      data: summary,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Portfolio summary error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get portfolio summary',
      details: error.message
    });
  }
});

module.exports = router;
