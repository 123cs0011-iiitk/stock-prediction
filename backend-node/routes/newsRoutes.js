/**
 * News Routes - Stock news endpoints
 */

const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');

/**
 * GET /api/news/:symbol
 * Get news for a specific symbol
 */
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { limit = 10 } = req.query;

    logger.info(`News request for ${symbol}`);

    // For now, return mock news data
    // In production, you would integrate with news APIs like Finnhub, Polygon, etc.
    const mockNews = [
      {
        headline: `${symbol} Stock Analysis: Technical Indicators Show Bullish Trend`,
        summary: `Recent analysis of ${symbol} shows strong technical indicators pointing to a potential upward movement.`,
        url: `https://example.com/news/${symbol}-analysis`,
        source: 'Financial News',
        publishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
        category: 'analysis'
      },
      {
        headline: `${symbol} Earnings Report Expected Next Week`,
        summary: `Investors are eagerly awaiting the quarterly earnings report from ${symbol}.`,
        url: `https://example.com/news/${symbol}-earnings`,
        source: 'Market Watch',
        publishedAt: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
        category: 'earnings'
      },
      {
        headline: `Market Update: ${symbol} Trading Volume Increases`,
        summary: `Trading volume for ${symbol} has increased significantly in today's session.`,
        url: `https://example.com/news/${symbol}-volume`,
        source: 'Trading News',
        publishedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(), // 6 hours ago
        category: 'market'
      }
    ];

    const news = mockNews.slice(0, parseInt(limit));

    res.json({
      success: true,
      data: {
        symbol: symbol.toUpperCase(),
        news,
        count: news.length,
        timestamp: new Date().toISOString()
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('News error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch news',
      details: error.message
    });
  }
});

/**
 * GET /api/news/market/general
 * Get general market news
 */
router.get('/market/general', async (req, res) => {
  try {
    const { limit = 20 } = req.query;

    logger.info('General market news request');

    // Mock general market news
    const mockNews = [
      {
        headline: 'Market Opens Higher as Tech Stocks Lead Gains',
        summary: 'Major indices opened higher today with technology stocks leading the gains.',
        url: 'https://example.com/news/market-opens-higher',
        source: 'Market News',
        publishedAt: new Date(Date.now() - 30 * 60 * 1000).toISOString(), // 30 minutes ago
        category: 'market'
      },
      {
        headline: 'Federal Reserve Signals Potential Rate Changes',
        summary: 'The Federal Reserve has indicated potential changes to interest rates in the coming months.',
        url: 'https://example.com/news/fed-rate-changes',
        source: 'Economic News',
        publishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
        category: 'economics'
      },
      {
        headline: 'Energy Sector Sees Strong Performance This Week',
        summary: 'Energy stocks have shown strong performance throughout the week.',
        url: 'https://example.com/news/energy-sector-performance',
        source: 'Sector News',
        publishedAt: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString(), // 4 hours ago
        category: 'sectors'
      }
    ];

    const news = mockNews.slice(0, parseInt(limit));

    res.json({
      success: true,
      data: {
        news,
        count: news.length,
        category: 'general_market',
        timestamp: new Date().toISOString()
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('General news error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch general market news',
      details: error.message
    });
  }
});

module.exports = router;
