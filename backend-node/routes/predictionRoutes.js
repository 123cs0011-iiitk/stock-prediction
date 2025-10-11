/**
 * Prediction Routes - Stock prediction endpoints
 * Integrates with FastAPI ML service for predictions
 */

const express = require('express');
const router = express.Router();
const mlService = require('../services/mlService');
const logger = require('../utils/logger');

/**
 * GET /api/predict/:symbol
 * Get stock prediction using ML algorithms
 */
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { algorithm = 'ensemble', ...parameters } = req.query;

    logger.info(`Prediction request for ${symbol} using ${algorithm}`);

    // Get prediction from ML service
    const prediction = await mlService.getStockPrediction(symbol, algorithm, parameters);

    res.json({
      success: true,
      data: prediction,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get prediction',
      details: error.message
    });
  }
});

/**
 * POST /api/predict/batch
 * Batch predict multiple symbols
 */
router.post('/batch', async (req, res) => {
  try {
    const { symbols, algorithm = 'ensemble' } = req.body;

    if (!symbols || !Array.isArray(symbols)) {
      return res.status(400).json({
        success: false,
        error: 'Symbols array is required'
      });
    }

    logger.info(`Batch prediction request for ${symbols.length} symbols`);

    const predictions = await mlService.batchPredict(symbols, algorithm);

    res.json({
      success: true,
      data: predictions,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Batch prediction error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get batch predictions',
      details: error.message
    });
  }
});

/**
 * GET /api/predict/analysis/:symbol
 * Get comprehensive ML analysis for a symbol
 */
router.get('/analysis/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;

    logger.info(`Analysis request for ${symbol}`);

    const analysis = await mlService.getComprehensiveAnalysis(symbol);

    res.json({
      success: true,
      data: analysis,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get analysis',
      details: error.message
    });
  }
});

/**
 * POST /api/predict/train/:symbol
 * Train ML models for a specific symbol
 */
router.post('/train/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { algorithm = 'ensemble', ...parameters } = req.body;

    logger.info(`Training request for ${symbol} using ${algorithm}`);

    const result = await mlService.trainModels(symbol, algorithm, parameters);

    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Training error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to train models',
      details: error.message
    });
  }
});

/**
 * GET /api/predict/status/:symbol
 * Get model status for a symbol
 */
router.get('/status/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;

    const status = await mlService.getModelStatus(symbol);

    res.json({
      success: true,
      data: status,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Status check error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get model status',
      details: error.message
    });
  }
});

/**
 * POST /api/predict/recommend/:symbol
 * Get ML model recommendations for a symbol
 */
router.post('/recommend/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { marketConditions = {} } = req.body;

    const recommendations = await mlService.getModelRecommendations(symbol, marketConditions);

    res.json({
      success: true,
      data: recommendations,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Recommendation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get recommendations',
      details: error.message
    });
  }
});

module.exports = router;
