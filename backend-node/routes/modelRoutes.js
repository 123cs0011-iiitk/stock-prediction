/**
 * Model Routes - ML algorithm management endpoints
 * Integrates with FastAPI ML service for algorithm operations
 */

const express = require('express');
const router = express.Router();
const mlService = require('../services/mlService');
const logger = require('../utils/logger');

/**
 * GET /api/models/algorithms
 * Get available ML algorithms
 */
router.get('/algorithms', async (req, res) => {
  try {
    logger.info('Requesting available algorithms');

    const algorithms = await mlService.getAvailableAlgorithms();

    res.json({
      success: true,
      data: algorithms,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Failed to get algorithms:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get available algorithms',
      details: error.message
    });
  }
});

/**
 * POST /api/models/:type/:algorithm
 * Run specific ML algorithm
 */
router.post('/:type/:algorithm', async (req, res) => {
  try {
    const { type, algorithm } = req.params;
    const { data, parameters = {} } = req.body;

    if (!data) {
      return res.status(400).json({
        success: false,
        error: 'Data is required for algorithm execution'
      });
    }

    logger.info(`Running ${type}/${algorithm} algorithm`);

    const result = await mlService.runAlgorithm(type, algorithm, data, parameters);

    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Algorithm execution error (${req.params.type}/${req.params.algorithm}):`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to execute algorithm',
      details: error.message
    });
  }
});

/**
 * GET /api/models/:type/:algorithm/metrics/:symbol
 * Get algorithm performance metrics
 */
router.get('/:type/:algorithm/metrics/:symbol', async (req, res) => {
  try {
    const { type, algorithm, symbol } = req.params;

    logger.info(`Getting metrics for ${type}/${algorithm} on ${symbol}`);

    const metrics = await mlService.getAlgorithmMetrics(type, algorithm, symbol);

    res.json({
      success: true,
      data: metrics,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Metrics error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get algorithm metrics',
      details: error.message
    });
  }
});

/**
 * POST /api/models/train
 * Train ML models with custom parameters
 */
router.post('/train', async (req, res) => {
  try {
    const { symbol, algorithm, parameters = {} } = req.body;

    if (!symbol || !algorithm) {
      return res.status(400).json({
        success: false,
        error: 'Symbol and algorithm are required'
      });
    }

    logger.info(`Training ${algorithm} for ${symbol}`);

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
 * GET /api/models/status
 * Get overall ML service status
 */
router.get('/status', async (req, res) => {
  try {
    const health = await mlService.healthCheck();

    res.json({
      success: true,
      data: health,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Status check error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get ML service status',
      details: error.message
    });
  }
});

module.exports = router;
