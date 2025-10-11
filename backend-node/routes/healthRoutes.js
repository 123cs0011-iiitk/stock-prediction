/**
 * Health Routes - Health check endpoints
 */

const express = require('express');
const router = express.Router();
const mlService = require('../services/mlService');
const dataService = require('../services/dataService');
const { sequelize } = require('../config/database');
const logger = require('../utils/logger');

/**
 * GET /api/health
 * Comprehensive health check
 */
router.get('/', async (req, res) => {
  try {
    const healthChecks = {
      api: {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '2.0.0'
      },
      database: {
        status: 'unknown'
      },
      ml_service: {
        status: 'unknown'
      },
      data_service: {
        status: 'healthy'
      }
    };

    // Check database connection
    try {
      await sequelize.authenticate();
      healthChecks.database.status = 'healthy';
    } catch (error) {
      healthChecks.database.status = 'unhealthy';
      healthChecks.database.error = error.message;
    }

    // Check ML service
    try {
      const mlHealth = await mlService.healthCheck();
      healthChecks.ml_service = mlHealth;
    } catch (error) {
      healthChecks.ml_service.status = 'unhealthy';
      healthChecks.ml_service.error = error.message;
    }

    // Determine overall status
    const overallStatus = 
      healthChecks.database.status === 'healthy' && 
      healthChecks.ml_service.status === 'healthy' ? 'healthy' : 'degraded';

    const statusCode = overallStatus === 'healthy' ? 200 : 503;

    res.status(statusCode).json({
      success: true,
      status: overallStatus,
      services: healthChecks,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Health check error:', error);
    res.status(500).json({
      success: false,
      status: 'unhealthy',
      error: 'Health check failed',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * GET /api/health/database
 * Database-specific health check
 */
router.get('/database', async (req, res) => {
  try {
    await sequelize.authenticate();
    
    res.json({
      success: true,
      status: 'healthy',
      message: 'Database connection successful',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Database health check error:', error);
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'Database connection failed',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * GET /api/health/ml
 * ML service health check
 */
router.get('/ml', async (req, res) => {
  try {
    const mlHealth = await mlService.healthCheck();
    
    const statusCode = mlHealth.status === 'healthy' ? 200 : 503;
    
    res.status(statusCode).json({
      success: mlHealth.status === 'healthy',
      data: mlHealth,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('ML service health check error:', error);
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'ML service health check failed',
      details: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

module.exports = router;
