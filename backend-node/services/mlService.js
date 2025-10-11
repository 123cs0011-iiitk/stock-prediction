/**
 * ML Service - Communication with FastAPI ML backend
 * Handles all ML-related requests by forwarding them to the FastAPI service
 */

const axios = require('axios');
const logger = require('../utils/logger');

class MLService {
  constructor() {
    // FastAPI ML service configuration
    this.mlServiceUrl = process.env.ML_SERVICE_URL || 'http://localhost:8000';
    this.timeout = 30000; // 30 seconds timeout for ML operations
    
    // Create axios instance with default config
    this.mlClient = axios.create({
      baseURL: this.mlServiceUrl,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });

    // Add request interceptor for logging
    this.mlClient.interceptors.request.use(
      (config) => {
        logger.info(`ML Service Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        logger.error('ML Service Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Add response interceptor for logging
    this.mlClient.interceptors.response.use(
      (response) => {
        logger.info(`ML Service Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        logger.error('ML Service Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Check if ML service is available
   */
  async healthCheck() {
    try {
      const response = await this.mlClient.get('/health');
      return {
        status: 'healthy',
        ml_service: response.data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('ML Service health check failed:', error.message);
      return {
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }

  /**
   * Get available ML algorithms
   */
  async getAvailableAlgorithms() {
    try {
      const response = await this.mlClient.get('/api/v1/algorithms/available');
      return response.data;
    } catch (error) {
      logger.error('Failed to get available algorithms:', error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Get stock prediction using ML algorithms
   */
  async getStockPrediction(symbol, algorithm = 'ensemble', parameters = {}) {
    try {
      const payload = {
        symbol: symbol.toUpperCase(),
        algorithm,
        parameters
      };
      
      const response = await this.mlClient.post('/api/v1/predictions/predict', payload);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get prediction for ${symbol}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Train ML models for a specific symbol
   */
  async trainModels(symbol, algorithm, parameters = {}) {
    try {
      const payload = {
        symbol: symbol.toUpperCase(),
        algorithm,
        parameters
      };
      
      const response = await this.mlClient.post('/api/v1/algorithms/train', payload);
      return response.data;
    } catch (error) {
      logger.error(`Failed to train models for ${symbol}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Get model status for a symbol
   */
  async getModelStatus(symbol) {
    try {
      const response = await this.mlClient.get(`/api/v1/algorithms/status/${symbol.toUpperCase()}`);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get model status for ${symbol}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Run specific ML algorithm
   */
  async runAlgorithm(algorithmType, algorithmName, data, parameters = {}) {
    try {
      const payload = {
        algorithm: algorithmName,
        data,
        parameters
      };
      
      const response = await this.mlClient.post(`/api/v1/algorithms/${algorithmType}/${algorithmName}`, payload);
      return response.data;
    } catch (error) {
      logger.error(`Failed to run ${algorithmName}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Get algorithm performance metrics
   */
  async getAlgorithmMetrics(algorithmType, algorithmName, symbol) {
    try {
      const response = await this.mlClient.get(`/api/v1/algorithms/${algorithmType}/${algorithmName}/metrics/${symbol.toUpperCase()}`);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get metrics for ${algorithmName}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Batch predict multiple symbols
   */
  async batchPredict(symbols, algorithm = 'ensemble') {
    try {
      const payload = {
        symbols: symbols.map(s => s.toUpperCase()),
        algorithm
      };
      
      const response = await this.mlClient.post('/api/v1/predictions/batch', payload);
      return response.data;
    } catch (error) {
      logger.error('Failed to batch predict:', error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Get comprehensive ML analysis for a symbol
   */
  async getComprehensiveAnalysis(symbol) {
    try {
      const response = await this.mlClient.get(`/api/v1/predictions/analysis/${symbol.toUpperCase()}`);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get comprehensive analysis for ${symbol}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }

  /**
   * Get ML model recommendations for a symbol
   */
  async getModelRecommendations(symbol, marketConditions = {}) {
    try {
      const payload = {
        symbol: symbol.toUpperCase(),
        market_conditions: marketConditions
      };
      
      const response = await this.mlClient.post('/api/v1/algorithms/recommend', payload);
      return response.data;
    } catch (error) {
      logger.error(`Failed to get model recommendations for ${symbol}:`, error.message);
      throw new Error(`ML Service Error: ${error.message}`);
    }
  }
}

module.exports = new MLService();
