const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const logger = require('./logger');

class PythonExecutor {
  constructor() {
    this.pythonPath = process.env.PYTHON_PATH || 'python';
    this.scriptsPath = path.join(__dirname, '../scripts/python');
    this.timeout = 30000; // 30 seconds timeout
  }

  /**
   * Execute a Python script with arguments
   * @param {string} scriptName - Name of the Python script
   * @param {Array} args - Arguments to pass to the script
   * @param {Object} inputData - Data to pass as JSON input
   * @returns {Promise<Object>} - Result from the Python script
   */
  async executeScript(scriptName, args = [], inputData = null) {
    return new Promise((resolve, reject) => {
      const scriptPath = path.join(this.scriptsPath, `${scriptName}.py`);
      
      // Validate script exists
      fs.access(scriptPath)
        .then(() => {
          const pythonProcess = spawn(this.pythonPath, [scriptPath, ...args], {
            stdio: ['pipe', 'pipe', 'pipe'],
            timeout: this.timeout
          });

          let stdout = '';
          let stderr = '';

          // Handle stdout
          pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
          });

          // Handle stderr
          pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
          });

          // Send input data if provided
          if (inputData) {
            pythonProcess.stdin.write(JSON.stringify(inputData));
            pythonProcess.stdin.end();
          }

          // Handle process completion
          pythonProcess.on('close', (code) => {
            if (code === 0) {
              try {
                const result = JSON.parse(stdout);
                resolve(result);
              } catch (parseError) {
                logger.error('Failed to parse Python script output:', parseError);
                resolve({ 
                  success: false, 
                  error: 'Invalid JSON output from Python script',
                  rawOutput: stdout 
                });
              }
            } else {
              logger.error(`Python script failed with code ${code}:`, stderr);
              reject(new Error(`Python script failed: ${stderr}`));
            }
          });

          // Handle process timeout
          pythonProcess.on('error', (error) => {
            logger.error('Python process error:', error);
            reject(error);
          });

          // Set timeout
          setTimeout(() => {
            pythonProcess.kill('SIGTERM');
            reject(new Error('Python script execution timeout'));
          }, this.timeout);

        })
        .catch((error) => {
          reject(new Error(`Python script not found: ${scriptPath}`));
        });
    });
  }

  /**
   * Execute ML prediction using Python
   * @param {string} symbol - Stock symbol
   * @param {Array} prices - Historical prices
   * @param {Array} volumes - Historical volumes
   * @returns {Promise<Object>} - Prediction results
   */
  async executeMLPrediction(symbol, prices, volumes) {
    const inputData = {
      symbol,
      prices,
      volumes,
      timestamp: new Date().toISOString()
    };

    return await this.executeScript('ml_prediction', [], inputData);
  }

  /**
   * Execute technical indicators calculation
   * @param {Array} prices - Historical prices
   * @param {Array} volumes - Historical volumes
   * @returns {Promise<Object>} - Technical indicators
   */
  async calculateTechnicalIndicators(prices, volumes) {
    const inputData = {
      prices,
      volumes,
      timestamp: new Date().toISOString()
    };

    return await this.executeScript('technical_indicators', [], inputData);
  }

  /**
   * Execute ARIMA time series prediction
   * @param {Array} prices - Historical prices
   * @param {number} steps - Number of future steps to predict
   * @returns {Promise<Object>} - ARIMA prediction results
   */
  async executeARIMAPrediction(prices, steps = 5) {
    const inputData = {
      prices,
      steps,
      timestamp: new Date().toISOString()
    };

    return await this.executeScript('arima_prediction', [], inputData);
  }

  /**
   * Execute ensemble model training and prediction
   * @param {string} symbol - Stock symbol
   * @param {Array} prices - Historical prices
   * @param {Array} volumes - Historical volumes
   * @returns {Promise<Object>} - Ensemble prediction results
   */
  async executeEnsemblePrediction(symbol, prices, volumes) {
    const inputData = {
      symbol,
      prices,
      volumes,
      timestamp: new Date().toISOString()
    };

    return await this.executeScript('ensemble_prediction', [], inputData);
  }

  /**
   * Execute data preprocessing
   * @param {Object} rawData - Raw stock data
   * @returns {Promise<Object>} - Processed data
   */
  async preprocessData(rawData) {
    const inputData = {
      rawData,
      timestamp: new Date().toISOString()
    };

    return await this.executeScript('data_preprocessing', [], inputData);
  }

  /**
   * Check if Python environment is available
   * @returns {Promise<boolean>} - True if Python is available
   */
  async checkPythonEnvironment() {
    return new Promise((resolve) => {
      const pythonProcess = spawn(this.pythonPath, ['--version']);
      
      pythonProcess.on('close', (code) => {
        resolve(code === 0);
      });
      
      pythonProcess.on('error', () => {
        resolve(false);
      });
    });
  }

  /**
   * Get Python environment info
   * @returns {Promise<Object>} - Python environment information
   */
  async getPythonEnvironmentInfo() {
    try {
      const result = await this.executeScript('environment_info', []);
      return result;
    } catch (error) {
      return {
        success: false,
        error: error.message,
        pythonPath: this.pythonPath,
        available: await this.checkPythonEnvironment()
      };
    }
  }
}

module.exports = new PythonExecutor();
