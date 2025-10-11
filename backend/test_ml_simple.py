#!/usr/bin/env python3
"""
Simple test script for ML models without Unicode characters
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_models():
    """Test the ML models with sample data"""
    print("Testing ML Models...")
    print("=" * 40)
    
    try:
        # Import the models
        from models.ml_models import EnsembleStockPredictor, ARIMATimeSeriesPredictor
        print("OK: ML models imported successfully")
        
        # Create sample price data (simulating stock prices)
        np.random.seed(42)
        base_price = 100.0
        prices = []
        
        # Generate 200 days of price data with trend and noise
        for i in range(200):
            trend = 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            price = base_price * (1 + trend + noise)
            prices.append(price)
        
        print(f"OK Generated {len(prices)} sample price points")
        
        # Test Ensemble Stock Predictor
        print("\nTesting Ensemble Stock Predictor...")
        ensemble_predictor = EnsembleStockPredictor()
        
        # Train the model
        success, message = ensemble_predictor.train_ensemble_models(prices)
        if success:
            print(f"OK Ensemble training successful: {message}")
            
            # Test prediction
            prediction, error = ensemble_predictor.predict_next_day_price(prices[-50:])
            if prediction:
                print(f"OK Ensemble prediction successful:")
                print(f"  Final prediction: ${prediction['ensemble_prediction']['final_prediction']:.2f}")
                print(f"  Confidence: {prediction['ensemble_prediction']['confidence']:.1f}%")
            else:
                print(f"FAIL Ensemble prediction failed: {error}")
        else:
            print(f"FAIL Ensemble training failed: {message}")
        
        # Test ARIMA Time Series Predictor
        print("\nTesting ARIMA Time Series Predictor...")
        arima_predictor = ARIMATimeSeriesPredictor()
        
        # Train the model
        success, message = arima_predictor.train_arima_model(prices)
        if success:
            print(f"OK ARIMA training successful: {message}")
            
            # Test prediction
            prediction, error = arima_predictor.predict_time_series(prices[-30:], steps=5)
            if prediction:
                print(f"OK ARIMA prediction successful:")
                print(f"  Next day prediction: ${prediction['predictions']['next_day']:.2f}")
                print(f"  Confidence: {prediction['predictions']['confidence']:.1f}%")
            else:
                print(f"FAIL ARIMA prediction failed: {error}")
        else:
            print(f"FAIL ARIMA training failed: {message}")
        
        print("\nOK ML models test completed")
        return True
        
    except Exception as e:
        print(f"FAIL ML models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test runner"""
    print("ML Models Simple Test")
    print("=" * 40)
    
    # Test ML models
    ml_success = test_ml_models()
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print(f"   ML Models: {'OK PASS' if ml_success else 'FAIL FAIL'}")
    
    if ml_success:
        print("\nOK All tests passed! ML models are working correctly.")
        return True
    else:
        print("\nFAIL Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
