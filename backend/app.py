from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import random
from models.ml_models import StockPredictor, ARIMAPredictor

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Cache for stock data to avoid repeated API calls
stock_cache = {}
cache_duration = 300  # 5 minutes

# ML Models for stock prediction
ml_predictor = StockPredictor()
arima_predictor = ARIMAPredictor()
ml_models_trained = {}

# Mock stock database for demonstration
MOCK_STOCKS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'base_price': 150.0,
        'volatility': 0.02
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'industry': 'Internet Services',
        'base_price': 2800.0,
        'volatility': 0.025
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'sector': 'Technology',
        'industry': 'Software',
        'base_price': 300.0,
        'volatility': 0.018
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'sector': 'Automotive',
        'industry': 'Electric Vehicles',
        'base_price': 800.0,
        'volatility': 0.04
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'sector': 'Consumer Discretionary',
        'industry': 'E-commerce',
        'base_price': 3200.0,
        'volatility': 0.03
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'sector': 'Technology',
        'industry': 'Semiconductors',
        'base_price': 500.0,
        'volatility': 0.035
    },
    'META': {
        'name': 'Meta Platforms Inc.',
        'sector': 'Technology',
        'industry': 'Social Media',
        'base_price': 350.0,
        'volatility': 0.03
    },
    'NFLX': {
        'name': 'Netflix Inc.',
        'sector': 'Communication Services',
        'industry': 'Streaming',
        'base_price': 450.0,
        'volatility': 0.04
    }
}

def get_cached_data(symbol):
    """Get cached stock data if it's still valid"""
    if symbol in stock_cache:
        cache_time, data = stock_cache[symbol]
        if datetime.now() - cache_time < timedelta(seconds=cache_duration):
            return data
    return None

def set_cached_data(symbol, data):
    """Cache stock data with timestamp"""
    stock_cache[symbol] = (datetime.now(), data)

def generate_mock_price(base_price, volatility, days_back=0):
    """Generate realistic mock price data"""
    # Add some randomness based on days back
    time_factor = 1 + (days_back * 0.001)
    random_factor = random.uniform(0.95, 1.05)
    volatility_factor = random.uniform(1 - volatility, 1 + volatility)
    
    return round(base_price * time_factor * random_factor * volatility_factor, 2)

def generate_mock_historical_data(symbol, days=365):
    """Generate mock historical data for a stock"""
    if symbol not in MOCK_STOCKS:
        return None
    
    stock_info = MOCK_STOCKS[symbol]
    base_price = stock_info['base_price']
    volatility = stock_info['volatility']
    
    dates = []
    prices = []
    volumes = []
    sma_20 = []
    sma_50 = []
    rsi_values = []
    
    current_price = base_price
    
    for i in range(days, 0, -1):
        # Generate date
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Generate price with realistic movement
        price = generate_mock_price(base_price, volatility, i)
        prices.append(price)
        current_price = price
        
        # Generate volume (correlated with price movement)
        volume = random.randint(1000000, 10000000)
        volumes.append(volume)
        
        # Calculate moving averages (simplified)
        if len(prices) >= 20:
            sma_20.append(round(sum(prices[-20:]) / 20, 2))
        else:
            sma_20.append(price)
            
        if len(prices) >= 50:
            sma_50.append(round(sum(prices[-50:]) / 50, 2))
        else:
            sma_50.append(price)
        
        # Generate RSI (simplified)
        rsi = random.uniform(30, 70)
        rsi_values.append(round(rsi, 1))
    
    return {
        'dates': dates,
        'prices': prices,
        'volumes': volumes,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'rsi': rsi_values
    }

def get_mock_stock_info(symbol):
    """Get mock stock information"""
    if symbol not in MOCK_STOCKS:
        return None
    
    stock_info = MOCK_STOCKS[symbol]
    base_price = stock_info['base_price']
    
    # Generate current price with some variation
    current_price = generate_mock_price(base_price, stock_info['volatility'])
    price_change = random.uniform(-base_price * 0.1, base_price * 0.1)
    change_percent = (price_change / current_price) * 100
    
    return {
        'name': stock_info['name'],
        'price': current_price,
        'change': round(price_change, 2),
        'changePercent': round(change_percent, 2),
        'volume': random.randint(1000000, 50000000),
        'marketCap': base_price * random.randint(1000000, 10000000),
        'currency': 'USD',
        'pe': round(random.uniform(15, 35), 1),
        'dividend': round(random.uniform(0, 3), 2),
        'sector': stock_info['sector'],
        'industry': stock_info['industry']
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Stock Prediction API'
    })

@app.route('/api/search', methods=['GET'])
def search_stocks():
    """Search for stocks by symbol or company name"""
    query = request.args.get('q', '').strip().upper()
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    # Check if stock exists in our mock database
    if query not in MOCK_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    try:
        # Get mock stock info
        stock_info = get_mock_stock_info(query)
        
        result = {
            'symbol': query,
            'name': stock_info['name'],
            'price': stock_info['price'],
            'change': stock_info['change'],
            'changePercent': stock_info['changePercent'],
            'volume': stock_info['volume'],
            'marketCap': stock_info['marketCap'],
            'currency': stock_info['currency']
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error searching stock: {str(e)}'}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Get detailed stock data and historical prices"""
    symbol = symbol.upper()
    
    # Check cache first
    cached_data = get_cached_data(symbol)
    if cached_data:
        return jsonify(cached_data)
    
    # Check if stock exists in our mock database
    if symbol not in MOCK_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    try:
        # Get mock stock info
        stock_info = get_mock_stock_info(symbol)
        
        # Generate mock historical data
        historical_data = generate_mock_historical_data(symbol)
        
        if not historical_data:
            return jsonify({'error': 'Unable to generate historical data'}), 500
        
        # Calculate technical indicators from mock data
        prices = historical_data['prices']
        volatility = np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252) * 100 if len(prices) > 1 else 20.0
        
        # Prepare data for frontend
        stock_data = {
            'symbol': symbol,
            'info': stock_info,
            'historical': historical_data,
            'technical': {
                'sma_20': historical_data['sma_20'][-1] if historical_data['sma_20'] else None,
                'sma_50': historical_data['sma_50'][-1] if historical_data['sma_50'] else None,
                'rsi': historical_data['rsi'][-1] if historical_data['rsi'] else None,
                'volatility': round(volatility, 1)
            }
        }
        
        # Cache the data
        set_cached_data(symbol, stock_data)
        
        return jsonify(stock_data)
    
    except Exception as e:
        return jsonify({'error': f'Error fetching stock data: {str(e)}'}), 500

@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    """Get stock price prediction using ML models and technical analysis"""
    symbol = symbol.upper()
    
    # Check if stock exists in our mock database
    if symbol not in MOCK_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    try:
        # Get mock stock info
        stock_info = get_mock_stock_info(symbol)
        current_price = stock_info['price']
        
        # Generate mock historical data for prediction
        historical_data = generate_mock_historical_data(symbol, days=100)
        
        if not historical_data:
            return jsonify({'error': 'Unable to generate historical data'}), 500
        
        # Extract prices and volume for ML models
        prices = historical_data['prices']
        volume = historical_data.get('volume', [random.randint(1000000, 10000000) for _ in range(len(prices))])
        
        # Train ML models if not already trained for this symbol
        if symbol not in ml_models_trained:
            # Train StockPredictor
            success, message = ml_predictor.train_models(prices, volume)
            if success:
                ml_models_trained[symbol] = True
            else:
                print(f"ML training failed for {symbol}: {message}")
        
        # Train ARIMA model
        arima_success, arima_message = arima_predictor.train(prices)
        
        # Get ML predictions
        ml_prediction = None
        if symbol in ml_models_trained:
            ml_prediction, ml_error = ml_predictor.predict_next_day(prices, volume)
        
        # Get ARIMA predictions
        arima_prediction = None
        if arima_success:
            arima_prediction, arima_error = arima_predictor.predict(prices, steps=5)
        
        # Calculate technical indicators
        sma_20 = sum(prices[-20:]) / 20 if len(prices) >= 20 else current_price
        sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else current_price
        
        # Calculate trend based on ML predictions if available
        if ml_prediction:
            ensemble_pred = ml_prediction['ensemble']
            trend_change = (ensemble_pred - current_price) / current_price
            if trend_change > 0.01:
                trend = 'bullish'
                confidence = min(ml_prediction['confidence'], 90)
            elif trend_change < -0.01:
                trend = 'bearish'
                confidence = min(ml_prediction['confidence'], 90)
            else:
                trend = 'neutral'
                confidence = ml_prediction['confidence']
        else:
            # Fallback to simple trend analysis
            recent_trend = random.uniform(-0.02, 0.02)
            if sma_20 > sma_50 and recent_trend > 0:
                trend = 'bullish'
                confidence = min(70 + (recent_trend * 1000), 85)
            elif sma_20 < sma_50 and recent_trend < 0:
                trend = 'bearish'
                confidence = min(70 + abs(recent_trend * 1000), 85)
            else:
                trend = 'neutral'
                confidence = 50
        
        # Calculate support and resistance levels
        support = current_price * random.uniform(0.85, 0.95)
        resistance = current_price * random.uniform(1.05, 1.15)
        
        # Calculate next day range
        volatility = MOCK_STOCKS[symbol]['volatility']
        next_day_low = current_price * (1 - volatility)
        next_day_high = current_price * (1 + volatility)
        
        # Prepare prediction response
        prediction = {
            'symbol': symbol,
            'current_price': current_price,
            'prediction': {
                'trend': trend,
                'confidence': round(confidence, 1),
                'support_level': round(support, 2),
                'resistance_level': round(resistance, 2),
                'next_day_range': {
                    'low': round(next_day_low, 2),
                    'high': round(next_day_high, 2)
                }
            },
            'technical_indicators': {
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'trend_strength': round(abs(trend_change if ml_prediction else recent_trend * 100), 2)
            },
            'ml_predictions': ml_prediction,
            'arima_predictions': arima_prediction,
            'model_status': {
                'ml_models_trained': symbol in ml_models_trained,
                'arima_trained': arima_success
            }
        }
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': f'Error generating prediction: {str(e)}'}), 500

@app.route('/api/portfolio/analyze', methods=['POST'])
def analyze_portfolio():
    """Analyze portfolio performance and risk"""
    try:
        data = request.get_json()
        portfolio = data.get('portfolio', [])
        
        if not portfolio:
            return jsonify({'error': 'Portfolio data required'}), 400
        
        total_value = 0
        total_cost = 0
        positions = []
        
        for position in portfolio:
            symbol = position['symbol']
            shares = position['shares']
            cost_basis = position['costBasis']
            
            try:
                # Get mock current price
                if symbol in MOCK_STOCKS:
                    stock_info = get_mock_stock_info(symbol)
                    current_price = stock_info['price']
                    
                    current_value = current_price * shares
                    total_value += current_value
                    total_cost += cost_basis
                    
                    gain_loss = current_value - cost_basis
                    gain_loss_percent = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                    
                    positions.append({
                        'symbol': symbol,
                        'shares': shares,
                        'costBasis': cost_basis,
                        'currentPrice': current_price,
                        'currentValue': current_value,
                        'gainLoss': gain_loss,
                        'gainLossPercent': gain_loss_percent
                    })
                else:
                    # If stock not in mock database, use cost basis
                    total_cost += cost_basis
                    positions.append({
                        'symbol': symbol,
                        'shares': shares,
                        'costBasis': cost_basis,
                        'currentPrice': 0,
                        'currentValue': cost_basis,
                        'gainLoss': 0,
                        'gainLossPercent': 0
                    })
            
            except Exception as e:
                # If we can't get current price, use cost basis
                total_cost += cost_basis
                positions.append({
                    'symbol': symbol,
                    'shares': shares,
                    'costBasis': cost_basis,
                    'currentPrice': 0,
                    'currentValue': cost_basis,
                    'gainLoss': 0,
                    'gainLossPercent': 0
                })
        
        portfolio_return = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        analysis = {
            'summary': {
                'totalCost': total_cost,
                'totalValue': total_value,
                'totalGainLoss': total_value - total_cost,
                'totalReturnPercent': round(portfolio_return, 2)
            },
            'positions': positions,
            'risk_metrics': {
                'diversification_score': min(len(positions) * 10, 100) if positions else 0,
                'volatility_warning': 'High' if len(positions) < 5 else 'Medium' if len(positions) < 10 else 'Low'
            }
        }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': f'Error analyzing portfolio: {str(e)}'}), 500

@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Get status of ML models"""
    try:
        status = {
            'ml_predictor': ml_predictor.get_model_info(),
            'arima_predictor': {
                'status': 'Trained' if arima_predictor.is_trained else 'Not trained',
                'model_type': 'ARIMA'
            },
            'trained_symbols': list(ml_models_trained.keys()),
            'total_symbols': len(ml_models_trained)
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f'Error getting model status: {str(e)}'}), 500

@app.route('/api/models/train/<symbol>', methods=['POST'])
def train_models_for_symbol(symbol):
    """Manually train ML models for a specific symbol"""
    symbol = symbol.upper()
    
    if symbol not in MOCK_STOCKS:
        return jsonify({'error': 'Stock not found'}), 404
    
    try:
        # Generate historical data
        historical_data = generate_mock_historical_data(symbol, days=100)
        if not historical_data:
            return jsonify({'error': 'Unable to generate historical data'}), 500
        
        prices = historical_data['prices']
        volume = historical_data.get('volume', [random.randint(1000000, 10000000) for _ in range(len(prices))])
        
        # Train ML models
        ml_success, ml_message = ml_predictor.train_models(prices, volume)
        arima_success, arima_message = arima_predictor.train(prices)
        
        if ml_success:
            ml_models_trained[symbol] = True
        
        result = {
            'symbol': symbol,
            'ml_training': {
                'success': ml_success,
                'message': ml_message
            },
            'arima_training': {
                'success': arima_success,
                'message': arima_message
            },
            'models_trained': symbol in ml_models_trained
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

# TODO: Future API Integration
# When you're ready to add real API keys, replace the mock functions above with:
# 1. Uncomment yfinance import
# 2. Replace get_mock_stock_info() calls with yf.Ticker(symbol).info
# 3. Replace generate_mock_historical_data() calls with ticker.history()
# 4. Add your API keys to .env file
# 5. Update requirements.txt to include yfinance

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (placeholder for future ML implementation)"""
    # This will be implemented with real data in Phase 4
    return [random.uniform(30, 70) for _ in range(len(prices))]

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
