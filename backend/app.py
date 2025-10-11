from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
# Removed random import - no longer needed without mock data
from models.ml_models import EnsembleStockPredictor, ARIMATimeSeriesPredictor
from services.alpha_vantage import alpha_vantage
from services.multi_source_data_manager import multi_source_data_manager
from services.postgresql_database import postgres_db

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Cache for stock data to avoid repeated API calls (legacy - now using database)
stock_cache = {}
cache_duration = 300  # 5 minutes

# ML Models for stock prediction
ensemble_predictor = EnsembleStockPredictor()
arima_predictor = ARIMATimeSeriesPredictor(order=(1, 1, 1))
trained_symbols = {}  # Track which symbols have trained models

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

def get_historical_data_from_alpha_vantage(symbol, outputsize='compact'):
    """Get historical data from Alpha Vantage API"""
    try:
        # Get daily time series data
        historical_data = alpha_vantage.get_daily_time_series(symbol, outputsize)
        
        if not historical_data or 'historical_data' not in historical_data:
            return None
        
        # Process the data
        processed_data = {
            'dates': [day['date'] for day in historical_data['historical_data']],
            'prices': [day['close'] for day in historical_data['historical_data']],
            'volumes': [day['volume'] for day in historical_data['historical_data']],
            'opens': [day['open'] for day in historical_data['historical_data']],
            'highs': [day['high'] for day in historical_data['historical_data']],
            'lows': [day['low'] for day in historical_data['historical_data']]
        }
        
        return processed_data
        
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

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
    """Search for stocks using intelligent data manager"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        # Use multi-source data manager for search
        results = multi_source_data_manager.search_stocks(query)
        
        if not results:
            return jsonify({
                'error': f'No stocks found for query "{query}"'
            }), 404
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Error searching stocks: {str(e)}'}), 500

@app.route('/api/stock/price/<symbol>', methods=['GET'])
def get_live_stock_price(symbol):
    """Get real-time stock price using Yahoo Finance with Alpha Vantage fallback"""
    symbol = symbol.upper().strip()
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    try:
        # Use multi-source data manager for intelligent data fetching
        quote_data = multi_source_data_manager.get_live_quote(symbol, force_refresh)
        
        if not quote_data:
            return jsonify({
                'error': f'Stock symbol "{symbol}" not found or no data available',
                'details': 'Failed to fetch data from all sources'
            }), 404
        
        return jsonify(quote_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Internal server error while fetching stock data',
            'details': str(e)
        }), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Get detailed stock data and historical prices with intelligent caching"""
    symbol = symbol.upper()
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    try:
        # Get comprehensive stock data using multi-source data manager
        comprehensive_data = multi_source_data_manager.get_comprehensive_stock_data(symbol, force_refresh)
        
        if not comprehensive_data:
            return jsonify({'error': 'Unable to fetch stock data'}), 500
        
        quote_data = comprehensive_data['quote']
        company_info = comprehensive_data['company_info']
        historical_data = comprehensive_data['historical_data']
        
        # Calculate technical indicators from real data
        if historical_data and len(historical_data) > 0:
            prices = [day['close'] for day in historical_data]
            
            if len(prices) > 1:
                volatility = np.std(np.diff(prices) / np.array(prices[:-1])) * np.sqrt(252) * 100
            else:
                volatility = 20.0
            
            # Calculate moving averages
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            # Calculate RSI (simplified)
            if len(prices) >= 14:
                price_changes = np.diff(prices[-14:])
                gains = np.where(price_changes > 0, price_changes, 0)
                losses = np.where(price_changes < 0, -price_changes, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50
        else:
            # Fallback values if no historical data
            volatility = 20.0
            sma_20 = quote_data['price']
            sma_50 = quote_data['price']
            rsi = 50
        
        # Prepare stock info
        stock_info = {
            'name': company_info.get('name', f'{symbol} Corporation') if company_info else f'{symbol} Corporation',
            'price': quote_data['price'],
            'change': quote_data['change'],
            'changePercent': quote_data['changePercent'],
            'volume': quote_data['volume'],
            'marketCap': company_info.get('market_cap', 'N/A') if company_info else 'N/A',
            'currency': quote_data['currency'],
            'sector': company_info.get('sector', 'N/A') if company_info else 'N/A',
            'industry': company_info.get('industry', 'N/A') if company_info else 'N/A'
        }
        
        # Format historical data for frontend
        formatted_historical = {
            'dates': [day['date'] for day in historical_data] if historical_data else [],
            'prices': [day['close'] for day in historical_data] if historical_data else [],
            'volumes': [day['volume'] for day in historical_data] if historical_data else [],
            'opens': [day['open'] for day in historical_data] if historical_data else [],
            'highs': [day['high'] for day in historical_data] if historical_data else [],
            'lows': [day['low'] for day in historical_data] if historical_data else []
        }
        
        # Prepare data for frontend
        stock_data = {
            'symbol': symbol,
            'info': stock_info,
            'historical': formatted_historical,
            'technical': {
                'sma_20': round(sma_20, 2),
                'sma_50': round(sma_50, 2),
                'rsi': round(rsi, 1),
                'volatility': round(volatility, 1)
            },
            'data_sources': comprehensive_data['data_sources'],
            'source': 'Data Manager (Yahoo Finance + Alpha Vantage)'
        }
        
        return jsonify(stock_data)
    
    except Exception as e:
        return jsonify({'error': f'Error fetching stock data: {str(e)}'}), 500

@app.route('/api/predict/<symbol>', methods=['GET'])
def predict_stock(symbol):
    """Get stock price prediction using ML models with real Alpha Vantage data"""
    symbol = symbol.upper()
    
    try:
        # Get live stock price data
        quote_data = alpha_vantage.get_global_quote(symbol)
        current_price = quote_data['price']
        
        # Get historical data for ML training
        historical_data = get_historical_data_from_alpha_vantage(symbol, outputsize='compact')
        
        if not historical_data:
            return jsonify({'error': 'Unable to fetch historical data for ML training'}), 500
        
        # Extract prices and volume for ML models
        prices = historical_data['prices']
        volumes = historical_data['volumes']
        
        # Train ensemble models if not already trained for this symbol
        if symbol not in trained_symbols:
            print(f"Training ML models for {symbol}...")
            
            # Train Ensemble Stock Predictor
            ensemble_success, ensemble_message = ensemble_predictor.train_ensemble_models(prices, volumes)
            if ensemble_success:
                trained_symbols[symbol] = {
                    'ensemble_trained': True,
                    'ensemble_message': ensemble_message
                }
            else:
                print(f"Ensemble training failed for {symbol}: {ensemble_message}")
                trained_symbols[symbol] = {
                    'ensemble_trained': False,
                    'ensemble_message': ensemble_message
                }
            
            # Train ARIMA model
            arima_success, arima_message = arima_predictor.train_arima_model(prices)
            if arima_success:
                trained_symbols[symbol]['arima_trained'] = True
                trained_symbols[symbol]['arima_message'] = arima_message
            else:
                print(f"ARIMA training failed for {symbol}: {arima_message}")
                trained_symbols[symbol]['arima_trained'] = False
                trained_symbols[symbol]['arima_message'] = arima_message
        
        # Get ML predictions
        ensemble_prediction = None
        if trained_symbols[symbol]['ensemble_trained']:
            ensemble_prediction, ensemble_error = ensemble_predictor.predict_next_day_price(prices, volumes)
            if ensemble_error:
                print(f"Ensemble prediction error for {symbol}: {ensemble_error}")
        
        # Get ARIMA predictions
        arima_prediction = None
        if trained_symbols[symbol]['arima_trained']:
            arima_prediction, arima_error = arima_predictor.predict_time_series(prices, steps=5)
            if arima_error:
                print(f"ARIMA prediction error for {symbol}: {arima_error}")
        
        # Calculate technical indicators from real data
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
        
        # Calculate trend based on ensemble predictions
        if ensemble_prediction:
            final_pred = ensemble_prediction['ensemble_prediction']['final_prediction']
            confidence = ensemble_prediction['ensemble_prediction']['confidence']
            trend_change = (final_pred - current_price) / current_price
            
            if trend_change > 0.02:
                trend = 'bullish'
            elif trend_change < -0.02:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            # Fallback to simple trend analysis
            if len(prices) >= 20:
                recent_trend = (prices[-1] - prices[-20]) / prices[-20]
            else:
                recent_trend = 0
            
            if recent_trend > 0.01 and sma_20 > sma_50:
                trend = 'bullish'
                confidence = 65
            elif recent_trend < -0.01 and sma_20 < sma_50:
                trend = 'bearish'
                confidence = 65
            else:
                trend = 'neutral'
                confidence = 50
        
        # Calculate support and resistance levels based on historical data
        if len(prices) >= 20:
            recent_lows = min(prices[-20:])
            recent_highs = max(prices[-20:])
            support = recent_lows * 0.98  # 2% below recent low
            resistance = recent_highs * 1.02  # 2% above recent high
        else:
            support = current_price * 0.95
            resistance = current_price * 1.05
        
        # Calculate volatility and next day range
        if len(prices) > 1:
            daily_returns = np.diff(prices) / prices[:-1]
            volatility = np.std(daily_returns) * np.sqrt(252) * 100
        else:
            volatility = 20
        
        next_day_low = current_price * (1 - volatility/100)
        next_day_high = current_price * (1 + volatility/100)
        
        # Prepare comprehensive prediction response
        prediction = {
            'symbol': symbol,
            'current_price': current_price,
            'data_source': 'Alpha Vantage',
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
                'volatility': round(volatility, 1),
                'trend_strength': round(abs(trend_change if ensemble_prediction else recent_trend * 100), 2)
            },
            'ml_predictions': ensemble_prediction,
            'arima_predictions': arima_prediction,
            'model_status': {
                'ensemble_trained': trained_symbols[symbol]['ensemble_trained'],
                'arima_trained': trained_symbols[symbol]['arima_trained'],
                'training_samples': len(prices),
                'data_points_used': len(prices)
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
                # Get live current price from Alpha Vantage
                try:
                    quote_data = alpha_vantage.get_global_quote(symbol)
                    current_price = quote_data['price']
                    
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
                        'gainLossPercent': gain_loss_percent,
                        'dataSource': 'Alpha Vantage'
                    })
                except Exception as price_error:
                    # If live price unavailable, use cost basis
                    print(f"Could not get live price for {symbol}: {price_error}")
                    total_cost += cost_basis
                    positions.append({
                        'symbol': symbol,
                        'shares': shares,
                        'costBasis': cost_basis,
                        'currentPrice': 0,
                        'currentValue': cost_basis,
                        'gainLoss': 0,
                        'gainLossPercent': 0,
                        'dataSource': 'Cost Basis (Live price unavailable)'
                    })
            
            except Exception as e:
                # If we can't get current price, use cost basis
                print(f"Error processing position for {symbol}: {e}")
                total_cost += cost_basis
                positions.append({
                    'symbol': symbol,
                    'shares': shares,
                    'costBasis': cost_basis,
                    'currentPrice': 0,
                    'currentValue': cost_basis,
                    'gainLoss': 0,
                    'gainLossPercent': 0,
                    'dataSource': 'Cost Basis (Error fetching price)'
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

@app.route('/api/data/stats', methods=['GET'])
def get_data_statistics():
    """Get comprehensive data statistics and database information"""
    try:
        stats = multi_source_data_manager.get_data_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Error getting data statistics: {str(e)}'}), 500

@app.route('/api/data/cleanup', methods=['POST'])
def cleanup_old_data():
    """Clean up old data from database"""
    try:
        multi_source_data_manager.cleanup_old_data()
        return jsonify({'message': 'Database cleanup completed successfully'})
    except Exception as e:
        return jsonify({'error': f'Error during cleanup: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def comprehensive_health_check():
    """Comprehensive health check for all services"""
    try:
        health = multi_source_data_manager.health_check()
        status_code = 200 if health['overall_status'] == 'healthy' else 503
        return jsonify(health), status_code
    except Exception as e:
        return jsonify({'error': f'Health check failed: {str(e)}'}), 500

@app.route('/api/news/<symbol>', methods=['GET'])
def get_stock_news(symbol):
    """Get news data for a stock from multiple sources"""
    symbol = symbol.upper().strip()
    limit = int(request.args.get('limit', 10))
    
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    try:
        news_data = multi_source_data_manager.get_news_data(symbol, limit)
        return jsonify({
            'symbol': symbol,
            'news': news_data,
            'count': len(news_data),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Error fetching news: {str(e)}'}), 500

@app.route('/api/models/status', methods=['GET'])
def get_models_status():
    """Get status of ML models"""
    try:
        status = {
            'ensemble_predictor': ensemble_predictor.get_ensemble_model_info(),
            'arima_predictor': arima_predictor.get_arima_model_info(),
            'trained_symbols': list(trained_symbols.keys()),
            'total_symbols': len(trained_symbols),
            'symbol_details': trained_symbols
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f'Error getting model status: {str(e)}'}), 500

@app.route('/api/models/train/<symbol>', methods=['POST'])
def train_models_for_symbol(symbol):
    """Manually train ML models for a specific symbol using Alpha Vantage data"""
    symbol = symbol.upper()
    
    try:
        # Get historical data from Alpha Vantage
        historical_data = get_historical_data_from_alpha_vantage(symbol, outputsize='full')
        if not historical_data:
            return jsonify({'error': 'Unable to fetch historical data for training'}), 500
        
        prices = historical_data['prices']
        volumes = historical_data['volumes']
        
        if len(prices) < 100:
            return jsonify({'error': f'Insufficient historical data for training (need at least 100 data points, got {len(prices)})'}), 400
        
        # Train Ensemble models
        ensemble_success, ensemble_message = ensemble_predictor.train_ensemble_models(prices, volumes)
        
        # Train ARIMA model
        arima_success, arima_message = arima_predictor.train_arima_model(prices)
        
        # Update training status
        trained_symbols[symbol] = {
            'ensemble_trained': ensemble_success,
            'ensemble_message': ensemble_message,
            'arima_trained': arima_success,
            'arima_message': arima_message,
            'training_samples': len(prices),
            'last_trained': datetime.now().isoformat()
        }
        
        result = {
            'symbol': symbol,
            'data_source': 'Alpha Vantage',
            'training_data': {
                'samples': len(prices),
                'date_range': f"{historical_data['dates'][-1]} to {historical_data['dates'][0]}"
            },
            'ensemble_training': {
                'success': ensemble_success,
                'message': ensemble_message,
                'algorithm': 'Linear Regression + Random Forest Ensemble'
            },
            'arima_training': {
                'success': arima_success,
                'message': arima_message,
                'algorithm': 'ARIMA Time Series'
            },
            'models_ready': ensemble_success and arima_success
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error training models: {str(e)}'}), 500

# Real-time stock data is now available via Alpha Vantage API
# Use /api/stock/price/<symbol> endpoint for live prices
# Set ALPHA_VANTAGE_API_KEY in your .env file

# All ML algorithms and data processing now use real Alpha Vantage data
# Mock data has been completely removed from the system

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
