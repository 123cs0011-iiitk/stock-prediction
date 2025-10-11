#!/usr/bin/env python3
"""
Technical Indicators Calculation Script
Calculates various technical indicators from price and volume data
"""

import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict

def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return [np.mean(prices)] * len(prices)
    
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(np.mean(prices[:i+1]))
        else:
            sma.append(np.mean(prices[i-period+1:i+1]))
    return sma

def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return [np.mean(prices)] * len(prices)
    
    ema = []
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    ema.append(np.mean(prices[:period]))
    
    for i in range(period, len(prices)):
        ema_value = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_value)
    
    return ema

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return [50] * len(prices)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = [50]  # First value is neutral
    
    for i in range(period, len(gains)):
        avg_gain = np.mean(gains[i-period:i])
        avg_loss = np.mean(losses[i-period:i])
        
        if avg_loss == 0:
            rsi_value = 100
        else:
            rs = avg_gain / avg_loss
            rsi_value = 100 - (100 / (1 + rs))
        
        rsi.append(rsi_value)
    
    return rsi

def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, period)
    
    upper_band = []
    lower_band = []
    
    for i in range(len(prices)):
        if i < period - 1:
            # Use available data
            start_idx = max(0, i - period + 1)
            period_prices = prices[start_idx:i+1]
            std = np.std(period_prices)
        else:
            period_prices = prices[i-period+1:i+1]
            std = np.std(period_prices)
        
        upper_band.append(sma[i] + (std * std_dev))
        lower_band.append(sma[i] - (std * std_dev))
    
    return {
        'upper_band': upper_band,
        'middle_band': sma,
        'lower_band': lower_band
    }

def calculate_macd(prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, List[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    
    macd_line = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal_period)
    
    histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }

def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, List[float]]:
    """Calculate Stochastic Oscillator"""
    k_percent = []
    d_percent = []
    
    for i in range(len(closes)):
        if i < period - 1:
            start_idx = 0
        else:
            start_idx = i - period + 1
        
        period_highs = highs[start_idx:i+1]
        period_lows = lows[start_idx:i+1]
        
        highest_high = max(period_highs)
        lowest_low = min(period_lows)
        
        if highest_high == lowest_low:
            k_value = 50
        else:
            k_value = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        k_percent.append(k_value)
    
    # D% is 3-period SMA of K%
    d_percent = calculate_sma(k_percent, 3)
    
    return {
        'k_percent': k_percent,
        'd_percent': d_percent
    }

def calculate_volume_indicators(prices: List[float], volumes: List[float]) -> Dict[str, List[float]]:
    """Calculate volume-based indicators"""
    # Volume SMA
    volume_sma = calculate_sma(volumes, 20)
    
    # On-Balance Volume (OBV)
    obv = [volumes[0]]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif prices[i] < prices[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    # Volume Price Trend (VPT)
    vpt = [volumes[0] * prices[0]]
    for i in range(1, len(prices)):
        price_change = (prices[i] - prices[i-1]) / prices[i-1]
        vpt.append(vpt[-1] + (volumes[i] * price_change))
    
    return {
        'volume_sma': volume_sma,
        'obv': obv,
        'vpt': vpt
    }

def calculate_volatility(prices: List[float], period: int = 20) -> List[float]:
    """Calculate price volatility"""
    volatility = []
    
    for i in range(len(prices)):
        if i < period - 1:
            start_idx = 0
        else:
            start_idx = i - period + 1
        
        period_prices = prices[start_idx:i+1]
        returns = np.diff(period_prices) / period_prices[:-1]
        vol = np.std(returns) * np.sqrt(252) * 100  # Annualized volatility in %
        volatility.append(vol)
    
    return volatility

def main():
    """Main execution function"""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        prices = input_data.get('prices', [])
        volumes = input_data.get('volumes', [])
        
        if len(prices) < 20:
            result = {
                'success': False,
                'error': 'Insufficient data for technical indicators (need at least 20 data points)',
                'data_points': len(prices)
            }
        else:
            # For stochastic calculation, we need highs and lows
            # If not provided, estimate from prices
            highs = input_data.get('highs', prices)
            lows = input_data.get('lows', prices)
            
            # Calculate all technical indicators
            indicators = {
                'sma_20': calculate_sma(prices, 20),
                'sma_50': calculate_sma(prices, 50),
                'ema_12': calculate_ema(prices, 12),
                'ema_26': calculate_ema(prices, 26),
                'rsi': calculate_rsi(prices, 14),
                'bollinger_bands': calculate_bollinger_bands(prices, 20, 2),
                'macd': calculate_macd(prices, 12, 26, 9),
                'stochastic': calculate_stochastic(highs, lows, prices, 14),
                'volume_indicators': calculate_volume_indicators(prices, volumes),
                'volatility': calculate_volatility(prices, 20)
            }
            
            # Get latest values for summary
            latest_indicators = {
                'sma_20': indicators['sma_20'][-1] if indicators['sma_20'] else 0,
                'sma_50': indicators['sma_50'][-1] if indicators['sma_50'] else 0,
                'rsi': indicators['rsi'][-1] if indicators['rsi'] else 50,
                'volatility': indicators['volatility'][-1] if indicators['volatility'] else 0,
                'current_price': prices[-1]
            }
            
            # Generate trading signals
            signals = generate_trading_signals(latest_indicators)
            
            result = {
                'success': True,
                'data_points': len(prices),
                'indicators': indicators,
                'latest_indicators': latest_indicators,
                'signals': signals,
                'timestamp': input_data.get('timestamp')
            }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result))

def generate_trading_signals(indicators: Dict) -> Dict[str, str]:
    """Generate basic trading signals based on indicators"""
    signals = {}
    
    # RSI signals
    rsi = indicators.get('rsi', 50)
    if rsi > 70:
        signals['rsi'] = 'overbought'
    elif rsi < 30:
        signals['rsi'] = 'oversold'
    else:
        signals['rsi'] = 'neutral'
    
    # Moving average signals
    sma_20 = indicators.get('sma_20', 0)
    sma_50 = indicators.get('sma_50', 0)
    current_price = indicators.get('current_price', 0)
    
    if current_price > sma_20 > sma_50:
        signals['trend'] = 'bullish'
    elif current_price < sma_20 < sma_50:
        signals['trend'] = 'bearish'
    else:
        signals['trend'] = 'neutral'
    
    # Volatility signals
    volatility = indicators.get('volatility', 0)
    if volatility > 30:
        signals['volatility'] = 'high'
    elif volatility < 15:
        signals['volatility'] = 'low'
    else:
        signals['volatility'] = 'normal'
    
    return signals

if __name__ == '__main__':
    main()
