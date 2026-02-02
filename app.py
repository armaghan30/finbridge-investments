from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.utils
import json
from datetime import datetime, timedelta
import csv
import os
import time
import random
from scipy.stats import norm
try:
    import talib
except ImportError:
    # Fallback if TA-Lib is not available
    talib = None
    print("Warning: TA-Lib not available. Technical indicators will be limited.")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fast_data_service import fast_data_service

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# Market configurations
MARKETS = {
    'NYSE': {
        'name': 'New York Stock Exchange',
        'stocks': {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corp.',
            'GOOGL': 'Alphabet Inc.',
            'TSLA': 'Tesla Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corp.',
            'NFLX': 'Netflix Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'PG': 'Procter & Gamble Co.',
            'HD': 'Home Depot Inc.',
            'MA': 'Mastercard Inc.',
            'DIS': 'Walt Disney Co.',
            'PYPL': 'PayPal Holdings Inc.',
            'ADBE': 'Adobe Inc.',
            'CRM': 'Salesforce Inc.',
            'NKE': 'Nike Inc.',
            'INTC': 'Intel Corp.',
            'WMT': 'Walmart Inc.',
            'BAC': 'Bank of America Corp.',
            'KO': 'Coca-Cola Co.',
            'PFE': 'Pfizer Inc.',
            'T': 'AT&T Inc.',
            'VZ': 'Verizon Communications Inc.',
            'CMCSA': 'Comcast Corp.',
            'ABT': 'Abbott Laboratories',
            'LLY': 'Eli Lilly and Co.',
            'UNH': 'UnitedHealth Group Inc.'
        }
    },
    'PSX': {
        'name': 'Pakistan Stock Exchange',
        'stocks': {
            'OGDC': 'Oil & Gas Development Co.',
            'PPL': 'Pakistan Petroleum Ltd.',
            'MCB': 'Muslim Commercial Bank',
            'UBL': 'United Bank Ltd.',
            'HBL': 'Habib Bank Ltd.',
            'LUCK': 'Lucky Cement Ltd.',
            'ENGRO': 'Engro Corp.',
            'FFC': 'Fauji Fertilizer Co.',
            'EFERT': 'Engro Fertilizers Ltd.',
            'ATRL': 'Attock Refinery Ltd.',
            'PSO': 'Pakistan State Oil',
            'SHEL': 'Shell Pakistan Ltd.',
            'NESTLE': 'Nestle Pakistan Ltd.',
            'UNILEVER': 'Unilever Pakistan Ltd.',
            'COLGATE': 'Colgate Palmolive Pakistan',
            'PAKT': 'Pakistan Tobacco Co.',
            'ICI': 'ICI Pakistan Ltd.',
            'FFBL': 'Fauji Fertilizer Bin Qasim',
            'DCL': 'Dewan Cement Ltd.',
            'DGKC': 'D.G. Khan Cement Co.'
        }
    }
}

# Cache for storing data to avoid repeated API calls
data_cache = {}
cache_duration = 300  # 5 minutes

# Debug mode - set to False to reduce console output
DEBUG_MODE = False

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'SMA': {'name': 'Simple Moving Average', 'params': [20, 50, 200]},
    'EMA': {'name': 'Exponential Moving Average', 'params': [12, 26, 50]},
    'RSI': {'name': 'Relative Strength Index', 'params': [14]},
    'MACD': {'name': 'MACD', 'params': [12, 26, 9]},
    'BB': {'name': 'Bollinger Bands', 'params': [20, 2]},
    'STOCH': {'name': 'Stochastic Oscillator', 'params': [14, 3, 3]},
    'ATR': {'name': 'Average True Range', 'params': [14]},
    'ADX': {'name': 'Average Directional Index', 'params': [14]},
    'CCI': {'name': 'Commodity Channel Index', 'params': [14]},
    'WILLR': {'name': 'Williams %R', 'params': [14]},
    'ROC': {'name': 'Rate of Change', 'params': [10]},
    'OBV': {'name': 'On Balance Volume', 'params': []},
    'VWAP': {'name': 'Volume Weighted Average Price', 'params': []}
}

# Market screening criteria
SCREENING_CRITERIA = {
    'price': {
        'min': 0,
        'max': 10000,
        'current': {'min': 0, 'max': 1000}
    },
    'market_cap': {
        'micro': {'min': 0, 'max': 300000000},
        'small': {'min': 300000000, 'max': 2000000000},
        'mid': {'min': 2000000000, 'max': 10000000000},
        'large': {'min': 10000000000, 'max': 100000000000},
        'mega': {'min': 100000000000, 'max': float('inf')}
    },
    'pe_ratio': {
        'min': 0,
        'max': 100,
        'current': {'min': 0, 'max': 50}
    },
    'dividend_yield': {
        'min': 0,
        'max': 20,
        'current': {'min': 0, 'max': 10}
    },
    'beta': {
        'min': 0,
        'max': 3,
        'current': {'min': 0, 'max': 2}
    },
    'volume': {
        'min': 0,
        'max': float('inf'),
        'current': {'min': 100000, 'max': 10000000}
    }
}

def get_cached_data(key):
    """Get data from cache if it exists and is not expired"""
    if key in data_cache:
        timestamp, data = data_cache[key]
        if time.time() - timestamp < cache_duration:
            return data
    return None

def set_cached_data(key, data):
    """Store data in cache with timestamp"""
    data_cache[key] = (time.time(), data)

def fetch_stock_data(ticker, period='7d', interval='1h', start_date=None, end_date=None):
    """
    Fetch stock data using yfinance with enhanced error handling and caching
    """
    # Create cache key including date range
    date_suffix = f"_{start_date}_{end_date}" if start_date and end_date else ""
    cache_key = f"stock_data_{ticker}_{period}_{interval}{date_suffix}"
    cached_data = get_cached_data(cache_key)
    if cached_data:
        if DEBUG_MODE:
            print(f"Using cached data for {ticker}")
        return cached_data
    
    try:
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5))
        
        # For PSX stocks, try different suffixes
        if ticker in MARKETS['PSX']['stocks']:
            # Try different PSX suffixes
            possible_tickers = [
                f"{ticker}.PSX",
                f"{ticker}.PK",
                ticker,
                f"{ticker}.IS"
            ]
            
            for test_ticker in possible_tickers:
                try:
                    stock = yf.Ticker(test_ticker)
                    if start_date and end_date:
                        data = stock.history(start=start_date, end=end_date, interval=interval)
                    else:
                        data = stock.history(period=period, interval=interval)
                    
                    if hasattr(data, 'empty') and not data.empty and len(data) > 5:  # Ensure we have meaningful data
                        if DEBUG_MODE:
                            print(f"Successfully fetched data for {test_ticker}")
                        set_cached_data(cache_key, data)
                        return data
                except Exception as e:
                    if DEBUG_MODE:
                        print(f"Failed to fetch {test_ticker}: {e}")
                    time.sleep(0.5)  # Small delay between attempts
                    continue
            
            # If no PSX data available, return sample data for demonstration
            if DEBUG_MODE:
                print(f"No data available for {ticker}, generating sample data")
            sample_data = generate_sample_data(ticker, period, interval, start_date, end_date)
            set_cached_data(cache_key, sample_data)
            return sample_data
        else:
            # For NYSE stocks
            stock = yf.Ticker(ticker)
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                data = stock.history(period=period, interval=interval)
            
            if hasattr(data, 'empty') and (data.empty or len(data) < 5):
                if DEBUG_MODE:
                    print(f"No data available for {ticker}, generating sample data")
                sample_data = generate_sample_data(ticker, period, interval, start_date, end_date)
                set_cached_data(cache_key, sample_data)
                return sample_data
            set_cached_data(cache_key, data)
            return data
            
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error fetching data for {ticker}: {e}")
        sample_data = generate_sample_data(ticker, period, interval, start_date, end_date)
        set_cached_data(cache_key, sample_data)
        return sample_data

def generate_sample_data(ticker, period='7d', interval='1h', start_date=None, end_date=None):
    """
    Generate sample data for demonstration when real data is not available
    """
    try:
        # Use provided date range or default to last 7 days
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
        
        # Create timestamps based on interval
        if interval == '1h':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
        elif interval == '1d':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
        
        # Generate realistic price movements based on ticker
        if ticker in MARKETS['NYSE']['stocks']:
            base_price = 150.0  # Higher base price for NYSE stocks
            currency = 'USD'
        else:
            base_price = 50.0   # Lower base price for PSX stocks
            currency = 'PKR'
        
        prices = []
        current_price = base_price
        
        for i in range(len(timestamps)):
            # Add some random movement with trend
            trend = np.random.normal(0, 0.3)  # Small trend
            volatility = np.random.normal(0, 0.8)  # Volatility
            change = trend + volatility
            current_price += change
            current_price = max(current_price, 1.0)  # Ensure price doesn't go negative
            prices.append(current_price)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Open': prices,
            'High': [p + abs(np.random.normal(0, 0.5)) for p in prices],
            'Low': [max(p - abs(np.random.normal(0, 0.5)), 0.1) for p in prices],
            'Close': prices,
            'Volume': [int(np.random.uniform(100000, 1000000)) for _ in prices]
        }, index=timestamps)
        
        # Add currency info to the DataFrame
        data.attrs['currency'] = currency
        
        return data
        
    except Exception as e:
        print(f"Error generating sample data: {e}")
        return None

def calculate_capm(beta, market_return, risk_free_rate):
    """
    Calculate expected return using CAPM model
    Formula: E(Ri) = Rf + βi(Rm - Rf)
    """
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return expected_return

def run_monte_carlo_simulation(start_price, mean_return, volatility, time_horizon, num_simulations):
    """
    Run Monte Carlo simulation using geometric Brownian motion
    """
    dt = 1/252  # Daily time step
    simulations = []
    
    for _ in range(num_simulations):
        price_path = [start_price]
        for _ in range(time_horizon):
            # Geometric Brownian motion
            drift = (mean_return - 0.5 * volatility**2) * dt
            diffusion = volatility * np.sqrt(dt) * np.random.normal()
            new_price = price_path[-1] * np.exp(drift + diffusion)
            price_path.append(new_price)
        simulations.append(price_path)
    
    return simulations

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate call option price using Black-Scholes model
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate put option price using Black-Scholes model
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price

def get_financial_ratios(ticker):
    """
    Get financial ratios for a stock with caching
    """
    cache_key = f"ratios_{ticker}"
    cached_ratios = get_cached_data(cache_key)
    if cached_ratios:
        return cached_ratios
    
    try:
        time.sleep(random.uniform(0.3, 0.8))  # Small delay to avoid rate limiting
        
        # For PSX stocks, try different suffixes
        if ticker in MARKETS['PSX']['stocks']:
            possible_tickers = [
                f"{ticker}.PSX",
                f"{ticker}.PK",
                ticker,
                f"{ticker}.IS"
            ]
            
            info = None
            for test_ticker in possible_tickers:
                try:
                    stock = yf.Ticker(test_ticker)
                    info = stock.info
                    if info and len(info) > 5:  # Check if we got meaningful data
                        break
                except:
                    continue
        else:
            stock = yf.Ticker(ticker)
            info = stock.info
        
        # Generate sample ratios if no real data available
        if not info or len(info) < 5:
            ratios = {
                'pe_ratio': round(np.random.uniform(10, 30), 2),
                'pb_ratio': round(np.random.uniform(1, 5), 2),
                'debt_equity': round(np.random.uniform(0.1, 2.0), 2),
                'roe': round(np.random.uniform(5, 25), 2),
                'market_cap': np.random.uniform(1e9, 1e12),
                'dividend_yield': round(np.random.uniform(0, 5), 2),
                'beta': round(np.random.uniform(0.5, 2.0), 2),
                'eps': round(np.random.uniform(1, 10), 2)
            }
        else:
            ratios = {
                'pe_ratio': info.get('trailingPE', round(np.random.uniform(10, 30), 2)),
                'pb_ratio': info.get('priceToBook', round(np.random.uniform(1, 5), 2)),
                'debt_equity': info.get('debtToEquity', round(np.random.uniform(0.1, 2.0), 2)),
                'roe': info.get('returnOnEquity', round(np.random.uniform(5, 25), 2)) * 100 if info.get('returnOnEquity') else round(np.random.uniform(5, 25), 2),
                'market_cap': info.get('marketCap', np.random.uniform(1e9, 1e12)),
                'dividend_yield': info.get('dividendYield', round(np.random.uniform(0, 5), 2)) * 100 if info.get('dividendYield') else round(np.random.uniform(0, 5), 2),
                'beta': info.get('beta', round(np.random.uniform(0.5, 2.0), 2)),
                'eps': info.get('trailingEps', round(np.random.uniform(1, 10), 2))
            }
        
        set_cached_data(cache_key, ratios)
        return ratios
    except Exception as e:
        print(f"Error fetching ratios for {ticker}: {e}")
        # Return sample ratios
        ratios = {
            'pe_ratio': round(np.random.uniform(10, 30), 2),
            'pb_ratio': round(np.random.uniform(1, 5), 2),
            'debt_equity': round(np.random.uniform(0.1, 2.0), 2),
            'roe': round(np.random.uniform(5, 25), 2),
            'market_cap': np.random.uniform(1e9, 1e12),
            'dividend_yield': round(np.random.uniform(0, 5), 2),
            'beta': round(np.random.uniform(0.5, 2.0), 2),
            'eps': round(np.random.uniform(1, 10), 2)
        }
        set_cached_data(cache_key, ratios)
        return ratios

def calculate_technical_indicators(data):
    """Calculate technical indicators for stock data"""
    indicators = {}
    
    try:
        if talib is None:
            # Fallback calculations without TA-Lib
            return calculate_basic_indicators(data)
        
        # Ensure data is numeric and convert to numpy arrays
        high = np.array(data['High'].astype(float))
        low = np.array(data['Low'].astype(float))
        close = np.array(data['Close'].astype(float))
        volume = np.array(data['Volume'].astype(float))
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(high) | np.isnan(low) | np.isnan(close) | np.isnan(volume))
        if not np.any(valid_mask) or np.sum(valid_mask) < 20:
            return calculate_basic_indicators(data)
        
        high = high[valid_mask]
        low = low[valid_mask]
        close = close[valid_mask]
        volume = volume[valid_mask]
        
        # Simple Moving Averages
        indicators['SMA_20'] = talib.SMA(close, timeperiod=min(20, len(close)-1))
        indicators['SMA_50'] = talib.SMA(close, timeperiod=min(50, len(close)-1))
        indicators['SMA_200'] = talib.SMA(close, timeperiod=min(200, len(close)-1))
        
        # Exponential Moving Averages
        indicators['EMA_12'] = talib.EMA(close, timeperiod=min(12, len(close)-1))
        indicators['EMA_26'] = talib.EMA(close, timeperiod=min(26, len(close)-1))
        indicators['EMA_50'] = talib.EMA(close, timeperiod=min(50, len(close)-1))
        
        # RSI
        indicators['RSI'] = talib.RSI(close, timeperiod=min(14, len(close)-1))
        
        # MACD
        if len(close) >= 26:
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['MACD'] = macd
            indicators['MACD_SIGNAL'] = macd_signal
            indicators['MACD_HIST'] = macd_hist
        
        # Bollinger Bands
        if len(close) >= 20:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['BB_UPPER'] = bb_upper
            indicators['BB_MIDDLE'] = bb_middle
            indicators['BB_LOWER'] = bb_lower
        
        # Stochastic Oscillator
        if len(close) >= 14:
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            indicators['STOCH_K'] = slowk
            indicators['STOCH_D'] = slowd
        
        # Average True Range
        if len(close) >= 14:
            indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Average Directional Index
        if len(close) >= 14:
            indicators['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Commodity Channel Index
        if len(close) >= 14:
            indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Williams %R
        if len(close) >= 14:
            indicators['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Rate of Change
        if len(close) >= 10:
            indicators['ROC'] = talib.ROC(close, timeperiod=10)
        
        # On Balance Volume
        indicators['OBV'] = talib.OBV(close, volume)
        
        # VWAP (Volume Weighted Average Price)
        vwap = np.cumsum(close * volume) / np.cumsum(volume)
        indicators['VWAP'] = vwap
        
        # Additional custom indicators
        if len(close) >= 10:
            # Price Rate of Change
            indicators['PRICE_ROC'] = ((close - np.roll(close, 10)) / np.roll(close, 10)) * 100
            
            # Volume Rate of Change
            indicators['VOLUME_ROC'] = ((volume - np.roll(volume, 10)) / np.roll(volume, 10)) * 100
        
        # Money Flow Index
        if len(close) >= 14:
            indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # Parabolic SAR
        indicators['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        # Return basic indicators if calculation fails
        indicators = calculate_basic_indicators(data)
    
    return indicators

def calculate_basic_indicators(data):
    """Calculate basic indicators without TA-Lib"""
    indicators = {}
    
    try:
        close = np.array(data['Close'].astype(float))
        high = np.array(data['High'].astype(float))
        low = np.array(data['Low'].astype(float))
        volume = np.array(data['Volume'].astype(float))
        
        # Remove NaN values
        valid_mask = ~(np.isnan(close) | np.isnan(high) | np.isnan(low) | np.isnan(volume))
        if not np.any(valid_mask):
            return indicators
        
        close = close[valid_mask]
        high = high[valid_mask]
        low = low[valid_mask]
        volume = volume[valid_mask]
        
        # Simple Moving Averages
        if len(close) >= 20:
            indicators['SMA_20'] = np.convolve(close, np.ones(20)/20, mode='valid')
        if len(close) >= 50:
            indicators['SMA_50'] = np.convolve(close, np.ones(50)/50, mode='valid')
        
        # RSI (simplified)
        if len(close) >= 14:
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
            avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
            
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            indicators['RSI'] = rsi
        
        # VWAP
        vwap = np.cumsum(close * volume) / np.cumsum(volume)
        indicators['VWAP'] = vwap
        
    except Exception as e:
        print(f"Error calculating basic indicators: {e}")
    
    return indicators

def calculate_support_resistance(data, window=20):
    """Calculate support and resistance levels"""
    try:
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(high) - window):
            # Resistance (local maxima)
            if high[i] == max(high[i-window:i+window+1]):
                resistance_levels.append(high[i])
            
            # Support (local minima)
            if low[i] == min(low[i-window:i+window+1]):
                support_levels.append(low[i])
        
        # Get current levels (last 5 levels)
        current_resistance = sorted(set(resistance_levels))[-5:] if resistance_levels else []
        current_support = sorted(set(support_levels))[:5] if support_levels else []
        
        return {
            'resistance': current_resistance,
            'support': current_support,
            'current_price': close[-1] if len(close) > 0 else 0
        }
    except Exception as e:
        print(f"Error calculating support/resistance: {e}")
        return {'resistance': [], 'support': [], 'current_price': 0}

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    try:
        high = data['High'].max()
        low = data['Low'].min()
        current_price = data['Close'].iloc[-1]
        
        diff = high - low
        
        levels = {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
        
        # Determine current position
        current_position = None
        for level, price in levels.items():
            if abs(current_price - price) / price < 0.02:  # Within 2%
                current_position = level
                break
        
        return {
            'levels': levels,
            'current_position': current_position,
            'high': high,
            'low': low,
            'current_price': current_price
        }
    except Exception as e:
        print(f"Error calculating Fibonacci levels: {e}")
        return {'levels': {}, 'current_position': None, 'high': 0, 'low': 0, 'current_price': 0}

def calculate_risk_metrics(data):
    """Calculate comprehensive risk metrics"""
    try:
        returns = data['Close'].pct_change().dropna()
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized volatility
            'var_95': np.percentile(returns, 5) * 100,  # 95% VaR
            'var_99': np.percentile(returns, 1) * 100,  # 99% VaR
            'max_drawdown': calculate_max_drawdown(data['Close']),
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'sortino_ratio': calculate_sortino_ratio(returns),
            'calmar_ratio': calculate_calmar_ratio(returns, data['Close']),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'beta': 1.0,  # Will be calculated with market data
            'correlation': 0.5  # Will be calculated with market data
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return {}

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    try:
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max * 100).min()
        return abs(drawdown)
    except:
        return 0

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    try:
        excess_return = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_return / volatility if volatility > 0 else 0
    except:
        return 0

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino ratio"""
    try:
        excess_return = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
        return excess_return / downside_deviation if downside_deviation > 0 else 0
    except:
        return 0

def calculate_calmar_ratio(returns, prices, risk_free_rate=0.02):
    """Calculate Calmar ratio"""
    try:
        excess_return = returns.mean() * 252 - risk_free_rate
        max_dd = calculate_max_drawdown(prices)
        return excess_return / max_dd if max_dd > 0 else 0
    except:
        return 0

def screen_market(criteria):
    """Screen market based on criteria"""
    screened_stocks = []
    
    try:
        for market_name, market_data in MARKETS.items():
            for ticker, company_name in market_data['stocks'].items():
                try:
                    # Get stock data and ratios
                    stock_data = fetch_stock_data(ticker, period='30d')
                    ratios = get_financial_ratios(ticker)
                    
                    if stock_data is None or stock_data.empty:
                        continue
                    
                    current_price = stock_data['Close'].iloc[-1]
                    volume = stock_data['Volume'].iloc[-1]
                    
                    # Apply screening criteria
                    passes_screen = True
                    
                    # Price filter
                    if not (criteria.get('price_min', 0) <= current_price <= criteria.get('price_max', float('inf'))):
                        passes_screen = False
                    
                    # Market cap filter
                    market_cap = ratios.get('market_cap', 0)
                    if criteria.get('market_cap_min', 0) > market_cap or market_cap > criteria.get('market_cap_max', float('inf')):
                        passes_screen = False
                    
                    # P/E ratio filter
                    pe_ratio = ratios.get('pe_ratio', 0)
                    if criteria.get('pe_min', 0) > pe_ratio or pe_ratio > criteria.get('pe_max', float('inf')):
                        passes_screen = False
                    
                    # Dividend yield filter
                    dividend_yield = ratios.get('dividend_yield', 0)
                    if criteria.get('dividend_min', 0) > dividend_yield or dividend_yield > criteria.get('dividend_max', float('inf')):
                        passes_screen = False
                    
                    # Beta filter
                    beta = ratios.get('beta', 1.0)
                    if criteria.get('beta_min', 0) > beta or beta > criteria.get('beta_max', float('inf')):
                        passes_screen = False
                    
                    # Volume filter
                    if criteria.get('volume_min', 0) > volume or volume > criteria.get('volume_max', float('inf')):
                        passes_screen = False
                    
                    if passes_screen:
                        screened_stocks.append({
                            'ticker': ticker,
                            'company_name': company_name,
                            'market': market_name,
                            'current_price': current_price,
                            'volume': volume,
                            'market_cap': market_cap,
                            'pe_ratio': pe_ratio,
                            'dividend_yield': dividend_yield,
                            'beta': beta,
                            'change_percent': ((current_price - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100) if len(stock_data) > 1 else 0
                        })
                
                except Exception as e:
                    print(f"Error screening {ticker}: {e}")
                    continue
        
        # Sort by specified criteria
        sort_by = criteria.get('sort_by', 'market_cap')
        sort_order = criteria.get('sort_order', 'desc')
        
        if sort_by in ['market_cap', 'current_price', 'volume', 'pe_ratio', 'dividend_yield', 'beta', 'change_percent']:
            screened_stocks.sort(key=lambda x: x[sort_by], reverse=(sort_order == 'desc'))
        
        return screened_stocks[:criteria.get('limit', 50)]
    
    except Exception as e:
        print(f"Error in market screening: {e}")
        return []

def analyze_stock_sentiment(ticker):
    """Analyze stock sentiment based on technical indicators"""
    try:
        data = fetch_stock_data(ticker, period='60d')
        if data is None or (hasattr(data, 'empty') and data.empty):
            return {'sentiment': 'neutral', 'score': 0, 'signals': []}
        
        indicators = calculate_technical_indicators(data)
        
        signals = []
        score = 0
        
        # RSI signals
        if 'RSI' in indicators and len(indicators['RSI']) > 0:
            rsi = indicators['RSI'][-1]
            if not np.isnan(rsi):
                if rsi < 30:
                    signals.append('RSI oversold (bullish)')
                    score += 1
                elif rsi > 70:
                    signals.append('RSI overbought (bearish)')
                    score -= 1
        
        # MACD signals
        if 'MACD' in indicators and 'MACD_SIGNAL' in indicators and len(indicators['MACD']) > 0:
            macd = indicators['MACD'][-1]
            macd_signal = indicators['MACD_SIGNAL'][-1]
            if not np.isnan(macd) and not np.isnan(macd_signal):
                if macd > macd_signal:
                    signals.append('MACD bullish crossover')
                    score += 1
                elif macd < macd_signal:
                    signals.append('MACD bearish crossover')
                    score -= 1
        
        # Bollinger Bands signals
        if 'BB_UPPER' in indicators and 'BB_LOWER' in indicators and len(indicators['BB_UPPER']) > 0:
            current_price = data['Close'].iloc[-1]
            bb_upper = indicators['BB_UPPER'][-1]
            bb_lower = indicators['BB_LOWER'][-1]
            
            if not np.isnan(bb_upper) and not np.isnan(bb_lower):
                if current_price < bb_lower:
                    signals.append('Price below lower Bollinger Band (bullish)')
                    score += 1
                elif current_price > bb_upper:
                    signals.append('Price above upper Bollinger Band (bearish)')
                    score -= 1
        
        # Moving Average signals
        if 'SMA_20' in indicators and 'SMA_50' in indicators and len(indicators['SMA_20']) > 0:
            sma_20 = indicators['SMA_20'][-1]
            sma_50 = indicators['SMA_50'][-1]
            current_price = data['Close'].iloc[-1]
            
            if not np.isnan(sma_20) and not np.isnan(sma_50):
                if current_price > sma_20 > sma_50:
                    signals.append('Price above moving averages (bullish)')
                    score += 1
                elif current_price < sma_20 < sma_50:
                    signals.append('Price below moving averages (bearish)')
                    score -= 1
        
        # Volume analysis
        try:
            volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            if not np.isnan(volume_avg) and not np.isnan(current_volume):
                if current_volume > volume_avg * 1.5:
                    signals.append('High volume (potential breakout)')
                    score += 0.5
        except:
            pass
        
        # Determine sentiment
        if score >= 2:
            sentiment = 'bullish'
        elif score <= -2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': score,
            'signals': signals,
            'confidence': min(abs(score) / 3, 1.0)  # Confidence level 0-1
        }
    
    except Exception as e:
        print(f"Error analyzing sentiment for {ticker}: {e}")
        return {'sentiment': 'neutral', 'score': 0, 'signals': []}

def generate_stock_report(ticker, market='NYSE'):
    """Generate comprehensive stock analysis report"""
    try:
        # Fetch data
        data = fetch_stock_data(ticker, period='1y')
        if data is None or (hasattr(data, 'empty') and data.empty):
            return None
            
        ratios = get_financial_ratios(ticker)
        indicators = calculate_technical_indicators(data)
        support_resistance = calculate_support_resistance(data)
        fibonacci = calculate_fibonacci_levels(data)
        risk_metrics = calculate_risk_metrics(data)
        sentiment = analyze_stock_sentiment(ticker)
        
        # Calculate additional metrics
        current_price = data['Close'].iloc[-1]
        price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
        
        # Performance metrics
        returns = data['Close'].pct_change().dropna()
        annual_return = returns.mean() * 252 * 100 if len(returns) > 0 else 0
        annual_volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        # Moving averages
        sma_20 = indicators.get('SMA_20', [0])[-1] if 'SMA_20' in indicators and len(indicators['SMA_20']) > 0 else 0
        sma_50 = indicators.get('SMA_50', [0])[-1] if 'SMA_50' in indicators and len(indicators['SMA_50']) > 0 else 0
        sma_200 = indicators.get('SMA_200', [0])[-1] if 'SMA_200' in indicators and len(indicators['SMA_200']) > 0 else 0
        
        # Handle NaN values
        sma_20 = 0 if np.isnan(sma_20) else sma_20
        sma_50 = 0 if np.isnan(sma_50) else sma_50
        sma_200 = 0 if np.isnan(sma_200) else sma_200
        
        # Volume analysis
        try:
            volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_signal = 'high' if not np.isnan(volume_avg) and not np.isnan(current_volume) and current_volume > volume_avg * 1.5 else 'normal'
        except:
            volume_signal = 'normal'
        
        report = {
            'ticker': ticker,
            'market': market,
            'current_price': current_price,
            'price_change': price_change,
            'volume': data['Volume'].iloc[-1],
            'high_52w': data['High'].max(),
            'low_52w': data['Low'].min(),
            
            # Technical Analysis
            'technical_indicators': {
                'rsi': indicators.get('RSI', [0])[-1] if 'RSI' in indicators and len(indicators['RSI']) > 0 else 0,
                'macd': indicators.get('MACD', [0])[-1] if 'MACD' in indicators and len(indicators['MACD']) > 0 else 0,
                'macd_signal': indicators.get('MACD_SIGNAL', [0])[-1] if 'MACD_SIGNAL' in indicators and len(indicators['MACD_SIGNAL']) > 0 else 0,
                'bb_upper': indicators.get('BB_UPPER', [0])[-1] if 'BB_UPPER' in indicators and len(indicators['BB_UPPER']) > 0 else 0,
                'bb_lower': indicators.get('BB_LOWER', [0])[-1] if 'BB_LOWER' in indicators and len(indicators['BB_LOWER']) > 0 else 0,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200
            },
            
            # Support and Resistance
            'support_resistance': support_resistance,
            'fibonacci_levels': fibonacci,
            
            # Risk Metrics
            'risk_metrics': risk_metrics,
            
            # Performance Metrics
            'performance': {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0)
            },
            
            # Fundamental Analysis
            'fundamentals': {
                'market_cap': ratios.get('market_cap', 0),
                'pe_ratio': ratios.get('pe_ratio', 0),
                'pb_ratio': ratios.get('pb_ratio', 0),
                'dividend_yield': ratios.get('dividend_yield', 0),
                'beta': ratios.get('beta', 1.0),
                'roe': ratios.get('roe', 0),
                'debt_equity': ratios.get('debt_equity', 0)
            },
            
            # Sentiment Analysis
            'sentiment': sentiment,
            
            # Trading Signals
            'signals': {
                'trend': 'bullish' if current_price > sma_20 > sma_50 else 'bearish' if current_price < sma_20 < sma_50 else 'neutral',
                'momentum': 'bullish' if ('RSI' in indicators and len(indicators['RSI']) > 0 and indicators['RSI'][-1] > 50) else 'bearish',
                'volume': volume_signal,
                'volatility': 'high' if annual_volatility > 30 else 'medium' if annual_volatility > 15 else 'low'
            },
            
            # Recommendations
            'recommendations': generate_recommendations(ticker, data, indicators, ratios, sentiment)
        }
        
        return report
    
    except Exception as e:
        print(f"Error generating report for {ticker}: {e}")
        return None

def generate_recommendations(ticker, data, indicators, ratios, sentiment):
    """Generate trading recommendations"""
    recommendations = []
    
    try:
        current_price = data['Close'].iloc[-1]
        rsi = indicators.get('RSI', [50])[-1] if 'RSI' in indicators else 50
        pe_ratio = ratios.get('pe_ratio', 20)
        dividend_yield = ratios.get('dividend_yield', 0)
        beta = ratios.get('beta', 1.0)
        
        # Technical recommendations
        if rsi < 30:
            recommendations.append({
                'type': 'technical',
                'action': 'buy',
                'reason': 'RSI indicates oversold conditions',
                'confidence': 'high'
            })
        elif rsi > 70:
            recommendations.append({
                'type': 'technical',
                'action': 'sell',
                'reason': 'RSI indicates overbought conditions',
                'confidence': 'high'
            })
        
        # Fundamental recommendations
        if pe_ratio < 15 and dividend_yield > 2:
            recommendations.append({
                'type': 'fundamental',
                'action': 'buy',
                'reason': 'Undervalued with good dividend yield',
                'confidence': 'medium'
            })
        elif pe_ratio > 30:
            recommendations.append({
                'type': 'fundamental',
                'action': 'sell',
                'reason': 'Potentially overvalued',
                'confidence': 'medium'
            })
        
        # Risk-based recommendations
        if beta > 1.5:
            recommendations.append({
                'type': 'risk',
                'action': 'caution',
                'reason': 'High beta - more volatile than market',
                'confidence': 'medium'
            })
        
        # Sentiment-based recommendations
        if sentiment['sentiment'] == 'bullish' and sentiment['confidence'] > 0.7:
            recommendations.append({
                'type': 'sentiment',
                'action': 'buy',
                'reason': 'Strong bullish sentiment signals',
                'confidence': 'medium'
            })
        
        return recommendations
    
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html', markets=MARKETS)

@app.route('/api/stock-data')
def get_stock_data():
    ticker = request.args.get('ticker', 'AAPL')
    market = request.args.get('market', 'NYSE')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    interval = request.args.get('interval', '1h')
    
    # Create cache key
    cache_key = f'stock_data_{ticker}_{start_date}_{end_date}_{interval}'
    
    # Check cache first
    if cache_key in data_cache:
        cache_time, cached_result = data_cache[cache_key]
        if time.time() - cache_time < cache_duration:
            return jsonify({
                'success': True,
                'data': cached_result['data'],
                'info': cached_result['info'],
                'cached': True
            })
    
    try:
        # Use fast data service for instant results
        data = fast_data_service.get_stock_data(ticker, period='1y', interval=interval, start_date=start_date, end_date=end_date)
        if data is None or (hasattr(data, 'empty') and data.empty):
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Get currency from data attributes or determine based on market
        currency = getattr(data, 'attrs', {}).get('currency', 'USD' if market == 'NYSE' else 'PKR')
        
        # Convert to list of dictionaries with timestamps
        stock_data = []
        for index, row in data.iterrows():
            stock_data.append({
                'timestamp': int(index.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        # Calculate price change
        if len(stock_data) >= 2:
            current_price = stock_data[-1]['close']
            previous_price = stock_data[-2]['close']
            change_percent = ((current_price - previous_price) / previous_price) * 100
        else:
            current_price = stock_data[-1]['close'] if stock_data else 0
            change_percent = 0
        
        # Use fast data service for stock info
        stock_info = fast_data_service.get_stock_info(ticker)
        
        # Cache the result
        result = {
            'data': stock_data,
            'info': stock_info
        }
        data_cache[cache_key] = (time.time(), result)
        
        return jsonify({
            'success': True,
            'data': stock_data,
            'info': stock_info,
            'cached': False
        })
        
    except Exception as e:
        print(f"Error in get_stock_data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capm', methods=['POST'])
def calculate_capm_api():
    try:
        data = request.get_json()
        beta = float(data['beta'])
        market_return = float(data['market_return'])
        risk_free_rate = float(data['risk_free_rate'])
        
        expected_return = calculate_capm(beta, market_return, risk_free_rate)
        
        return jsonify({
            'success': True,
            'expected_return': expected_return,
            'formula': f'E(Ri) = Rf + βi(Rm - Rf) = {risk_free_rate} + {beta} × ({market_return} - {risk_free_rate})'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/monte-carlo', methods=['POST'])
def run_monte_carlo_api():
    try:
        data = request.get_json()
        start_price = float(data['start_price'])
        mean_return = float(data['mean_return']) / 100  # Convert percentage to decimal
        volatility = float(data['volatility']) / 100  # Convert percentage to decimal
        time_horizon = int(data['time_horizon'])
        num_simulations = int(data['num_simulations'])
        
        simulations = run_monte_carlo_simulation(
            start_price, mean_return, volatility, time_horizon, num_simulations
        )
        
        return jsonify({
            'success': True,
            'simulations': simulations,
            'parameters': {
                'start_price': start_price,
                'mean_return': mean_return * 100,
                'volatility': volatility * 100,
                'time_horizon': time_horizon,
                'num_simulations': num_simulations
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/black-scholes', methods=['POST'])
def calculate_black_scholes_api():
    try:
        data = request.get_json()
        stock_price = float(data['stock_price'])
        strike_price = float(data['strike_price'])
        time_to_expiry = float(data['time_to_expiry'])
        volatility = float(data['volatility']) / 100  # Convert percentage to decimal
        risk_free_rate = float(data['risk_free_rate']) / 100  # Convert percentage to decimal
        
        call_price = black_scholes_call(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        put_price = black_scholes_put(stock_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        
        return jsonify({
            'success': True,
            'call_price': call_price,
            'put_price': put_price,
            'parameters': {
                'stock_price': stock_price,
                'strike_price': strike_price,
                'time_to_expiry': time_to_expiry,
                'volatility': volatility * 100,
                'risk_free_rate': risk_free_rate * 100
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/fama-french', methods=['POST'])
def calculate_fama_french_api():
    try:
        data = request.get_json()
        market_return = float(data.get('marketReturn', 10.0)) / 100
        risk_free_rate = float(data.get('riskFreeRate', 2.0)) / 100
        beta = float(data.get('beta', 1.0))
        smb_factor = float(data.get('smbFactor', 2.0)) / 100  # Small minus Big factor
        hml_factor = float(data.get('hmlFactor', 4.0)) / 100  # High minus Low factor
        smb_beta = float(data.get('smbBeta', 0.5))
        hml_beta = float(data.get('hmlBeta', 0.3))
        
        # Fama-French 3-factor model calculation
        # Expected Return = Rf + β1(Rm - Rf) + β2(SMB) + β3(HML)
        market_premium = market_return - risk_free_rate
        expected_return = risk_free_rate + (beta * market_premium) + (smb_beta * smb_factor) + (hml_beta * hml_factor)
        
        # Calculate factor contributions
        market_contribution = beta * market_premium
        smb_contribution = smb_beta * smb_factor
        hml_contribution = hml_beta * hml_factor
        
        return jsonify({
            'success': True,
            'expected_return': round(expected_return * 100, 2),
            'market_contribution': round(market_contribution * 100, 2),
            'smb_contribution': round(smb_contribution * 100, 2),
            'hml_contribution': round(hml_contribution * 100, 2),
            'risk_free_rate': round(risk_free_rate * 100, 2)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/mpt-optimization', methods=['POST'])
def calculate_mpt_api():
    try:
        data = request.get_json()
        returns = data.get('returns', [])  # List of expected returns for each asset
        volatilities = data.get('volatilities', [])  # List of volatilities for each asset
        correlations = data.get('correlations', [])  # Correlation matrix
        target_return = float(data.get('targetReturn', 10.0)) / 100
        risk_free_rate = float(data.get('riskFreeRate', 2.0)) / 100
        
        if len(returns) != len(volatilities) or len(returns) < 2:
            return jsonify({'success': False, 'error': 'Invalid input: need at least 2 assets with returns and volatilities'})
        
        # Create covariance matrix from volatilities and correlations
        n_assets = len(returns)
        cov_matrix = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i == j:
                    cov_matrix[i][j] = volatilities[i]**2
                else:
                    # Use correlation if provided, otherwise assume 0.3
                    corr = correlations[i][j] if i < len(correlations) and j < len(correlations[i]) else 0.3
                    cov_matrix[i][j] = corr * volatilities[i] * volatilities[j]
        
        # Simple optimization: find weights that minimize variance for target return
        # This is a simplified version - in practice, you'd use scipy.optimize
        
        # For demonstration, we'll create a simple efficient frontier
        num_portfolios = 50
        efficient_frontier = []
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            
            # Calculate portfolio return and risk
            portfolio_return = np.sum(weights * returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Calculate Sharpe ratio
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            efficient_frontier.append({
                'return': round(portfolio_return * 100, 2),
                'risk': round(portfolio_risk * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'weights': [round(w, 3) for w in weights]
            })
        
        # Find optimal portfolio (highest Sharpe ratio)
        optimal_portfolio = max(efficient_frontier, key=lambda x: x['sharpe_ratio'])
        
        return jsonify({
            'success': True,
            'efficient_frontier': efficient_frontier,
            'optimal_portfolio': optimal_portfolio,
            'risk_free_rate': round(risk_free_rate * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/financial-ratios')
def get_financial_ratios_api():
    ticker = request.args.get('ticker', 'AAPL')
    
    try:
        ratios = get_financial_ratios(ticker)
        return jsonify({
            'success': True,
            'ratios': ratios
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock-info')
def get_stock_info_api():
    ticker = request.args.get('ticker', 'AAPL')
    market = request.args.get('market', 'NYSE')
    
    try:
        # Get stock info from yfinance
        if ticker in MARKETS['PSX']['stocks']:
            possible_tickers = [f"{ticker}.PSX", f"{ticker}.PK", ticker, f"{ticker}.IS"]
        else:
            possible_tickers = [ticker]
        
        stock_info = None
        for test_ticker in possible_tickers:
            try:
                stock = yf.Ticker(test_ticker)
                info = stock.info
                if info and len(info) > 5:
                    stock_info = info
                    break
            except:
                continue
        
        # Extract key metrics
        metrics = {
            'beta': stock_info.get('beta', 1.0) if stock_info else 1.0,
            'market_cap': stock_info.get('marketCap', 0) if stock_info else 0,
            'pe_ratio': stock_info.get('trailingPE', 0) if stock_info else 0,
            'pb_ratio': stock_info.get('priceToBook', 0) if stock_info else 0,
            'dividend_yield': stock_info.get('dividendYield', 0) if stock_info else 0,
            'roe': stock_info.get('returnOnEquity', 0) if stock_info else 0,
            'debt_equity': stock_info.get('debtToEquity', 0) if stock_info else 0,
            'current_price': stock_info.get('currentPrice', 0) if stock_info else 0,
            'volume': stock_info.get('volume', 0) if stock_info else 0,
            'currency': 'PKR' if market == 'PSX' else 'USD'
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/markets')
def get_markets():
    """Get available markets and their stocks"""
    return jsonify({
        'success': True,
        'markets': MARKETS
    })

@app.route('/api/market-stocks/<market>')
def get_market_stocks(market):
    """Get stocks for a specific market"""
    if market in MARKETS:
        return jsonify({
            'success': True,
            'market': MARKETS[market]
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Market not found'
        })

@app.route('/api/financial-metrics', methods=['POST'])
def calculate_financial_metrics_api():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL')
        market = data.get('market', 'NYSE')
        
        # Get stock data and financial ratios
        stock_data = fetch_stock_data(ticker, market)
        ratios = get_financial_ratios(ticker)
        
        # Calculate additional metrics
        metrics = calculate_comprehensive_metrics(ticker, stock_data, ratios)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'market': market,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/technical-indicators')
def get_technical_indicators():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '60d')
    
    try:
        data = fetch_stock_data(ticker, period=period)
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'No data available'})
        
        indicators = calculate_technical_indicators(data)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_indicators = {}
        for key, value in indicators.items():
            if isinstance(value, np.ndarray):
                serializable_indicators[key] = value.tolist()
            else:
                serializable_indicators[key] = value
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'indicators': serializable_indicators
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/support-resistance')
def get_support_resistance():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '60d')
    
    try:
        data = fetch_stock_data(ticker, period=period)
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'No data available'})
        
        levels = calculate_support_resistance(data)
        fibonacci = calculate_fibonacci_levels(data)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'support_resistance': levels,
            'fibonacci_levels': fibonacci
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/risk-metrics')
def get_risk_metrics():
    ticker = request.args.get('ticker', 'AAPL')
    period = request.args.get('period', '1y')
    
    try:
        data = fetch_stock_data(ticker, period=period)
        if data is None or data.empty:
            return jsonify({'success': False, 'error': 'No data available'})
        
        risk_metrics = calculate_risk_metrics(data)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'risk_metrics': risk_metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market-screener', methods=['POST'])
def market_screener():
    try:
        criteria = request.get_json()
        
        # Apply default criteria if not provided
        default_criteria = {
            'price_min': 0,
            'price_max': 10000,
            'market_cap_min': 0,
            'market_cap_max': float('inf'),
            'pe_min': 0,
            'pe_max': 100,
            'dividend_min': 0,
            'dividend_max': 20,
            'beta_min': 0,
            'beta_max': 3,
            'volume_min': 0,
            'volume_max': float('inf'),
            'sort_by': 'market_cap',
            'sort_order': 'desc',
            'limit': 50
        }
        
        # Update with provided criteria
        default_criteria.update(criteria)
        
        screened_stocks = screen_market(default_criteria)
        
        return jsonify({
            'success': True,
            'criteria': default_criteria,
            'results': screened_stocks,
            'count': len(screened_stocks)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock-sentiment')
def get_stock_sentiment():
    ticker = request.args.get('ticker', 'AAPL')
    
    try:
        sentiment = analyze_stock_sentiment(ticker)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'sentiment': sentiment
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock-report')
def get_stock_report():
    ticker = request.args.get('ticker', 'AAPL')
    market = request.args.get('market', 'NYSE')
    
    try:
        report = generate_stock_report(ticker, market)
        
        if report is None:
            return jsonify({'success': False, 'error': 'Unable to generate report'})
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/screening-criteria')
def get_screening_criteria():
    """Get available screening criteria"""
    return jsonify({
        'success': True,
        'criteria': SCREENING_CRITERIA
    })

@app.route('/api/technical-indicators-config')
def get_technical_indicators_config():
    """Get technical indicators configuration"""
    return jsonify({
        'success': True,
        'indicators': TECHNICAL_INDICATORS
    })

@app.route('/api/portfolio-analysis', methods=['POST'])
def portfolio_analysis():
    """Analyze a portfolio of stocks"""
    try:
        data = request.get_json()
        portfolio = data.get('portfolio', [])  # List of tickers
        weights = data.get('weights', [])  # Portfolio weights
        
        if len(portfolio) != len(weights):
            return jsonify({'success': False, 'error': 'Portfolio and weights must have same length'})
        
        portfolio_data = []
        total_return = 0
        total_volatility = 0
        
        for i, ticker in enumerate(portfolio):
            try:
                stock_data = fetch_stock_data(ticker, period='1y')
                if stock_data is not None and not stock_data.empty:
                    returns = stock_data['Close'].pct_change().dropna()
                    annual_return = returns.mean() * 252 * 100
                    annual_volatility = returns.std() * np.sqrt(252) * 100
                    
                    portfolio_data.append({
                        'ticker': ticker,
                        'weight': weights[i],
                        'annual_return': annual_return,
                        'annual_volatility': annual_volatility,
                        'current_price': stock_data['Close'].iloc[-1]
                    })
                    
                    total_return += weights[i] * annual_return
                    total_volatility += weights[i] * annual_volatility
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'total_return': total_return,
            'total_volatility': total_volatility,
            'sharpe_ratio': (total_return - 2.0) / total_volatility if total_volatility > 0 else 0,
            'diversification_score': len(portfolio_data) / 10,  # Simple diversification metric
            'holdings': portfolio_data
        }
        
        return jsonify({
            'success': True,
            'portfolio_analysis': portfolio_metrics
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def watchlist_management():
    """Manage watchlist (simplified - in production, use database)"""
    try:
        if request.method == 'GET':
            # Return sample watchlist
            watchlist = [
                {'ticker': 'AAPL', 'added_date': '2024-01-01', 'notes': 'Tech leader'},
                {'ticker': 'MSFT', 'added_date': '2024-01-02', 'notes': 'Cloud growth'},
                {'ticker': 'GOOGL', 'added_date': '2024-01-03', 'notes': 'AI potential'}
            ]
            return jsonify({'success': True, 'watchlist': watchlist})
        
        elif request.method == 'POST':
            data = request.get_json()
            ticker = data.get('ticker')
            notes = data.get('notes', '')
            
            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker required'})
            
            # In production, save to database
            return jsonify({
                'success': True,
                'message': f'{ticker} added to watchlist',
                'ticker': ticker,
                'notes': notes
            })
        
        elif request.method == 'DELETE':
            data = request.get_json()
            ticker = data.get('ticker')
            
            if not ticker:
                return jsonify({'success': False, 'error': 'Ticker required'})
            
            # In production, remove from database
            return jsonify({
                'success': True,
                'message': f'{ticker} removed from watchlist'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market-overview')
def market_overview():
    """Get market overview and summary statistics - Optimized for performance"""
    try:
        # Check cache first
        cache_key = 'market_overview'
        if cache_key in data_cache:
            cache_time, cached_data = data_cache[cache_key]
            if time.time() - cache_time < cache_duration:
                return jsonify({
                    'success': True,
                    'market_overview': cached_data,
                    'timestamp': datetime.now().isoformat(),
                    'cached': True
                })
        
        market_stats = {
            'NYSE': {
                'total_stocks': len(MARKETS['NYSE']['stocks']),
                'top_gainers': [],
                'top_losers': [],
                'most_active': [],
                'market_sentiment': 'bullish'
            },
            'PSX': {
                'total_stocks': len(MARKETS['PSX']['stocks']),
                'top_gainers': [],
                'top_losers': [],
                'most_active': [],
                'market_sentiment': 'neutral'
            }
        }
        
        # Use fast data service for instant market overview
        market_stats = fast_data_service.get_market_overview()
        
        # Cache the result
        data_cache[cache_key] = (time.time(), market_stats)
        
        return jsonify({
            'success': True,
            'market_overview': market_stats,
            'timestamp': datetime.now().isoformat(),
            'cached': False
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def calculate_comprehensive_metrics(ticker, stock_data, ratios):
    """Calculate comprehensive financial metrics with risk categorization"""
    
    # Initialize metrics dictionary
    metrics = {}
    
    try:
        # 1. Beta (Volatility vs market)
        beta = ratios.get('beta', 1.0)
        metrics['beta'] = {
            'value': round(beta, 3),
            'category': 'Low' if beta < 0.8 else 'Medium' if beta <= 1.2 else 'High',
            'description': 'Volatility compared to market',
            'interpretation': f"{'Low' if beta < 0.8 else 'Medium' if beta <= 1.2 else 'High'} volatility relative to market"
        }
        
        # 2. Standard Deviation (Return volatility)
        if not stock_data.empty and len(stock_data) > 1:
            returns = stock_data['Close'].pct_change().dropna()
            std_dev = returns.std() * 100
            metrics['standard_deviation'] = {
                'value': round(std_dev, 2),
                'category': 'Low' if std_dev < 10 else 'Medium' if std_dev <= 20 else 'High',
                'description': 'Return volatility',
                'interpretation': f"{'Low' if std_dev < 10 else 'Medium' if std_dev <= 20 else 'High'} volatility"
            }
        else:
            metrics['standard_deviation'] = {
                'value': 15.0,
                'category': 'Medium',
                'description': 'Return volatility',
                'interpretation': 'Medium volatility (estimated)'
            }
        
        # 3. Max Drawdown (Worst historical loss)
        if not stock_data.empty and len(stock_data) > 1:
            cumulative = (1 + stock_data['Close'].pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max * 100).min()
            metrics['max_drawdown'] = {
                'value': round(abs(drawdown), 2),
                'category': 'Low' if abs(drawdown) < 10 else 'Medium' if abs(drawdown) <= 25 else 'High',
                'description': 'Worst historical loss',
                'interpretation': f"{'Low' if abs(drawdown) < 10 else 'Medium' if abs(drawdown) <= 25 else 'High'} risk"
            }
        else:
            metrics['max_drawdown'] = {
                'value': 15.0,
                'category': 'Medium',
                'description': 'Worst historical loss',
                'interpretation': 'Medium risk (estimated)'
            }
        
        # 4. Value at Risk (VaR) - 95% confidence
        if not stock_data.empty and len(stock_data) > 1:
            returns = stock_data['Close'].pct_change().dropna()
            var_95 = np.percentile(returns, 5) * 100
            metrics['var'] = {
                'value': round(abs(var_95), 2),
                'category': 'Low' if abs(var_95) < 5 else 'Medium' if abs(var_95) <= 10 else 'High',
                'description': 'Maximum expected loss (95% confidence)',
                'interpretation': f"{'Low' if abs(var_95) < 5 else 'Medium' if abs(var_95) <= 10 else 'High'} risk"
            }
        else:
            metrics['var'] = {
                'value': 8.0,
                'category': 'Medium',
                'description': 'Maximum expected loss (95% confidence)',
                'interpretation': 'Medium risk (estimated)'
            }
        
        # 5. Debt-to-Equity (D/E)
        de_ratio = ratios.get('debtToEquity', 0.5)
        metrics['debt_to_equity'] = {
            'value': round(de_ratio, 2),
            'category': 'Low' if de_ratio < 0.5 else 'Medium' if de_ratio <= 1 else 'High',
            'description': 'Leverage level',
            'interpretation': f"{'Low' if de_ratio < 0.5 else 'Medium' if de_ratio <= 1 else 'High'} leverage"
        }
        
        # 6. Interest Coverage Ratio
        interest_coverage = ratios.get('interestCoverage', 2.5)
        metrics['interest_coverage'] = {
            'value': round(interest_coverage, 2),
            'category': 'Low' if interest_coverage > 3 else 'Medium' if interest_coverage >= 1.5 else 'High',
            'description': 'Ability to pay interest',
            'interpretation': f"{'Low' if interest_coverage > 3 else 'Medium' if interest_coverage >= 1.5 else 'High'} risk"
        }
        
        # 7. Current Ratio
        current_ratio = ratios.get('currentRatio', 1.2)
        metrics['current_ratio'] = {
            'value': round(current_ratio, 2),
            'category': 'Healthy' if current_ratio > 1.5 else 'Watch' if current_ratio >= 1 else 'Risky',
            'description': 'Short-term liquidity',
            'interpretation': f"{'Healthy' if current_ratio > 1.5 else 'Watch' if current_ratio >= 1 else 'Risky'} liquidity"
        }
        
        # 8. Altman Z-Score (Simplified calculation)
        # Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        # Where: A = Working Capital/Total Assets, B = Retained Earnings/Total Assets, 
        # C = EBIT/Total Assets, D = Market Value/Total Liabilities, E = Sales/Total Assets
        # For simplicity, we'll use estimated values
        altman_z = ratios.get('altmanZScore', 2.5)
        metrics['altman_z_score'] = {
            'value': round(altman_z, 2),
            'category': 'Safe' if altman_z > 3 else 'Watch' if altman_z >= 1.8 else 'Risky',
            'description': 'Bankruptcy risk',
            'interpretation': f"{'Safe' if altman_z > 3 else 'Watch' if altman_z >= 1.8 else 'Risky'} financial health"
        }
        
        # 9. CAGR / Average Return
        if not stock_data.empty and len(stock_data) > 1:
            total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
            days = (stock_data.index[-1] - stock_data.index[0]).days
            cagr = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) ** (365/days) - 1) * 100
            avg_return = total_return / (days / 365)
            
            metrics['cagr'] = {
                'value': round(cagr, 2),
                'category': 'High' if cagr > 15 else 'Medium' if cagr >= 8 else 'Low',
                'description': 'Compound annual growth rate',
                'interpretation': f"{'High' if cagr > 15 else 'Medium' if cagr >= 8 else 'Low'} growth"
            }
        else:
            metrics['cagr'] = {
                'value': 10.0,
                'category': 'Medium',
                'description': 'Compound annual growth rate',
                'interpretation': 'Medium growth (estimated)'
            }
        
        # 10. EPS (Earnings Per Share)
        eps = ratios.get('trailingEps', 5.0)
        metrics['eps'] = {
            'value': round(eps, 2),
            'category': 'Strong' if eps > 10 else 'Medium' if eps >= 2 else 'Weak',
            'description': 'Profitability per share',
            'interpretation': f"{'Strong' if eps > 10 else 'Medium' if eps >= 2 else 'Weak'} earnings"
        }
        
        # 11. ROE (Return on Equity)
        roe = ratios.get('returnOnEquity', 12.0) * 100
        metrics['roe'] = {
            'value': round(roe, 2),
            'category': 'High' if roe > 15 else 'Medium' if roe >= 10 else 'Low',
            'description': 'Efficiency of shareholder capital',
            'interpretation': f"{'High' if roe > 15 else 'Medium' if roe >= 10 else 'Low'} efficiency"
        }
        
        # 12. ROA (Return on Assets)
        roa = ratios.get('returnOnAssets', 3.5) * 100
        metrics['roa'] = {
            'value': round(roa, 2),
            'category': 'High' if roa > 5 else 'Medium' if roa >= 2 else 'Low',
            'description': 'Efficiency of asset usage',
            'interpretation': f"{'High' if roa > 5 else 'Medium' if roa >= 2 else 'Low'} efficiency"
        }
        
        # 13. Dividend Yield
        dividend_yield = ratios.get('dividendYield', 2.0) * 100
        metrics['dividend_yield'] = {
            'value': round(dividend_yield, 2),
            'category': 'High' if dividend_yield > 3 else 'Medium' if dividend_yield >= 1 else 'Low',
            'description': 'Dividend income',
            'interpretation': f"{'High' if dividend_yield > 3 else 'Medium' if dividend_yield >= 1 else 'Low'} yield"
        }
        
        # 14. Net Profit Margin
        net_margin = ratios.get('profitMargins', 0.12) * 100
        metrics['net_profit_margin'] = {
            'value': round(net_margin, 2),
            'category': 'High' if net_margin > 15 else 'Medium' if net_margin >= 8 else 'Low',
            'description': 'Overall profitability',
            'interpretation': f"{'High' if net_margin > 15 else 'Medium' if net_margin >= 8 else 'Low'} profitability"
        }
        
        # 15. Free Cash Flow (FCF) - Simplified
        fcf = ratios.get('freeCashflow', 1000000000)  # In millions
        metrics['free_cash_flow'] = {
            'value': round(fcf / 1000000, 2),  # Convert to millions
            'category': 'Healthy' if fcf > 0 else 'Risky',
            'description': 'Cash after operations',
            'interpretation': f"{'Healthy' if fcf > 0 else 'Risky'} cash flow"
        }
        
        # 16. P/E Ratio (Price-to-Earnings)
        pe_ratio = ratios.get('trailingPE', 20.0)
        metrics['pe_ratio'] = {
            'value': round(pe_ratio, 2),
            'category': 'Cheap' if pe_ratio < 15 else 'Fair' if pe_ratio <= 25 else 'Expensive',
            'description': 'Price vs earnings',
            'interpretation': f"{'Cheap' if pe_ratio < 15 else 'Fair' if pe_ratio <= 25 else 'Expensive'} valuation"
        }
        
        # 17. PEG Ratio (Price/Earnings to Growth)
        peg_ratio = ratios.get('pegRatio', 1.5)
        metrics['peg_ratio'] = {
            'value': round(peg_ratio, 2),
            'category': 'Undervalued' if peg_ratio < 1 else 'Fair' if peg_ratio <= 2 else 'Overvalued',
            'description': 'Valuation vs growth',
            'interpretation': f"{'Undervalued' if peg_ratio < 1 else 'Fair' if peg_ratio <= 2 else 'Overvalued'}"
        }
        
        # 18. P/B Ratio (Price-to-Book)
        pb_ratio = ratios.get('priceToBook', 2.5)
        metrics['pb_ratio'] = {
            'value': round(pb_ratio, 2),
            'category': 'Undervalued' if pb_ratio < 1 else 'Fair' if pb_ratio <= 3 else 'Expensive',
            'description': 'Price vs book value',
            'interpretation': f"{'Undervalued' if pb_ratio < 1 else 'Fair' if pb_ratio <= 3 else 'Expensive'} valuation"
        }
        
        # 19. EV/EBITDA
        ev_ebitda = ratios.get('enterpriseToEbitda', 15.0)
        metrics['ev_ebitda'] = {
            'value': round(ev_ebitda, 2),
            'category': 'Fair' if ev_ebitda < 10 else 'Medium' if ev_ebitda <= 20 else 'High',
            'description': 'Valuation vs earnings',
            'interpretation': f"{'Fair' if ev_ebitda < 10 else 'Medium' if ev_ebitda <= 20 else 'High'} valuation"
        }
        
        # 20. P/S Ratio (Price-to-Sales)
        ps_ratio = ratios.get('priceToSalesTrailing12Months', 2.0)
        metrics['ps_ratio'] = {
            'value': round(ps_ratio, 2),
            'category': 'Undervalued' if ps_ratio < 1 else 'Fair' if ps_ratio <= 3 else 'Expensive',
            'description': 'Price vs sales',
            'interpretation': f"{'Undervalued' if ps_ratio < 1 else 'Fair' if ps_ratio <= 3 else 'Expensive'} valuation"
        }
        
        # 21. Sharpe Ratio (Risk-adjusted return)
        if not stock_data.empty and len(stock_data) > 1:
            returns = stock_data['Close'].pct_change().dropna()
            avg_return = returns.mean() * 252  # Annualized
            std_return = returns.std() * np.sqrt(252)  # Annualized
            risk_free_rate = 0.02  # 2% risk-free rate
            sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            metrics['sharpe_ratio'] = {
                'value': round(sharpe_ratio, 3),
                'category': 'Excellent' if sharpe_ratio > 2 else 'Good' if sharpe_ratio >= 1 else 'Poor',
                'description': 'Return per unit of total risk',
                'interpretation': f"{'Excellent' if sharpe_ratio > 2 else 'Good' if sharpe_ratio >= 1 else 'Poor'} risk-adjusted return"
            }
        else:
            metrics['sharpe_ratio'] = {
                'value': 1.2,
                'category': 'Good',
                'description': 'Return per unit of total risk',
                'interpretation': 'Good risk-adjusted return (estimated)'
            }
        
        # 22. Treynor Ratio (Beta-adjusted return)
        if not stock_data.empty and len(stock_data) > 1:
            returns = stock_data['Close'].pct_change().dropna()
            avg_return = returns.mean() * 252  # Annualized
            risk_free_rate = 0.02  # 2% risk-free rate
            treynor_ratio = (avg_return - risk_free_rate) / beta if beta > 0 else 0
            
            metrics['treynor_ratio'] = {
                'value': round(treynor_ratio, 3),
                'category': 'Strong' if treynor_ratio > 0.5 else 'Medium' if treynor_ratio >= 0.2 else 'Weak',
                'description': 'Return per unit of beta risk',
                'interpretation': f"{'Strong' if treynor_ratio > 0.5 else 'Medium' if treynor_ratio >= 0.2 else 'Weak'} beta-adjusted return"
            }
        else:
            metrics['treynor_ratio'] = {
                'value': 0.3,
                'category': 'Medium',
                'description': 'Return per unit of beta risk',
                'interpretation': 'Medium beta-adjusted return (estimated)'
            }
        
        # 23. Sortino Ratio (Downside risk-adjusted return)
        if not stock_data.empty and len(stock_data) > 1:
            returns = stock_data['Close'].pct_change().dropna()
            avg_return = returns.mean() * 252  # Annualized
            risk_free_rate = 0.02  # 2% risk-free rate
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
            sortino_ratio = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            metrics['sortino_ratio'] = {
                'value': round(sortino_ratio, 3),
                'category': 'Excellent' if sortino_ratio > 2 else 'Good' if sortino_ratio >= 1 else 'Poor',
                'description': 'Return per unit of downside risk',
                'interpretation': f"{'Excellent' if sortino_ratio > 2 else 'Good' if sortino_ratio >= 1 else 'Poor'} downside risk-adjusted return"
            }
        else:
            metrics['sortino_ratio'] = {
                'value': 1.5,
                'category': 'Good',
                'description': 'Return per unit of downside risk',
                'interpretation': 'Good downside risk-adjusted return (estimated)'
            }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Return basic metrics if calculation fails
        metrics = {
            'beta': {'value': 1.0, 'category': 'Medium', 'description': 'Volatility vs market', 'interpretation': 'Medium volatility'},
            'standard_deviation': {'value': 15.0, 'category': 'Medium', 'description': 'Return volatility', 'interpretation': 'Medium volatility'},
            'max_drawdown': {'value': 15.0, 'category': 'Medium', 'description': 'Worst historical loss', 'interpretation': 'Medium risk'}
        }
    
    return metrics

def save_results_to_csv(data, filename):
    """
    Save calculation results to CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'Calculation Type', 'Parameters', 'Results'])
            writer.writerow([timestamp, data['type'], str(data['parameters']), str(data['results'])])
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

# API Endpoints for Financial Calculations
@app.route('/api/calculate-capm', methods=['POST'])
def calculate_capm():
    """Calculate CAPM expected return"""
    try:
        data = request.get_json()
        risk_free_rate = float(data.get('riskFreeRate', 0.02))
        beta = float(data.get('beta', 1.0))
        market_return = float(data.get('marketReturn', 0.08))
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        return jsonify({
            'success': True,
            'expected_return': expected_return,
            'formula': 'E(Ri) = Rf + βi(Rm - Rf)'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calculate-monte-carlo', methods=['POST'])
def calculate_monte_carlo():
    """Run Monte Carlo simulation"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL')
        num_simulations = int(data.get('numSimulations', 1000))
        time_horizon = int(data.get('timeHorizon', 252))
        
        # Get stock data
        stock_data = fast_data_service.get_stock_data(ticker, period='1y')
        
        if stock_data.empty:
            return jsonify({'success': False, 'error': 'No data available for simulation'})
        
        # Calculate parameters
        returns = stock_data['Close'].pct_change().dropna()
        mean_return = returns.mean() * 252  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Run simulation
        np.random.seed(42)
        simulation_results = []
        
        for _ in range(num_simulations):
            price_path = [stock_data['Close'].iloc[-1]]
            for _ in range(time_horizon):
                change = np.random.normal(mean_return/252, volatility/np.sqrt(252))
                new_price = price_path[-1] * (1 + change)
                price_path.append(new_price)
            simulation_results.append(price_path)
        
        # Calculate statistics
        final_prices = [path[-1] for path in simulation_results]
        mean_final_price = np.mean(final_prices)
        std_final_price = np.std(final_prices)
        
        return jsonify({
            'success': True,
            'parameters': {
                'num_simulations': num_simulations,
                'time_horizon': time_horizon,
                'mean_return': mean_return * 100,
                'volatility': volatility * 100
            },
            'results': {
                'mean_final_price': mean_final_price,
                'std_final_price': std_final_price,
                'confidence_interval': [
                    np.percentile(final_prices, 5),
                    np.percentile(final_prices, 95)
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calculate-black-scholes', methods=['POST'])
def calculate_black_scholes():
    """Calculate Black-Scholes option prices"""
    try:
        data = request.get_json()
        stock_price = float(data.get('stockPrice', 100))
        strike_price = float(data.get('strikePrice', 100))
        time_to_expiry = float(data.get('timeToExpiry', 1))
        risk_free_rate = float(data.get('riskFreeRate', 0.05))
        volatility = float(data.get('volatility', 0.2))
        
        # Black-Scholes calculation
        d1 = (np.log(stock_price/strike_price) + (risk_free_rate + 0.5*volatility**2)*time_to_expiry) / (volatility*np.sqrt(time_to_expiry))
        d2 = d1 - volatility*np.sqrt(time_to_expiry)
        
        # Call option price
        call_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate*time_to_expiry) * norm.cdf(d2)
        
        # Put option price
        put_price = strike_price * np.exp(-risk_free_rate*time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
        
        return jsonify({
            'success': True,
            'call_price': call_price,
            'put_price': put_price,
            'parameters': {
                'stock_price': stock_price,
                'strike_price': strike_price,
                'time_to_expiry': time_to_expiry,
                'risk_free_rate': risk_free_rate,
                'volatility': volatility
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calculate-fama-french', methods=['POST'])
def calculate_fama_french():
    """Calculate Fama-French 3-Factor Model"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL')
        risk_free_rate = float(data.get('riskFreeRate', 0.02))
        
        # Get stock data
        stock_data = fast_data_service.get_stock_data(ticker, period='1y')
        
        if stock_data.empty:
            return jsonify({'success': False, 'error': 'No data available for calculation'})
        
        # Calculate returns
        returns = stock_data['Close'].pct_change().dropna()
        
        # Simulate factor returns (in real implementation, these would come from market data)
        market_return = 0.08  # 8% market return
        smb_return = 0.02     # Small minus Big factor
        hml_return = 0.03     # High minus Low factor
        
        # Factor loadings (simulated)
        beta_market = 1.0
        beta_smb = 0.2
        beta_hml = -0.1
        
        # Calculate expected return
        expected_return = risk_free_rate + beta_market * (market_return - risk_free_rate) + beta_smb * smb_return + beta_hml * hml_return
        
        # Factor contributions
        market_contribution = beta_market * (market_return - risk_free_rate) * 100
        smb_contribution = beta_smb * smb_return * 100
        hml_contribution = beta_hml * hml_return * 100
        
        return jsonify({
            'success': True,
            'expected_return': expected_return * 100,
            'market_contribution': market_contribution,
            'smb_contribution': smb_contribution,
            'hml_contribution': hml_contribution,
            'factor_loadings': {
                'market_beta': beta_market,
                'smb_beta': beta_smb,
                'hml_beta': beta_hml
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calculate-mpt', methods=['POST'])
def calculate_mpt():
    """Calculate Modern Portfolio Theory metrics"""
    try:
        data = request.get_json()
        tickers = data.get('tickers', ['AAPL', 'MSFT', 'GOOGL'])
        weights = data.get('weights', [0.33, 0.33, 0.34])
        
        if len(tickers) != len(weights):
            return jsonify({'success': False, 'error': 'Number of tickers must match number of weights'})
        
        # Get stock data for all tickers
        stock_data_list = []
        for ticker in tickers:
            stock_data = fast_data_service.get_stock_data(ticker, period='1y')
            if not stock_data.empty:
                stock_data_list.append(stock_data['Close'].pct_change().dropna())
        
        if len(stock_data_list) < 2:
            return jsonify({'success': False, 'error': 'Need at least 2 stocks for portfolio analysis'})
        
        # Calculate returns matrix
        min_length = min(len(returns) for returns in stock_data_list)
        returns_matrix = np.array([returns.iloc[-min_length:] for returns in stock_data_list])
        
        # Calculate portfolio metrics
        portfolio_returns = np.sum(returns_matrix * np.array(weights)[:, np.newaxis], axis=0)
        
        total_return = np.mean(portfolio_returns) * 252 * 100  # Annualized
        total_volatility = np.std(portfolio_returns) * np.sqrt(252) * 100  # Annualized
        
        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (total_return/100 - risk_free_rate) / (total_volatility/100)
        
        # Calculate diversification score
        individual_volatilities = [np.std(returns) * np.sqrt(252) * 100 for returns in stock_data_list]
        weighted_individual_vol = np.sum(np.array(individual_volatilities) * np.array(weights))
        diversification_score = 1 - (total_volatility / weighted_individual_vol)
        
        return jsonify({
            'success': True,
            'portfolio_analysis': {
                'total_return': total_return,
                'total_volatility': total_volatility,
                'sharpe_ratio': sharpe_ratio,
                'diversification_score': diversification_score
            },
            'constituents': {
                'tickers': tickers,
                'weights': weights,
                'individual_volatilities': individual_volatilities
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)