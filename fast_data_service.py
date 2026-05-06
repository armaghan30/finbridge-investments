"""
Fast Data Service - Provides instant stock data for demonstration
This service generates realistic PSX stock data instantly without external API calls
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class FastDataService:
    def __init__(self):
        self.stock_data_cache = {}
        # Prices sourced live from dps.psx.com.pk on May 6, 2026 (LDCP)
        self.market_data = {
            'PSX': {
                # --- Oil & Gas ---
                'OGDC': {'base_price': 311.42, 'volatility': 0.025, 'trend': 0.001},
                'PPL':  {'base_price': 215.97, 'volatility': 0.022, 'trend': 0.0008},
                'PSO':  {'base_price': 356.06, 'volatility': 0.024, 'trend': 0.001},
                'SHEL': {'base_price': 189.13, 'volatility': 0.02,  'trend': 0.0008},
                'ATRL': {'base_price': 900.28, 'volatility': 0.026, 'trend': 0.0015},
                'MARI': {'base_price': 637.03, 'volatility': 0.025, 'trend': 0.002},
                'POL':  {'base_price': 662.29, 'volatility': 0.024, 'trend': 0.001},
                'APL':  {'base_price': 595.29, 'volatility': 0.022, 'trend': 0.001},
                'SNGP': {'base_price': 97.37,  'volatility': 0.025, 'trend': 0.001},
                'SSGC': {'base_price': 27.65,  'volatility': 0.028, 'trend': 0.001},
                'NRL':  {'base_price': 369.50, 'volatility': 0.02,  'trend': 0.001},
                'PRL':  {'base_price': 35.81,  'volatility': 0.03,  'trend': 0.001},
                'HASCOL': {'base_price': 21.55, 'volatility': 0.04, 'trend': 0.001},
                'CNERGY': {'base_price': 8.19,  'volatility': 0.045,'trend': 0.001},
                # --- Banking ---
                'MCB':  {'base_price': 403.41, 'volatility': 0.018, 'trend': 0.001},
                'UBL':  {'base_price': 385.11, 'volatility': 0.02,  'trend': 0.0012},
                'HBL':  {'base_price': 285.17, 'volatility': 0.019, 'trend': 0.001},
                'MEBL': {'base_price': 483.32, 'volatility': 0.018, 'trend': 0.0015},
                'BAHL': {'base_price': 169.66, 'volatility': 0.019, 'trend': 0.001},
                'ABL':  {'base_price': 178.99, 'volatility': 0.02,  'trend': 0.001},
                'BAFL': {'base_price': 58.26,  'volatility': 0.022, 'trend': 0.001},
                'NBP':  {'base_price': 176.12, 'volatility': 0.022, 'trend': 0.001},
                'BOP':  {'base_price': 35.06,  'volatility': 0.03,  'trend': 0.001},
                'AKBL': {'base_price': 94.57,  'volatility': 0.025, 'trend': 0.001},
                'FABL': {'base_price': 87.51,  'volatility': 0.022, 'trend': 0.001},
                'JSBL': {'base_price': 13.65,  'volatility': 0.035, 'trend': 0.001},
                'SNBL': {'base_price': 18.80,  'volatility': 0.028, 'trend': 0.001},
                'BOK':  {'base_price': 33.00,  'volatility': 0.025, 'trend': 0.001},
                # --- Cement ---
                'LUCK': {'base_price': 409.10, 'volatility': 0.025, 'trend': 0.0015},
                'DGKC': {'base_price': 175.55, 'volatility': 0.028, 'trend': 0.001},
                'DCL':  {'base_price': 9.02,   'volatility': 0.035, 'trend': 0.001},
                'PIOC': {'base_price': 217.51, 'volatility': 0.028, 'trend': 0.001},
                'MLCF': {'base_price': 78.89,  'volatility': 0.032, 'trend': 0.001},
                'FCCL': {'base_price': 47.74,  'volatility': 0.03,  'trend': 0.001},
                'KOHC': {'base_price': 79.00,  'volatility': 0.025, 'trend': 0.001},
                'CHCC': {'base_price': 274.13, 'volatility': 0.025, 'trend': 0.001},
                'ACPL': {'base_price': 210.06, 'volatility': 0.022, 'trend': 0.001},
                'BWCL': {'base_price': 444.45, 'volatility': 0.028, 'trend': 0.001},
                'GWLC': {'base_price': 46.70,  'volatility': 0.032, 'trend': 0.001},
                'THCCL':{'base_price': 53.24,  'volatility': 0.03,  'trend': 0.001},
                # --- Fertilizer ---
                'FFC':   {'base_price': 515.42, 'volatility': 0.02,  'trend': 0.001},
                'EFERT': {'base_price': 197.06, 'volatility': 0.023, 'trend': 0.0012},
                'FFBL':  {'base_price': 88.94,  'volatility': 0.03,  'trend': 0.001},
                'ENGRO': {'base_price': 485.38, 'volatility': 0.022, 'trend': 0.001},
                'FATIMA':{'base_price': 132.74, 'volatility': 0.025, 'trend': 0.001},
                # --- Power ---
                'HUBC': {'base_price': 214.43, 'volatility': 0.02,  'trend': 0.001},
                'KEL':  {'base_price': 7.65,   'volatility': 0.04,  'trend': 0.001},
                'KAPCO':{'base_price': 26.99,  'volatility': 0.022, 'trend': 0.001},
                'NCPL': {'base_price': 66.52,  'volatility': 0.028, 'trend': 0.001},
                'PKGP': {'base_price': 44.50,  'volatility': 0.03,  'trend': 0.001},
                'TSPL': {'base_price': 9.15,   'volatility': 0.025, 'trend': 0.001},
                'EPQL': {'base_price': 23.18,  'volatility': 0.022, 'trend': 0.001},
                'SAPM': {'base_price': 8.72,   'volatility': 0.035, 'trend': 0.001},
                # --- Food & Consumer ---
                'NESTLE':   {'base_price': 7462.30, 'volatility': 0.015, 'trend': 0.001},
                'UNILEVER': {'base_price': 26129.09,'volatility': 0.018, 'trend': 0.001},
                'COLGATE':  {'base_price': 1090.35, 'volatility': 0.016, 'trend': 0.0008},
                'PAKT':     {'base_price': 1423.40, 'volatility': 0.02,  'trend': 0.001},
                'QUICE':    {'base_price': 26.13,   'volatility': 0.025, 'trend': 0.001},
                'FFL':      {'base_price': 16.43,   'volatility': 0.035, 'trend': 0.001},
                'NATF':     {'base_price': 373.96,  'volatility': 0.018, 'trend': 0.001},
                'ISML':     {'base_price': 1891.87, 'volatility': 0.018, 'trend': 0.001},
                'TREET':    {'base_price': 23.76,   'volatility': 0.028, 'trend': 0.001},
                'SING':     {'base_price': 31.32,   'volatility': 0.03,  'trend': 0.001},
                # --- Chemicals ---
                'ICI':  {'base_price': 596.48, 'volatility': 0.022, 'trend': 0.0012},
                'LOTTE':{'base_price': 26.14,  'volatility': 0.028, 'trend': 0.001},
                'INIL': {'base_price': 156.65, 'volatility': 0.022, 'trend': 0.001},
                'CEPB': {'base_price': 29.13,  'volatility': 0.025, 'trend': 0.001},
                'EPCL': {'base_price': 33.25,  'volatility': 0.025, 'trend': 0.001},
                'SITC': {'base_price': 880.00, 'volatility': 0.022, 'trend': 0.001},
                'PKGS': {'base_price': 708.31, 'volatility': 0.02,  'trend': 0.001},
                # --- Automobile ---
                'INDU': {'base_price': 2091.11, 'volatility': 0.022, 'trend': 0.001},
                'PSMC': {'base_price': 609.00,  'volatility': 0.025, 'trend': 0.001},
                'MTL':  {'base_price': 531.83,  'volatility': 0.02,  'trend': 0.001},
                'AGTL': {'base_price': 353.09,  'volatility': 0.022, 'trend': 0.001},
                'HCAR': {'base_price': 206.27,  'volatility': 0.022, 'trend': 0.001},
                'GHNI': {'base_price': 792.36,  'volatility': 0.025, 'trend': 0.001},
                'LOADS':{'base_price': 13.30,   'volatility': 0.028, 'trend': 0.001},
                # --- Pharmaceuticals ---
                'SEARL': {'base_price': 86.27,  'volatility': 0.02,  'trend': 0.0015},
                'GLAXO': {'base_price': 351.74, 'volatility': 0.018, 'trend': 0.001},
                'AGP':   {'base_price': 186.53, 'volatility': 0.025, 'trend': 0.001},
                'FEROZ': {'base_price': 375.61, 'volatility': 0.02,  'trend': 0.001},
                'HINOON':{'base_price': 925.00, 'volatility': 0.022, 'trend': 0.001},
                'GATM':  {'base_price': 22.01,  'volatility': 0.022, 'trend': 0.001},
                # --- Technology ---
                'TRG':    {'base_price': 51.97,  'volatility': 0.03,  'trend': 0.002},
                'NETSOL': {'base_price': 124.72, 'volatility': 0.028, 'trend': 0.002},
                'SYS':    {'base_price': 147.75, 'volatility': 0.025, 'trend': 0.002},
                'TELE':   {'base_price': 8.02,   'volatility': 0.04,  'trend': 0.001},
                'WTL':    {'base_price': 1.29,   'volatility': 0.045, 'trend': 0.001},
                # --- Steel ---
                'ASTL':  {'base_price': 15.67, 'volatility': 0.028, 'trend': 0.001},
                'ISL':   {'base_price': 74.55, 'volatility': 0.025, 'trend': 0.001},
                'MUGHAL':{'base_price': 70.36, 'volatility': 0.027, 'trend': 0.001},
                # --- Textile ---
                'NML':  {'base_price': 137.58,  'volatility': 0.022, 'trend': 0.001},
                'NCL':  {'base_price': 38.00,   'volatility': 0.025, 'trend': 0.001},
                'KTML': {'base_price': 45.82,   'volatility': 0.03,  'trend': 0.001},
                'SRVI': {'base_price': 1687.45, 'volatility': 0.02,  'trend': 0.001},
                # --- Glass & Allied ---
                'TGL':  {'base_price': 160.35, 'volatility': 0.025, 'trend': 0.001},
                'GATI': {'base_price': 86.30,  'volatility': 0.028, 'trend': 0.001},
                # --- Transport ---
                'PNSC': {'base_price': 529.27, 'volatility': 0.022, 'trend': 0.001},
                # --- Real Estate ---
                'JVDC': {'base_price': 127.27, 'volatility': 0.03,  'trend': 0.001},
                # --- Miscellaneous ---
                'JSCL': {'base_price': 18.20,  'volatility': 0.03,  'trend': 0.001},
            }
        }

    def get_stock_data(self, ticker, period='1y', interval='1d', start_date=None, end_date=None):
        """Get stock data instantly without external API calls"""
        cache_key = f"{ticker}_{period}_{interval}_{start_date}_{end_date}"

        if cache_key in self.stock_data_cache:
            return self.stock_data_cache[cache_key]

        # All stocks are PSX
        stock_info = self.market_data['PSX'].get(ticker, {
            'base_price': 100.0, 'volatility': 0.02, 'trend': 0.001
        })

        # Generate date range
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = datetime.now()
            if period == '1y':
                start_dt = end_dt - timedelta(days=365)
            elif period == '6mo':
                start_dt = end_dt - timedelta(days=180)
            elif period == '3mo':
                start_dt = end_dt - timedelta(days=90)
            elif period == '1mo':
                start_dt = end_dt - timedelta(days=30)
            else:
                start_dt = end_dt - timedelta(days=7)

        # Generate timestamps
        if interval == '1h':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
        elif interval == '1d':
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')

        # Generate realistic price data — seed from ticker for consistency
        ticker_seed = abs(hash(ticker)) % (2**32)
        prices = self._generate_price_series(
            stock_info['base_price'],
            stock_info['volatility'],
            stock_info['trend'],
            len(timestamps),
            seed=ticker_seed
        )

        # Create DataFrame — use seeded rng for OHLC spreads and volume
        vol_rng = np.random.default_rng(ticker_seed + 1)
        n = len(prices)
        highs = [prices[i] * (1 + abs(float(vol_rng.normal(0, 0.01)))) for i in range(n)]
        lows  = [prices[i] * (1 - abs(float(vol_rng.normal(0, 0.01)))) for i in range(n)]
        vols  = [int(vol_rng.uniform(1000000, 10000000)) for _ in range(n)]
        data = pd.DataFrame({
            'Open': prices,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': vols,
        }, index=timestamps)

        # Add market info
        data.attrs = {
            'currency': 'PKR',
            'market': 'PSX',
            'ticker': ticker
        }

        # Cache the result
        self.stock_data_cache[cache_key] = data
        return data

    def _generate_price_series(self, base_price, volatility, trend, length, seed=42):
        """Generate realistic price series using geometric Brownian motion — seeded for consistency"""
        rng = np.random.default_rng(seed)
        prices = [base_price]

        for i in range(1, length):
            change = rng.normal(trend, volatility)
            new_price = prices[-1] * (1 + change)
            new_price = max(new_price, base_price * 0.1)
            prices.append(new_price)

        return prices

    def get_stock_info(self, ticker):
        """Get current stock information — deterministic per ticker"""
        stock_info = self.market_data['PSX'].get(ticker, {
            'base_price': 100.0, 'volatility': 0.02, 'trend': 0.001
        })

        rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
        current_price = stock_info['base_price']
        change_percent = float(rng.normal(0, 2))

        return {
            'current_price': current_price,
            'change': change_percent,
            'volume': int(rng.uniform(1000000, 10000000)),
            'market_cap': float(current_price * rng.uniform(1e6, 1e9)),
            'high_52w': float(current_price * (1 + rng.uniform(0.1, 0.3))),
            'low_52w': float(current_price * (1 - rng.uniform(0.1, 0.3))),
            'currency': 'PKR'
        }

    def get_market_overview(self):
        """Get instant PSX market overview"""
        overview = {
            'PSX': {
                'total_stocks': len(self.market_data['PSX']),
                'top_gainers': [],
                'top_losers': [],
                'most_active': [],
                'market_sentiment': 'neutral'
            }
        }

        # Generate market data
        market_data = []
        for ticker in list(self.market_data['PSX'].keys())[:10]:
            stock_info = self.get_stock_info(ticker)
            market_data.append({
                'ticker': ticker,
                'current_price': stock_info['current_price'],
                'change_percent': stock_info['change'],
                'volume': stock_info['volume']
            })

        if market_data:
            sorted_by_change = sorted(market_data, key=lambda x: x['change_percent'], reverse=True)
            overview['PSX']['top_gainers'] = sorted_by_change[:3]
            overview['PSX']['top_losers'] = sorted_by_change[-3:]

            sorted_by_volume = sorted(market_data, key=lambda x: x['volume'], reverse=True)
            overview['PSX']['most_active'] = sorted_by_volume[:3]

        return overview

# Global instance
fast_data_service = FastDataService()
