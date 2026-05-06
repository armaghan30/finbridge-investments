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
        self.market_data = {
            'PSX': {
                'OGDC': {'base_price': 85.0, 'volatility': 0.025, 'trend': 0.001},
                'PPL': {'base_price': 65.0, 'volatility': 0.022, 'trend': 0.0008},
                'MCB': {'base_price': 180.0, 'volatility': 0.018, 'trend': 0.001},
                'UBL': {'base_price': 120.0, 'volatility': 0.02, 'trend': 0.0012},
                'HBL': {'base_price': 95.0, 'volatility': 0.019, 'trend': 0.001},
                'LUCK': {'base_price': 550.0, 'volatility': 0.025, 'trend': 0.0015},
                'ENGRO': {'base_price': 320.0, 'volatility': 0.022, 'trend': 0.001},
                'FFC': {'base_price': 140.0, 'volatility': 0.02, 'trend': 0.001},
                'EFERT': {'base_price': 75.0, 'volatility': 0.023, 'trend': 0.0012},
                'ATRL': {'base_price': 380.0, 'volatility': 0.026, 'trend': 0.0015},
                'PSO': {'base_price': 220.0, 'volatility': 0.024, 'trend': 0.001},
                'SHEL': {'base_price': 100.0, 'volatility': 0.02, 'trend': 0.0008},
                'NESTLE': {'base_price': 6500.0, 'volatility': 0.015, 'trend': 0.001},
                'UNILEVER': {'base_price': 2800.0, 'volatility': 0.018, 'trend': 0.001},
                'COLGATE': {'base_price': 2200.0, 'volatility': 0.016, 'trend': 0.0008},
                'PAKT': {'base_price': 850.0, 'volatility': 0.02, 'trend': 0.001},
                'ICI': {'base_price': 780.0, 'volatility': 0.022, 'trend': 0.0012},
                'FFBL': {'base_price': 25.0, 'volatility': 0.03, 'trend': 0.001},
                'DCL': {'base_price': 12.0, 'volatility': 0.035, 'trend': 0.001},
                'DGKC': {'base_price': 68.0, 'volatility': 0.028, 'trend': 0.001},
                'HUBC': {'base_price': 105.0, 'volatility': 0.02, 'trend': 0.001},
                'KEL': {'base_price': 4.5, 'volatility': 0.04, 'trend': 0.001},
                'MEBL': {'base_price': 220.0, 'volatility': 0.018, 'trend': 0.0015},
                'BAHL': {'base_price': 85.0, 'volatility': 0.019, 'trend': 0.001},
                'ABL': {'base_price': 75.0, 'volatility': 0.02, 'trend': 0.001},
                'BAFL': {'base_price': 55.0, 'volatility': 0.022, 'trend': 0.001},
                'MARI': {'base_price': 1800.0, 'volatility': 0.025, 'trend': 0.002},
                'POL': {'base_price': 450.0, 'volatility': 0.024, 'trend': 0.001},
                'PIOC': {'base_price': 95.0, 'volatility': 0.028, 'trend': 0.001},
                'MLCF': {'base_price': 42.0, 'volatility': 0.032, 'trend': 0.001},
                'FCCL': {'base_price': 22.0, 'volatility': 0.03, 'trend': 0.001},
                'KOHC': {'base_price': 180.0, 'volatility': 0.025, 'trend': 0.001},
                'INDU': {'base_price': 1400.0, 'volatility': 0.022, 'trend': 0.001},
                'PSMC': {'base_price': 280.0, 'volatility': 0.025, 'trend': 0.001},
                'MTL': {'base_price': 850.0, 'volatility': 0.02, 'trend': 0.001},
                'AGTL': {'base_price': 520.0, 'volatility': 0.022, 'trend': 0.001},
                'SEARL': {'base_price': 650.0, 'volatility': 0.02, 'trend': 0.0015},
                'GLAXO': {'base_price': 180.0, 'volatility': 0.018, 'trend': 0.001},
                'AGP': {'base_price': 95.0, 'volatility': 0.025, 'trend': 0.001},
                'GATM': {'base_price': 420.0, 'volatility': 0.022, 'trend': 0.001},
                # KSE-100 Additional Oil & Gas
                'APL': {'base_price': 380.0, 'volatility': 0.022, 'trend': 0.001},
                'SNGP': {'base_price': 65.0, 'volatility': 0.025, 'trend': 0.001},
                'SSGC': {'base_price': 25.0, 'volatility': 0.028, 'trend': 0.001},
                'NRL': {'base_price': 480.0, 'volatility': 0.02, 'trend': 0.001},
                'PRL': {'base_price': 32.0, 'volatility': 0.03, 'trend': 0.001},
                'HASCOL': {'base_price': 15.0, 'volatility': 0.04, 'trend': 0.001},
                'CNERGY': {'base_price': 12.0, 'volatility': 0.045, 'trend': 0.001},
                # KSE-100 Additional Banking
                'NBP': {'base_price': 45.0, 'volatility': 0.022, 'trend': 0.001},
                'BOP': {'base_price': 12.0, 'volatility': 0.03, 'trend': 0.001},
                'AKBL': {'base_price': 28.0, 'volatility': 0.025, 'trend': 0.001},
                'FABL': {'base_price': 35.0, 'volatility': 0.022, 'trend': 0.001},
                'JSBL': {'base_price': 9.0, 'volatility': 0.035, 'trend': 0.001},
                'SNBL': {'base_price': 18.0, 'volatility': 0.028, 'trend': 0.001},
                'BOK': {'base_price': 22.0, 'volatility': 0.025, 'trend': 0.001},
                # KSE-100 Additional Cement
                'CHCC': {'base_price': 340.0, 'volatility': 0.025, 'trend': 0.001},
                'ACPL': {'base_price': 280.0, 'volatility': 0.022, 'trend': 0.001},
                'BWCL': {'base_price': 120.0, 'volatility': 0.028, 'trend': 0.001},
                'GWLC': {'base_price': 45.0, 'volatility': 0.032, 'trend': 0.001},
                # KSE-100 Additional Fertilizer
                'FATIMA': {'base_price': 38.0, 'volatility': 0.025, 'trend': 0.001},
                # KSE-100 Additional Power
                'KAPCO': {'base_price': 55.0, 'volatility': 0.022, 'trend': 0.001},
                'NCPL': {'base_price': 25.0, 'volatility': 0.028, 'trend': 0.001},
                'PKGP': {'base_price': 18.0, 'volatility': 0.03, 'trend': 0.001},
                'TSPL': {'base_price': 42.0, 'volatility': 0.025, 'trend': 0.001},
                'EPQL': {'base_price': 45.0, 'volatility': 0.022, 'trend': 0.001},
                # KSE-100 Food & Consumer
                'QUICE': {'base_price': 85.0, 'volatility': 0.025, 'trend': 0.001},
                'FFL': {'base_price': 22.0, 'volatility': 0.035, 'trend': 0.001},
                'NATF': {'base_price': 550.0, 'volatility': 0.018, 'trend': 0.001},
                # KSE-100 Chemicals
                'LOTTE': {'base_price': 38.0, 'volatility': 0.028, 'trend': 0.001},
                'INIL': {'base_price': 185.0, 'volatility': 0.022, 'trend': 0.001},
                'CEPB': {'base_price': 155.0, 'volatility': 0.025, 'trend': 0.001},
                # KSE-100 Automobile
                'HCAR': {'base_price': 420.0, 'volatility': 0.022, 'trend': 0.001},
                'GHNI': {'base_price': 320.0, 'volatility': 0.025, 'trend': 0.001},
                # KSE-100 Pharmaceuticals
                'FEROZ': {'base_price': 290.0, 'volatility': 0.02, 'trend': 0.001},
                'HINOON': {'base_price': 145.0, 'volatility': 0.022, 'trend': 0.001},
                # KSE-100 Technology
                'TRG': {'base_price': 88.0, 'volatility': 0.03, 'trend': 0.002},
                'NETSOL': {'base_price': 145.0, 'volatility': 0.028, 'trend': 0.002},
                'SYS': {'base_price': 680.0, 'volatility': 0.025, 'trend': 0.002},
                'TELE': {'base_price': 8.0, 'volatility': 0.04, 'trend': 0.001},
                # KSE-100 Steel
                'ASTL': {'base_price': 85.0, 'volatility': 0.028, 'trend': 0.001},
                'ISL': {'base_price': 150.0, 'volatility': 0.025, 'trend': 0.001},
                'MUGHAL': {'base_price': 120.0, 'volatility': 0.027, 'trend': 0.001},
                # KSE-100 Textile
                'NML': {'base_price': 85.0, 'volatility': 0.022, 'trend': 0.001},
                'NCL': {'base_price': 65.0, 'volatility': 0.025, 'trend': 0.001},
                'KTML': {'base_price': 35.0, 'volatility': 0.03, 'trend': 0.001},
                # Miscellaneous
                'TGL': {'base_price': 95.0, 'volatility': 0.025, 'trend': 0.001},
                'JSCL': {'base_price': 35.0, 'volatility': 0.03, 'trend': 0.001},
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
