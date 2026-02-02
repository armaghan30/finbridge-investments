from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import random
from datetime import datetime, timedelta
from scipy.stats import norm

try:
    import talib
except ImportError:
    talib = None

from fast_data_service import fast_data_service

api_bp = Blueprint('api', __name__, url_prefix='/api')

# ---------- Cache ----------
_cache = {}
CACHE_TTL = 300

def _get(key):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None

def _set(key, data):
    _cache[key] = (time.time(), data)

# ---------- Market Config (PSX Only) ----------
MARKETS = {
    'PSX': {
        'name': 'Pakistan Stock Exchange',
        'stocks': {
            'OGDC': 'Oil & Gas Development Co.', 'PPL': 'Pakistan Petroleum Ltd.',
            'MCB': 'Muslim Commercial Bank', 'UBL': 'United Bank Ltd.',
            'HBL': 'Habib Bank Ltd.', 'LUCK': 'Lucky Cement Ltd.',
            'ENGRO': 'Engro Corp.', 'FFC': 'Fauji Fertilizer Co.',
            'EFERT': 'Engro Fertilizers Ltd.', 'ATRL': 'Attock Refinery Ltd.',
            'PSO': 'Pakistan State Oil', 'SHEL': 'Shell Pakistan Ltd.',
            'NESTLE': 'Nestle Pakistan Ltd.', 'UNILEVER': 'Unilever Pakistan Ltd.',
            'COLGATE': 'Colgate Palmolive Pakistan', 'PAKT': 'Pakistan Tobacco Co.',
            'ICI': 'ICI Pakistan Ltd.', 'FFBL': 'Fauji Fertilizer Bin Qasim',
            'DCL': 'Dewan Cement Ltd.', 'DGKC': 'D.G. Khan Cement Co.',
            'HUBC': 'Hub Power Co.', 'KEL': 'K-Electric Ltd.',
            'MEBL': 'Meezan Bank Ltd.', 'BAHL': 'Bank AL Habib Ltd.',
            'ABL': 'Allied Bank Ltd.', 'BAFL': 'Bank Alfalah Ltd.',
            'MARI': 'Mari Petroleum Co.', 'POL': 'Pakistan Oilfields Ltd.',
            'PIOC': 'Pioneer Cement Ltd.', 'MLCF': 'Maple Leaf Cement',
            'FCCL': 'Fauji Cement Co.', 'KOHC': 'Kohat Cement Co.',
            'INDU': 'Indus Motor Co.', 'PSMC': 'Pak Suzuki Motor Co.',
            'MTL': 'Millat Tractors Ltd.', 'AGTL': 'Al-Ghazi Tractors Ltd.',
            'SEARL': 'Searle Company Ltd.', 'GLAXO': 'GlaxoSmithKline Pakistan',
            'AGP': 'AGP Ltd.', 'GATM': 'Gatron Industries Ltd.',
        }
    }
}

SECTOR_MAP = {
    # Oil & Gas
    'OGDC': 'Oil & Gas', 'PPL': 'Oil & Gas', 'PSO': 'Oil & Gas',
    'SHEL': 'Oil & Gas', 'ATRL': 'Oil & Gas', 'MARI': 'Oil & Gas', 'POL': 'Oil & Gas',
    # Banking
    'MCB': 'Banking', 'UBL': 'Banking', 'HBL': 'Banking',
    'MEBL': 'Banking', 'BAHL': 'Banking', 'ABL': 'Banking', 'BAFL': 'Banking',
    # Cement
    'LUCK': 'Cement', 'DGKC': 'Cement', 'DCL': 'Cement',
    'PIOC': 'Cement', 'MLCF': 'Cement', 'FCCL': 'Cement', 'KOHC': 'Cement',
    # Fertilizer
    'FFC': 'Fertilizer', 'EFERT': 'Fertilizer', 'FFBL': 'Fertilizer', 'ENGRO': 'Fertilizer',
    # Power
    'HUBC': 'Power', 'KEL': 'Power',
    # Food & Consumer
    'NESTLE': 'Food & Consumer', 'UNILEVER': 'Food & Consumer',
    'COLGATE': 'Food & Consumer', 'PAKT': 'Food & Consumer',
    # Chemicals
    'ICI': 'Chemicals', 'GATM': 'Chemicals',
    # Automobile
    'INDU': 'Automobile', 'PSMC': 'Automobile', 'MTL': 'Automobile', 'AGTL': 'Automobile',
    # Pharmaceuticals
    'SEARL': 'Pharmaceuticals', 'GLAXO': 'Pharmaceuticals', 'AGP': 'Pharmaceuticals',
}

# ---------- PSX Ticker Resolution ----------
# PSX stocks on yfinance use the .KA suffix (Karachi Stock Exchange)
PSX_SUFFIXES = ['.KA', '.PK', '.IS']

def _resolve_yf_ticker(ticker):
    """Resolve a ticker to a yfinance-compatible symbol.
    PSX stocks need a suffix like .KA for real-time data."""
    if ticker in MARKETS.get('PSX', {}).get('stocks', {}):
        return f"{ticker}.KA"  # Primary PSX suffix on yfinance
    return ticker

def _try_psx_fetch(ticker, fetch_fn):
    """Try multiple PSX suffixes until we get data."""
    for suffix in PSX_SUFFIXES:
        try:
            yf_ticker = f"{ticker}{suffix}"
            result = fetch_fn(yf_ticker)
            if result is not None:
                return result
        except Exception:
            continue
    return None

# ---------- Helpers ----------
def _all_tickers():
    tickers = {}
    for mkt in MARKETS.values():
        tickers.update(mkt['stocks'])
    return tickers

def _find_market(ticker):
    for mkt_name, mkt in MARKETS.items():
        if ticker in mkt['stocks']:
            return mkt_name
    return 'PSX'

def _fetch_stock_data(ticker, period='1y', interval='1d', start_date=None, end_date=None):
    dsuffix = f"_{start_date}_{end_date}" if start_date and end_date else ""
    ckey = f"sd_{ticker}_{period}_{interval}{dsuffix}"
    cached = _get(ckey)
    if cached is not None:
        return cached

    is_psx = ticker in MARKETS.get('PSX', {}).get('stocks', {})

    def _do_fetch(yf_sym):
        time.sleep(random.uniform(0.3, 0.8))
        stock = yf.Ticker(yf_sym)
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            data = stock.history(period=period, interval=interval)
        if data is not None and not data.empty and len(data) > 2:
            return data
        return None

    try:
        if is_psx:
            data = _try_psx_fetch(ticker, _do_fetch)
        else:
            data = _do_fetch(ticker)

        if data is not None:
            _set(ckey, data)
            return data
    except Exception:
        pass

    # fallback to demo data
    data = fast_data_service.get_stock_data(ticker, period, interval, start_date, end_date)
    _set(ckey, data)
    return data

def _get_stock_info(ticker):
    ckey = f"info_{ticker}"
    cached = _get(ckey)
    if cached is not None:
        return cached

    is_psx = ticker in MARKETS.get('PSX', {}).get('stocks', {})

    def _do_info(yf_sym):
        stock = yf.Ticker(yf_sym)
        info = stock.info
        if info and ('currentPrice' in info or 'regularMarketPrice' in info):
            # Normalize field names
            if 'regularMarketPrice' in info and 'currentPrice' not in info:
                info['currentPrice'] = info['regularMarketPrice']
            if 'regularMarketPreviousClose' in info and 'previousClose' not in info:
                info['previousClose'] = info['regularMarketPreviousClose']
            return info
        return None

    try:
        if is_psx:
            info = _try_psx_fetch(ticker, _do_info)
        else:
            info = _do_info(ticker)

        if info:
            info['symbol'] = ticker
            info['exchange'] = 'PSX' if is_psx else _find_market(ticker)
            info['currency'] = 'PKR'
            _set(ckey, info)
            return info
    except Exception:
        pass

    # generate demo info
    market = _find_market(ticker)
    name = _all_tickers().get(ticker, ticker)
    base = random.uniform(20, 800)
    chg = random.uniform(-5, 5)
    info = {
        'symbol': ticker,
        'shortName': name,
        'longName': name,
        'currentPrice': round(base, 2),
        'previousClose': round(base * (1 - chg / 100), 2),
        'open': round(base * (1 + random.uniform(-0.02, 0.02)), 2),
        'dayHigh': round(base * 1.02, 2),
        'dayLow': round(base * 0.98, 2),
        'volume': random.randint(1_000_000, 80_000_000),
        'averageVolume': random.randint(5_000_000, 50_000_000),
        'marketCap': int(base * random.uniform(1e8, 1e10)),
        'trailingPE': round(random.uniform(5, 60), 2),
        'forwardPE': round(random.uniform(5, 45), 2),
        'dividendYield': round(random.uniform(0, 0.05), 4),
        'beta': round(random.uniform(0.5, 2.0), 2),
        'fiftyTwoWeekHigh': round(base * 1.3, 2),
        'fiftyTwoWeekLow': round(base * 0.7, 2),
        'fiftyDayAverage': round(base * 1.02, 2),
        'twoHundredDayAverage': round(base * 0.98, 2),
        'trailingEps': round(random.uniform(1, 20), 2),
        'priceToBook': round(random.uniform(1, 15), 2),
        'profitMargins': round(random.uniform(0.05, 0.35), 4),
        'returnOnEquity': round(random.uniform(0.05, 0.40), 4),
        'debtToEquity': round(random.uniform(10, 200), 2),
        'freeCashflow': int(random.uniform(1e8, 5e10)),
        'revenue': int(random.uniform(1e9, 4e11)),
        'revenueGrowth': round(random.uniform(-0.1, 0.4), 4),
        'earningsGrowth': round(random.uniform(-0.15, 0.5), 4),
        'sector': SECTOR_MAP.get(ticker, 'Technology'),
        'industry': 'Various',
        'exchange': market,
        'currency': 'PKR',
    }
    _set(ckey, info)
    return info

# ---------- Technical Indicators ----------
def _calc_sma(series, period):
    return series.rolling(window=period).mean()

def _calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _calc_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal
    return macd_line, signal, histogram

def _calc_bollinger(series, period=20, std_dev=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

def _calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def _calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def _calc_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def _calc_vwap(high, low, close, volume):
    tp = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tp_vol = (tp * volume).cumsum()
    return cum_tp_vol / cum_vol

# ---------- API Endpoints ----------

@api_bp.route('/search')
def search():
    q = request.args.get('q', '').upper().strip()
    if not q:
        return jsonify([])
    all_stocks = _all_tickers()
    results = []
    for sym, name in all_stocks.items():
        if q in sym or q in name.upper():
            results.append({
                'symbol': sym,
                'name': name,
                'market': _find_market(sym),
                'sector': SECTOR_MAP.get(sym, ''),
            })
    results.sort(key=lambda x: (0 if x['symbol'].startswith(q) else 1, x['symbol']))
    return jsonify(results[:15])


@api_bp.route('/stock-data')
def stock_data():
    ticker = request.args.get('ticker', 'OGDC').upper()
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    start = request.args.get('start_date')
    end = request.args.get('end_date')

    data = _fetch_stock_data(ticker, period, interval, start, end)
    if data is None or (hasattr(data, 'empty') and data.empty):
        return jsonify({'error': 'No data available'}), 404

    records = []
    for idx, row in data.iterrows():
        records.append({
            'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
            'open': round(float(row.get('Open', 0)), 2),
            'high': round(float(row.get('High', 0)), 2),
            'low': round(float(row.get('Low', 0)), 2),
            'close': round(float(row.get('Close', 0)), 2),
            'volume': int(row.get('Volume', 0)),
        })
    return jsonify({'ticker': ticker, 'data': records})


@api_bp.route('/stock-info')
def stock_info():
    ticker = request.args.get('ticker', 'OGDC').upper()
    info = _get_stock_info(ticker)
    return jsonify(info)


@api_bp.route('/stock-quote')
def stock_quote():
    ticker = request.args.get('ticker', 'OGDC').upper()
    info = _get_stock_info(ticker)
    price = info.get('currentPrice', 0)
    prev = info.get('previousClose', price)
    change = price - prev
    change_pct = (change / prev * 100) if prev else 0
    return jsonify({
        'symbol': ticker,
        'name': info.get('shortName', ticker),
        'price': round(price, 2),
        'change': round(change, 2),
        'changePercent': round(change_pct, 2),
        'volume': info.get('volume', 0),
        'marketCap': info.get('marketCap', 0),
        'pe': info.get('trailingPE'),
        'high': info.get('dayHigh'),
        'low': info.get('dayLow'),
        'open': info.get('open'),
        'prevClose': round(prev, 2),
        'week52High': info.get('fiftyTwoWeekHigh'),
        'week52Low': info.get('fiftyTwoWeekLow'),
        'avgVolume': info.get('averageVolume'),
        'beta': info.get('beta'),
        'eps': info.get('trailingEps'),
        'dividend': info.get('dividendYield'),
        'sector': info.get('sector', SECTOR_MAP.get(ticker, '')),
        'exchange': _find_market(ticker),
    })


# ============================================================
#  INDICATOR SCORING ENGINE - 30 indicators across 7 categories
#  Score 1 = Low Risk/Best, 4 = High Risk/Worst
# ============================================================

def _score_indicator(value, low, med, medhigh, direction='lower_better'):
    """Score an indicator 1-4 based on thresholds.
    direction='lower_better': lower values = score 1 (e.g. P/E, Beta, Debt)
    direction='higher_better': higher values = score 1 (e.g. ROE, Margins, Sharpe)
    """
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if direction == 'lower_better':
        if value <= low:
            return 1
        elif value <= med:
            return 2
        elif value <= medhigh:
            return 3
        else:
            return 4
    else:  # higher_better
        if value >= low:
            return 1
        elif value >= med:
            return 2
        elif value >= medhigh:
            return 3
        else:
            return 4


def _calculate_all_indicators(info, hist_data=None):
    """Calculate all 30 indicators and scores for a stock based on the scoring table."""

    price = info.get('currentPrice', 0) or 0
    prev = info.get('previousClose', price) or price
    mcap = info.get('marketCap', 0) or 0
    pe = info.get('trailingPE', 0) or 0
    eps = info.get('trailingEps', 0) or 0
    pb = info.get('priceToBook', 0) or 0
    beta = info.get('beta', 1.0) or 1.0
    div_yield_raw = info.get('dividendYield', 0) or 0
    div_yield = div_yield_raw * 100 if div_yield_raw < 1 else div_yield_raw
    profit_margin = (info.get('profitMargins', 0) or 0) * 100
    roe = (info.get('returnOnEquity', 0) or 0) * 100
    roa = (info.get('returnOnAssets', 0) or 0) * 100
    debt_equity = info.get('debtToEquity', 0) or 0
    revenue_growth = (info.get('revenueGrowth', 0) or 0) * 100
    earnings_growth = (info.get('earningsGrowth', 0) or 0) * 100
    fcf = info.get('freeCashflow', 0) or 0
    revenue = info.get('revenue', 0) or 0
    gross_margin = (info.get('grossMargins', 0) or 0) * 100
    operating_margin = (info.get('operatingMargins', 0) or 0) * 100
    current_ratio = info.get('currentRatio', 0) or 0
    quick_ratio = info.get('quickRatio', 0) or 0
    interest_coverage = info.get('interestCoverage', 0) or 0
    ev_ebitda = info.get('enterpriseToEbitda', 0) or 0
    forward_pe = info.get('forwardPE', 0) or 0
    peg = info.get('pegRatio', 0) or 0

    # Computed metrics
    ps_ratio = (mcap / revenue) if revenue > 0 else 0
    earnings_yield = (eps / price * 100) if price > 0 else 0
    div_growth = random.uniform(-2, 15)  # Simulated
    asset_turnover = (revenue / mcap * 2) if mcap > 0 else 0

    # Historical data calculations
    change_1y = 0
    std_dev = 0
    sharpe = 0
    sortino = 0
    treynor = 0
    var_95 = 0
    if hist_data is not None and len(hist_data) > 30:
        close = hist_data['Close']
        returns = close.pct_change().dropna()
        if len(returns) > 10:
            ann_return = float(returns.mean() * 252) * 100
            ann_vol = float(returns.std() * np.sqrt(252)) * 100
            std_dev = ann_vol
            change_1y = ((float(close.iloc[-1]) / float(close.iloc[0])) - 1) * 100 if float(close.iloc[0]) > 0 else 0
            sharpe = (ann_return - 10) / ann_vol if ann_vol > 0 else 0
            neg_ret = returns[returns < 0]
            downside_std = float(neg_ret.std() * np.sqrt(252)) * 100 if len(neg_ret) > 0 else 1
            sortino = (ann_return - 10) / downside_std if downside_std > 0 else 0
            treynor = (ann_return - 10) / beta if beta > 0 else 0
            var_95 = float(np.percentile(returns, 5)) * 100

    # EPS Growth (simulated 5Y CAGR)
    eps_growth = random.uniform(-5, 25)

    # Altman Z-Score (simplified)
    altman_z = 1.2 * (current_ratio * 0.3) + 1.4 * (roe / 100) + 3.3 * (profit_margin / 100) + 0.6 * (mcap / max(debt_equity * mcap / 100, 1)) + 1.0 * asset_turnover
    altman_z = min(max(altman_z, 0.5), 5.0)

    indicators = {}

    # === BASIC ===
    indicators['marketCap'] = {
        'value': mcap, 'category': 'Basic', 'name': 'Market Capitalization',
        'display': f'PKR {mcap/1e6:,.0f}M' if mcap < 1e9 else f'PKR {mcap/1e9:,.1f}B',
        'score': _score_indicator(mcap / 1e6, 100000, 30000, 10000, 'higher_better')
    }
    indicators['dividendYield'] = {
        'value': div_yield, 'category': 'Basic', 'name': 'Dividend Yield',
        'display': f'{div_yield:.2f}%',
        'score': _score_indicator(div_yield, 6, 4, 1, 'higher_better')
    }

    # === PERFORMANCE ===
    indicators['revenueGrowth'] = {
        'value': revenue_growth, 'category': 'Performance', 'name': 'Revenue Growth',
        'display': f'{revenue_growth:.1f}%',
        'score': _score_indicator(revenue_growth, 20, 10, 4, 'higher_better')
    }
    indicators['earningsYield'] = {
        'value': earnings_yield, 'category': 'Performance', 'name': 'Earnings Yield',
        'display': f'{earnings_yield:.1f}%',
        'score': _score_indicator(earnings_yield, 12, 8, 4, 'higher_better')
    }
    indicators['divGrowth'] = {
        'value': div_growth, 'category': 'Performance', 'name': 'Dividend Yield Growth',
        'display': f'{div_growth:.1f}%',
        'score': _score_indicator(div_growth, 10, 5, 1.5, 'higher_better')
    }
    indicators['roa'] = {
        'value': roa, 'category': 'Performance', 'name': 'Return on Assets',
        'display': f'{roa:.1f}%',
        'score': _score_indicator(roa, 12, 8, 4, 'higher_better')
    }
    indicators['roe'] = {
        'value': roe, 'category': 'Performance', 'name': 'Return on Equity',
        'display': f'{roe:.1f}%',
        'score': _score_indicator(roe, 20, 15, 10, 'higher_better')
    }
    indicators['netProfitMargin'] = {
        'value': profit_margin, 'category': 'Performance', 'name': 'Net Profit Margin',
        'display': f'{profit_margin:.1f}%',
        'score': _score_indicator(profit_margin, 20, 10, 5, 'higher_better')
    }
    indicators['freeCashFlow'] = {
        'value': fcf, 'category': 'Performance', 'name': 'Free Cash Flow',
        'display': f'PKR {fcf/1e6:,.0f}M' if abs(fcf) < 1e9 else f'PKR {fcf/1e9:,.1f}B',
        'score': 1 if fcf > 0 else (3 if fcf == 0 else 4)
    }

    # === VALUATION ===
    indicators['peRatio'] = {
        'value': pe, 'category': 'Valuation', 'name': 'P/E Ratio',
        'display': f'{pe:.1f}' if pe else 'N/A',
        'score': _score_indicator(pe, 10, 15, 25, 'lower_better') if pe > 0 else None
    }
    indicators['psRatio'] = {
        'value': round(ps_ratio, 2), 'category': 'Valuation', 'name': 'P/S Ratio',
        'display': f'{ps_ratio:.2f}',
        'score': _score_indicator(ps_ratio, 1, 2, 3, 'lower_better')
    }
    indicators['pbRatio'] = {
        'value': pb, 'category': 'Valuation', 'name': 'P/B Ratio',
        'display': f'{pb:.2f}',
        'score': _score_indicator(pb, 1.0, 2.0, 3.0, 'lower_better')
    }
    indicators['evEbitda'] = {
        'value': ev_ebitda, 'category': 'Valuation', 'name': 'EV/EBITDA',
        'display': f'{ev_ebitda:.1f}' if ev_ebitda else 'N/A',
        'score': _score_indicator(ev_ebitda, 7, 10, 14, 'lower_better') if ev_ebitda > 0 else None
    }
    indicators['pegRatio'] = {
        'value': peg, 'category': 'Valuation', 'name': 'PEG Ratio',
        'display': f'{peg:.2f}' if peg else 'N/A',
        'score': _score_indicator(peg, 1, 1.5, 2.5, 'lower_better') if peg > 0 else None
    }

    # === RISK INDICATORS ===
    indicators['beta'] = {
        'value': beta, 'category': 'Risk Indicators', 'name': 'Beta',
        'display': f'{beta:.2f}',
        'score': _score_indicator(beta, 0.75, 1.0, 1.25, 'lower_better')
    }
    indicators['sharpeRatio'] = {
        'value': round(sharpe, 2), 'category': 'Risk Indicators', 'name': 'Sharpe Ratio',
        'display': f'{sharpe:.2f}',
        'score': _score_indicator(sharpe, 1.0, 0.5, 0, 'higher_better')
    }
    indicators['treynorRatio'] = {
        'value': round(treynor, 2), 'category': 'Risk Indicators', 'name': 'Treynor Ratio',
        'display': f'{treynor:.1f}%',
        'score': _score_indicator(treynor, 10, 5, 1, 'higher_better')
    }
    indicators['sortinoRatio'] = {
        'value': round(sortino, 2), 'category': 'Risk Indicators', 'name': 'Sortino Ratio',
        'display': f'{sortino:.2f}',
        'score': _score_indicator(sortino, 1.0, 0.5, 0, 'higher_better')
    }
    indicators['altmanZ'] = {
        'value': round(altman_z, 2), 'category': 'Risk Indicators', 'name': 'Altman Z-Score',
        'display': f'{altman_z:.2f}',
        'score': _score_indicator(altman_z, 3.0, 2.5, 1.8, 'higher_better')
    }
    indicators['var'] = {
        'value': round(var_95, 2), 'category': 'Risk Indicators', 'name': 'Value at Risk (VaR)',
        'display': f'{var_95:.1f}%',
        'score': _score_indicator(abs(var_95), 1, 2, 3, 'lower_better')
    }

    # === SHORT-TERM LIQUIDITY ===
    indicators['currentRatio'] = {
        'value': current_ratio, 'category': 'Short-Term Liquidity', 'name': 'Current Ratio',
        'display': f'{current_ratio:.2f}',
        'score': _score_indicator(current_ratio, 1.5, 1.0, 0.5, 'higher_better')
    }
    indicators['quickRatio'] = {
        'value': quick_ratio, 'category': 'Short-Term Liquidity', 'name': 'Quick Ratio',
        'display': f'{quick_ratio:.2f}',
        'score': _score_indicator(quick_ratio, 1.5, 1.0, 0.5, 'higher_better')
    }

    # === LEVERAGE ===
    indicators['debtToEquity'] = {
        'value': debt_equity, 'category': 'Leverage', 'name': 'Debt-to-Equity Ratio',
        'display': f'{debt_equity:.1f}%',
        'score': _score_indicator(debt_equity / 100, 0.5, 1.0, 1.5, 'lower_better')
    }
    indicators['interestCoverage'] = {
        'value': interest_coverage, 'category': 'Leverage', 'name': 'Interest Coverage Ratio',
        'display': f'{interest_coverage:.1f}',
        'score': _score_indicator(interest_coverage, 5, 3, 1, 'higher_better')
    }

    # === EFFICIENCY ===
    indicators['assetTurnover'] = {
        'value': round(asset_turnover, 2), 'category': 'Efficiency', 'name': 'Asset Turnover',
        'display': f'{asset_turnover:.2f}',
        'score': _score_indicator(asset_turnover, 1.5, 1.0, 0.5, 'higher_better')
    }
    indicators['grossMargin'] = {
        'value': gross_margin, 'category': 'Efficiency', 'name': 'Gross Margin',
        'display': f'{gross_margin:.1f}%',
        'score': _score_indicator(gross_margin, 40, 30, 20, 'higher_better')
    }
    indicators['operatingMargin'] = {
        'value': operating_margin, 'category': 'Efficiency', 'name': 'Operating Margin',
        'display': f'{operating_margin:.1f}%',
        'score': _score_indicator(operating_margin, 25, 15, 5, 'higher_better')
    }

    # === GROWTH ===
    indicators['epsGrowth'] = {
        'value': round(eps_growth, 1), 'category': 'Growth', 'name': 'EPS Growth (5Y)',
        'display': f'{eps_growth:.1f}%',
        'score': _score_indicator(eps_growth, 15, 10, 5, 'higher_better')
    }
    indicators['revenueGrowth5Y'] = {
        'value': round(revenue_growth * 0.8, 1), 'category': 'Growth', 'name': 'Revenue Growth (5Y)',
        'display': f'{revenue_growth * 0.8:.1f}%',
        'score': _score_indicator(revenue_growth * 0.8, 12, 8, 3, 'higher_better')
    }

    # === VOLATILITY ===
    indicators['priceChange1Y'] = {
        'value': round(change_1y, 1), 'category': 'Volatility', 'name': 'Price Change (1Y)',
        'display': f'{change_1y:+.1f}%',
        'score': 1 if 10 <= change_1y <= 30 else (2 if 0 <= change_1y <= 10 or change_1y > 30 else (3 if -10 <= change_1y < 0 else 4))
    }
    indicators['stdDeviation'] = {
        'value': round(std_dev, 1), 'category': 'Volatility', 'name': 'Standard Deviation',
        'display': f'{std_dev:.1f}%',
        'score': _score_indicator(std_dev, 10, 20, 30, 'lower_better')
    }

    # Calculate overall score
    scores = [v['score'] for v in indicators.values() if v['score'] is not None]
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    # Category averages
    categories = {}
    for ind in indicators.values():
        cat = ind['category']
        if cat not in categories:
            categories[cat] = []
        if ind['score'] is not None:
            categories[cat].append(ind['score'])

    category_scores = {}
    for cat, cat_scores in categories.items():
        category_scores[cat] = round(sum(cat_scores) / len(cat_scores), 2) if cat_scores else None

    return {
        'indicators': indicators,
        'overallScore': avg_score,
        'categoryScores': category_scores,
        'totalIndicators': len(scores),
    }


def _score_label(score):
    """Convert numeric score to risk label."""
    if score is None:
        return 'N/A'
    if score <= 1.5:
        return 'Low Risk'
    elif score <= 2.5:
        return 'Medium Risk'
    elif score <= 3.25:
        return 'Medium-High Risk'
    else:
        return 'High Risk'


@api_bp.route('/screener', methods=['POST'])
def screener():
    filters = request.get_json() or {}
    market = filters.get('market', 'PSX')
    sort_by = filters.get('sort_by', 'marketCap')
    sort_dir = filters.get('sort_dir', 'desc')
    page = int(filters.get('page', 1))
    per_page = int(filters.get('per_page', 25))
    include_indicators = filters.get('include_indicators', True)

    stocks = MARKETS.get(market, MARKETS['PSX'])['stocks']
    results = []

    for ticker, name in stocks.items():
        info = _get_stock_info(ticker)
        price = info.get('currentPrice', 0)
        prev = info.get('previousClose', price)
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0
        mcap = info.get('marketCap', 0)
        pe = info.get('trailingPE', 0)
        div_yield = (info.get('dividendYield') or 0) * 100
        beta = info.get('beta', 1.0)
        vol = info.get('volume', 0)
        sector = info.get('sector', SECTOR_MAP.get(ticker, ''))

        # Apply filters
        if 'price_min' in filters and price < float(filters['price_min']):
            continue
        if 'price_max' in filters and price > float(filters['price_max']):
            continue
        if 'market_cap' in filters and filters['market_cap']:
            mc_ranges = {
                'micro': (0, 3e8), 'small': (3e8, 2e9), 'mid': (2e9, 1e10),
                'large': (1e10, 1e11), 'mega': (1e11, float('inf'))
            }
            lo, hi = mc_ranges.get(filters['market_cap'], (0, float('inf')))
            if not (lo <= mcap <= hi):
                continue
        if 'pe_min' in filters and pe < float(filters['pe_min']):
            continue
        if 'pe_max' in filters and pe > float(filters['pe_max']):
            continue
        if 'div_min' in filters and div_yield < float(filters['div_min']):
            continue
        if 'div_max' in filters and div_yield > float(filters['div_max']):
            continue
        if 'beta_min' in filters and beta < float(filters['beta_min']):
            continue
        if 'beta_max' in filters and beta > float(filters['beta_max']):
            continue
        if 'sector' in filters and filters['sector'] and sector != filters['sector']:
            continue

        row = {
            'symbol': ticker,
            'name': name,
            'price': round(price, 2),
            'change': round(change, 2),
            'changePercent': round(change_pct, 2),
            'marketCap': mcap,
            'pe': round(pe, 2) if pe else None,
            'dividendYield': round(div_yield, 2),
            'beta': round(beta, 2),
            'volume': vol,
            'sector': sector,
            'exchange': market,
        }

        # Calculate indicator scores
        if include_indicators:
            hist_data = _fetch_stock_data(ticker, '1y', '1d')
            scoring = _calculate_all_indicators(info, hist_data)
            row['overallScore'] = scoring['overallScore']
            row['scoreLabel'] = _score_label(scoring['overallScore'])
            row['categoryScores'] = scoring['categoryScores']
            row['indicators'] = scoring['indicators']
            row['totalIndicators'] = scoring['totalIndicators']

        results.append(row)

    # Sort - support sorting by overallScore
    reverse = sort_dir == 'desc'
    if sort_by == 'overallScore':
        results.sort(key=lambda x: x.get('overallScore', 99) or 99, reverse=not reverse)
    else:
        results.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    return jsonify({
        'results': results[start:end],
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
    })


@api_bp.route('/market-overview')
def market_overview():
    market = request.args.get('market', 'PSX')
    stocks = MARKETS.get(market, MARKETS['PSX'])['stocks']
    items = []
    for ticker, name in stocks.items():
        info = _get_stock_info(ticker)
        price = info.get('currentPrice', 0)
        prev = info.get('previousClose', price)
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0
        items.append({
            'symbol': ticker, 'name': name,
            'price': round(price, 2),
            'change': round(change, 2),
            'changePercent': round(change_pct, 2),
            'volume': info.get('volume', 0),
            'marketCap': info.get('marketCap', 0),
        })
    gainers = sorted(items, key=lambda x: x['changePercent'], reverse=True)[:10]
    losers = sorted(items, key=lambda x: x['changePercent'])[:10]
    active = sorted(items, key=lambda x: x['volume'], reverse=True)[:10]
    return jsonify({
        'market': market,
        'gainers': gainers,
        'losers': losers,
        'mostActive': active,
        'totalStocks': len(items),
    })


@api_bp.route('/technical-indicators')
def technical_indicators():
    ticker = request.args.get('ticker', 'OGDC').upper()
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')

    data = _fetch_stock_data(ticker, period, interval)
    if data is None or (hasattr(data, 'empty') and data.empty):
        return jsonify({'error': 'No data'}), 404

    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    rsi = _calc_rsi(close, 14)
    macd_line, signal, hist = _calc_macd(close)
    bb_upper, bb_middle, bb_lower = _calc_bollinger(close)
    atr = _calc_atr(high, low, close)
    stoch_k, stoch_d = _calc_stochastic(high, low, close)
    obv = _calc_obv(close, volume)
    vwap = _calc_vwap(high, low, close, volume)

    dates = [idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx) for idx in data.index]

    def safe_list(s):
        return [None if pd.isna(v) else round(float(v), 4) for v in s]

    return jsonify({
        'ticker': ticker,
        'dates': dates,
        'close': safe_list(close),
        'sma20': safe_list(_calc_sma(close, 20)),
        'sma50': safe_list(_calc_sma(close, 50)),
        'sma200': safe_list(_calc_sma(close, 200)),
        'ema12': safe_list(_calc_ema(close, 12)),
        'ema26': safe_list(_calc_ema(close, 26)),
        'rsi': safe_list(rsi),
        'macd': safe_list(macd_line),
        'macd_signal': safe_list(signal),
        'macd_histogram': safe_list(hist),
        'bb_upper': safe_list(bb_upper),
        'bb_middle': safe_list(bb_middle),
        'bb_lower': safe_list(bb_lower),
        'atr': safe_list(atr),
        'stoch_k': safe_list(stoch_k),
        'stoch_d': safe_list(stoch_d),
        'obv': safe_list(obv),
        'vwap': safe_list(vwap),
    })


@api_bp.route('/risk-metrics')
def risk_metrics():
    ticker = request.args.get('ticker', 'OGDC').upper()
    data = _fetch_stock_data(ticker, '1y', '1d')
    if data is None or len(data) < 30:
        return jsonify({'error': 'Insufficient data'}), 400

    returns = data['Close'].pct_change().dropna()
    ann_vol = float(returns.std() * np.sqrt(252))
    ann_return = float(returns.mean() * 252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    neg_returns = returns[returns < 0]
    downside_dev = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 else 0.001
    sortino = ann_return / downside_dev if downside_dev > 0 else 0
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    var_95 = float(np.percentile(returns, 5))
    var_99 = float(np.percentile(returns, 1))
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())

    # Beta vs SPY
    try:
        spy_data = _fetch_stock_data('SPY', '1y', '1d')
        spy_returns = spy_data['Close'].pct_change().dropna()
        min_len = min(len(returns), len(spy_returns))
        if min_len > 10:
            cov = np.cov(returns[-min_len:], spy_returns[-min_len:])
            beta = float(cov[0][1] / cov[1][1]) if cov[1][1] != 0 else 1.0
            corr = float(np.corrcoef(returns[-min_len:], spy_returns[-min_len:])[0][1])
        else:
            beta, corr = 1.0, 0.5
    except Exception:
        beta, corr = 1.0, 0.5

    return jsonify({
        'ticker': ticker,
        'annualizedReturn': round(ann_return * 100, 2),
        'annualizedVolatility': round(ann_vol * 100, 2),
        'sharpeRatio': round(sharpe, 3),
        'sortinoRatio': round(sortino, 3),
        'calmarRatio': round(calmar, 3),
        'maxDrawdown': round(max_dd * 100, 2),
        'var95': round(var_95 * 100, 2),
        'var99': round(var_99 * 100, 2),
        'skewness': round(skew, 3),
        'kurtosis': round(kurt, 3),
        'beta': round(beta, 3),
        'correlation': round(corr, 3),
    })


@api_bp.route('/financials')
def financials():
    ticker = request.args.get('ticker', 'OGDC').upper()
    info = _get_stock_info(ticker)
    revenue = info.get('revenue', 0)
    mcap = info.get('marketCap', 0)

    return jsonify({
        'ticker': ticker,
        'overview': {
            'marketCap': mcap,
            'revenue': revenue,
            'revenueGrowth': info.get('revenueGrowth', 0),
            'earningsGrowth': info.get('earningsGrowth', 0),
            'profitMargin': info.get('profitMargins', 0),
            'roe': info.get('returnOnEquity', 0),
            'debtToEquity': info.get('debtToEquity', 0),
            'freeCashflow': info.get('freeCashflow', 0),
            'eps': info.get('trailingEps', 0),
            'pe': info.get('trailingPE', 0),
            'forwardPE': info.get('forwardPE', 0),
            'priceToBook': info.get('priceToBook', 0),
            'dividendYield': info.get('dividendYield', 0),
        },
        'incomeStatement': _generate_income_statement(ticker, revenue),
        'balanceSheet': _generate_balance_sheet(ticker, mcap),
    })


def _generate_income_statement(ticker, base_revenue):
    years = [2024, 2023, 2022, 2021]
    stmts = []
    rev = base_revenue or random.uniform(1e9, 1e11)
    for yr in years:
        cogs = rev * random.uniform(0.4, 0.65)
        gross = rev - cogs
        opex = gross * random.uniform(0.3, 0.6)
        op_income = gross - opex
        net_income = op_income * random.uniform(0.7, 0.9)
        stmts.append({
            'year': yr, 'revenue': int(rev), 'costOfRevenue': int(cogs),
            'grossProfit': int(gross), 'operatingExpenses': int(opex),
            'operatingIncome': int(op_income), 'netIncome': int(net_income),
        })
        rev *= random.uniform(0.85, 1.0)
    return stmts


def _generate_balance_sheet(ticker, mcap):
    years = [2024, 2023, 2022, 2021]
    sheets = []
    assets = mcap * random.uniform(0.5, 1.5) if mcap else random.uniform(1e9, 1e11)
    for yr in years:
        liabilities = assets * random.uniform(0.3, 0.65)
        equity = assets - liabilities
        cash = assets * random.uniform(0.05, 0.2)
        debt = liabilities * random.uniform(0.3, 0.7)
        sheets.append({
            'year': yr, 'totalAssets': int(assets), 'totalLiabilities': int(liabilities),
            'stockholdersEquity': int(equity), 'cash': int(cash), 'totalDebt': int(debt),
        })
        assets *= random.uniform(0.9, 1.0)
    return sheets


@api_bp.route('/sentiment')
def sentiment():
    ticker = request.args.get('ticker', 'OGDC').upper()
    data = _fetch_stock_data(ticker, '3mo', '1d')
    if data is None or len(data) < 20:
        return jsonify({'error': 'Insufficient data'}), 400

    close = data['Close']
    rsi = _calc_rsi(close, 14)
    macd_l, sig, _ = _calc_macd(close)
    sma20 = _calc_sma(close, 20)
    sma50 = _calc_sma(close, 50)

    score = 50
    latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    if latest_rsi < 30:
        score += 15
    elif latest_rsi > 70:
        score -= 15
    elif latest_rsi < 50:
        score += 5
    else:
        score -= 5

    if not pd.isna(macd_l.iloc[-1]) and not pd.isna(sig.iloc[-1]):
        if macd_l.iloc[-1] > sig.iloc[-1]:
            score += 10
        else:
            score -= 10

    if not pd.isna(sma20.iloc[-1]) and not pd.isna(sma50.iloc[-1]):
        if close.iloc[-1] > sma20.iloc[-1]:
            score += 10
        else:
            score -= 10
        if sma20.iloc[-1] > sma50.iloc[-1]:
            score += 5
        else:
            score -= 5

    score = max(0, min(100, score))
    if score >= 65:
        signal = 'Buy'
    elif score <= 35:
        signal = 'Sell'
    else:
        signal = 'Hold'

    return jsonify({
        'ticker': ticker,
        'score': score,
        'signal': signal,
        'rsi': round(float(latest_rsi), 2),
        'macdSignal': 'Bullish' if not pd.isna(macd_l.iloc[-1]) and macd_l.iloc[-1] > sig.iloc[-1] else 'Bearish',
        'trendSignal': 'Above SMA' if close.iloc[-1] > sma20.iloc[-1] else 'Below SMA',
    })


# Financial Models
@api_bp.route('/capm', methods=['POST'])
def capm():
    params = request.get_json() or {}
    ticker = params.get('ticker', 'OGDC').upper()
    rf = float(params.get('risk_free_rate', 4.5)) / 100

    info = _get_stock_info(ticker)
    beta = info.get('beta', 1.0) or 1.0
    market_return = 0.10
    expected = rf + beta * (market_return - rf)

    return jsonify({
        'ticker': ticker,
        'beta': round(beta, 3),
        'riskFreeRate': round(rf * 100, 2),
        'marketReturn': round(market_return * 100, 2),
        'expectedReturn': round(expected * 100, 2),
        'riskPremium': round((market_return - rf) * 100, 2),
    })


@api_bp.route('/monte-carlo', methods=['POST'])
def monte_carlo():
    params = request.get_json() or {}
    ticker = params.get('ticker', 'OGDC').upper()
    simulations = min(int(params.get('simulations', 1000)), 5000)
    days = min(int(params.get('days', 252)), 504)

    data = _fetch_stock_data(ticker, '1y', '1d')
    if data is None or len(data) < 20:
        return jsonify({'error': 'Insufficient data'}), 400

    returns = data['Close'].pct_change().dropna()
    mu = float(returns.mean())
    sigma = float(returns.std())
    last_price = float(data['Close'].iloc[-1])

    paths = []
    for _ in range(min(simulations, 50)):
        daily_returns = np.random.normal(mu, sigma, days)
        price_path = last_price * np.cumprod(1 + daily_returns)
        paths.append([round(float(p), 2) for p in price_path])

    all_final = []
    for _ in range(simulations):
        daily_returns = np.random.normal(mu, sigma, days)
        final = last_price * np.prod(1 + daily_returns)
        all_final.append(float(final))

    return jsonify({
        'ticker': ticker,
        'currentPrice': round(last_price, 2),
        'paths': paths,
        'days': days,
        'simulations': simulations,
        'meanPrice': round(float(np.mean(all_final)), 2),
        'medianPrice': round(float(np.median(all_final)), 2),
        'percentile5': round(float(np.percentile(all_final, 5)), 2),
        'percentile95': round(float(np.percentile(all_final, 95)), 2),
        'stdDev': round(float(np.std(all_final)), 2),
    })


@api_bp.route('/black-scholes', methods=['POST'])
def black_scholes():
    params = request.get_json() or {}
    ticker = params.get('ticker', 'OGDC').upper()
    strike = float(params.get('strike_price', 150))
    expiry_days = int(params.get('expiry_days', 30))
    rf = float(params.get('risk_free_rate', 4.5)) / 100
    vol_input = params.get('volatility')

    info = _get_stock_info(ticker)
    S = info.get('currentPrice', 150)

    if vol_input:
        sigma = float(vol_input) / 100
    else:
        data = _fetch_stock_data(ticker, '1y', '1d')
        returns = data['Close'].pct_change().dropna()
        sigma = float(returns.std() * np.sqrt(252))

    T = expiry_days / 365.0
    if T <= 0 or sigma <= 0:
        return jsonify({'error': 'Invalid parameters'}), 400

    d1 = (np.log(S / strike) + (rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * norm.cdf(d1) - strike * np.exp(-rf * T) * norm.cdf(d2)
    put = strike * np.exp(-rf * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return jsonify({
        'ticker': ticker,
        'spotPrice': round(S, 2),
        'strikePrice': round(strike, 2),
        'expiryDays': expiry_days,
        'riskFreeRate': round(rf * 100, 2),
        'volatility': round(sigma * 100, 2),
        'callPrice': round(float(call), 4),
        'putPrice': round(float(put), 4),
        'd1': round(float(d1), 4),
        'd2': round(float(d2), 4),
        'callDelta': round(float(norm.cdf(d1)), 4),
        'putDelta': round(float(norm.cdf(d1) - 1), 4),
        'gamma': round(float(norm.pdf(d1) / (S * sigma * np.sqrt(T))), 6),
        'theta': round(float(-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))), 4),
        'vega': round(float(S * norm.pdf(d1) * np.sqrt(T)), 4),
    })


@api_bp.route('/markets')
def markets_list():
    return jsonify({name: {'name': m['name'], 'stockCount': len(m['stocks'])} for name, m in MARKETS.items()})


@api_bp.route('/market-stocks/<market>')
def market_stocks(market):
    m = MARKETS.get(market.upper())
    if not m:
        return jsonify({'error': 'Market not found'}), 404
    return jsonify([{'symbol': s, 'name': n} for s, n in m['stocks'].items()])


@api_bp.route('/sectors')
def sectors():
    sector_counts = {}
    for ticker, sector in SECTOR_MAP.items():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    return jsonify(sector_counts)


@api_bp.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
def watchlist():
    from flask_login import current_user
    from models import db
    from models.user import Watchlist

    if not current_user.is_authenticated:
        return jsonify({'error': 'Login required'}), 401

    if request.method == 'GET':
        items = Watchlist.query.filter_by(user_id=current_user.id).all()
        result = []
        for w in items:
            info = _get_stock_info(w.ticker)
            price = info.get('currentPrice', 0)
            prev = info.get('previousClose', price)
            change = price - prev
            change_pct = (change / prev * 100) if prev else 0
            result.append({
                'symbol': w.ticker,
                'name': _all_tickers().get(w.ticker, w.ticker),
                'price': round(price, 2),
                'change': round(change, 2),
                'changePercent': round(change_pct, 2),
                'notes': w.notes,
                'addedAt': w.added_at.isoformat() if w.added_at else None,
            })
        return jsonify(result)

    if request.method == 'POST':
        data = request.get_json() or {}
        ticker = data.get('ticker', '').upper()
        notes = data.get('notes', '')
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
        existing = Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first()
        if existing:
            existing.notes = notes
        else:
            db.session.add(Watchlist(user_id=current_user.id, ticker=ticker, notes=notes))
        db.session.commit()
        return jsonify({'status': 'ok'})

    if request.method == 'DELETE':
        data = request.get_json() or {}
        ticker = data.get('ticker', '').upper()
        item = Watchlist.query.filter_by(user_id=current_user.id, ticker=ticker).first()
        if item:
            db.session.delete(item)
            db.session.commit()
        return jsonify({'status': 'ok'})
