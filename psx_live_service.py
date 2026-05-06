"""
PSX Live Price Service
======================
Fetches real-time (15-min delayed) prices from Yahoo Finance for all PSX stocks.
Runs a background thread that refreshes every 60 s during market hours,
every 5 min outside market hours.

Usage:
    from psx_live_service import psx_live
    psx_live.start(initial_prices)   # call once on app startup
    price = psx_live.get_price('UBL')
    all_prices = psx_live.get_all()
"""

import threading
import time
import logging
from datetime import datetime

import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker mapping: internal app ticker → Yahoo Finance symbol
# Yahoo Finance uses PSX tickers with the .KA suffix (Karachi Stock Exchange).
# Most tickers are 1:1, but a few differ from PSX official names.
# ---------------------------------------------------------------------------
_INTERNAL_TO_YF = {
    # Our name        Yahoo Finance symbol
    'COLGATE':       'COLG.KA',    # Colgate-Palmolive Pakistan (PSX: COLG)
    'LOTTE':         'LOTCHEM.KA', # Lotte Chemical Pakistan   (PSX: LOTCHEM)
    'ISML':          'ISIL.KA',    # Ismail Industries Ltd.    (PSX: ISIL)
    'SAPM':          'SPWL.KA',    # Saif Power Ltd.           (PSX: SPWL)
    'UNILEVER':      'UPFL.KA',    # Unilever Pakistan Foods   (PSX: UPFL)
}

def internal_to_yf(ticker: str) -> str:
    """Convert an internal app ticker to its Yahoo Finance symbol."""
    return _INTERNAL_TO_YF.get(ticker, f"{ticker}.KA")

def yf_to_internal(yf_sym: str) -> str:
    """Convert a Yahoo Finance symbol back to the internal app ticker."""
    # Reverse lookup in the custom map first
    for internal, yf in _INTERNAL_TO_YF.items():
        if yf == yf_sym:
            return internal
    # Default: strip the .KA suffix
    return yf_sym.replace('.KA', '')


# ---------------------------------------------------------------------------
# Market hours (PSX: Mon–Fri, 09:15 – 15:30 PKT)
# ---------------------------------------------------------------------------
_PKT = pytz.timezone('Asia/Karachi')
_MARKET_OPEN  = (9,  15)   # (hour, minute)
_MARKET_CLOSE = (15, 30)

def _is_market_open() -> bool:
    now = datetime.now(_PKT)
    if now.weekday() >= 5:          # Saturday or Sunday
        return False
    mins_now   = now.hour * 60 + now.minute
    mins_open  = _MARKET_OPEN[0]  * 60 + _MARKET_OPEN[1]
    mins_close = _MARKET_CLOSE[0] * 60 + _MARKET_CLOSE[1]
    return mins_open <= mins_now <= mins_close


# ---------------------------------------------------------------------------
# Live service
# ---------------------------------------------------------------------------
class PSXLiveService:
    """
    Thread-safe container for live PSX stock prices.

    _store = {
        'TICKER': {
            'price':      float,   # current price (PKR)
            'prev_close': float,   # previous close
            'change':     float,   # absolute change
            'change_pct': float,   # percentage change
            'volume':     int,
        },
        ...
    }
    """

    def __init__(self):
        self._store: dict = {}
        self._last_update: float | None = None
        self._lock = threading.Lock()
        self._started = False

    # ------------------------------------------------------------------
    def start(self, tickers: list[str], base_prices: dict[str, float]):
        """
        Start the background refresh thread.

        Args:
            tickers:     list of internal ticker strings (e.g. ['UBL', 'MCB', …])
            base_prices: fallback hardcoded prices {ticker: price} used until
                         the first successful live fetch
        """
        if self._started:
            return
        self._started = True

        # Pre-populate store with hardcoded prices so UI has data immediately
        with self._lock:
            for t in tickers:
                bp = base_prices.get(t, 100.0)
                self._store[t] = {
                    'price':      bp,
                    'prev_close': bp,
                    'change':     0.0,
                    'change_pct': 0.0,
                    'volume':     0,
                }

        self._tickers = tickers

        thread = threading.Thread(target=self._run, daemon=True, name='psx-live')
        thread.start()
        logger.info('PSX live price service started (%d tickers)', len(tickers))

    # ------------------------------------------------------------------
    def _run(self):
        """Background loop: fetch → sleep → fetch → …"""
        # Initial fetch with a short delay so Flask has time to start
        time.sleep(3)
        self._fetch()

        while True:
            interval = 60 if _is_market_open() else 300   # 1 min / 5 min
            time.sleep(interval)
            self._fetch()

    # ------------------------------------------------------------------
    def _fetch(self):
        """Download latest prices for all tickers from Yahoo Finance."""
        try:
            # Build Yahoo Finance symbol list
            yf_syms = [internal_to_yf(t) for t in self._tickers]

            logger.debug('Fetching %d PSX tickers from Yahoo Finance…', len(yf_syms))

            # yf.download with period='1d' and interval='1m' gives the most
            # recent intraday prices.  Fall back to period='2d' daily bars when
            # outside market hours so we get the last closing price.
            kwargs = dict(
                tickers=yf_syms,
                period='1d',
                interval='1m',
                progress=False,
                threads=True,
                auto_adjust=True,
            )
            raw = yf.download(**kwargs)

            updated = 0
            if raw is not None and not raw.empty:
                updated = self._process(raw, yf_syms)

            # If very few tickers updated (yfinance returned sparse data),
            # also try a daily bar download as a supplement
            if updated < len(self._tickers) // 2:
                raw2 = yf.download(
                    tickers=yf_syms,
                    period='5d',
                    interval='1d',
                    progress=False,
                    threads=True,
                    auto_adjust=True,
                )
                if raw2 is not None and not raw2.empty:
                    self._process(raw2, yf_syms, overwrite=False)

            self._last_update = time.time()
            logger.info('PSX live prices updated (%d/%d tickers)',
                        updated, len(self._tickers))

        except Exception as exc:
            logger.warning('PSX live fetch failed: %s', exc)

    # ------------------------------------------------------------------
    def _process(self, raw, yf_syms: list[str], overwrite: bool = True) -> int:
        """Parse a yfinance DataFrame and update the store. Returns count updated."""
        updated = 0
        with self._lock:
            for yf_sym in yf_syms:
                internal = yf_to_internal(yf_sym)
                try:
                    # yf.download with multiple tickers uses MultiIndex columns
                    if ('Close', yf_sym) in raw.columns:
                        series = raw['Close'][yf_sym].dropna()
                    elif 'Close' in raw.columns:
                        # Single ticker fallback (shouldn't normally happen here)
                        series = raw['Close'].dropna()
                    else:
                        continue

                    if series.empty:
                        continue

                    price = float(series.iloc[-1])
                    if price <= 0:
                        continue

                    # Previous close: last-but-one bar or open of today
                    if ('Open', yf_sym) in raw.columns:
                        open_series = raw['Open'][yf_sym].dropna()
                        prev = float(open_series.iloc[0]) if not open_series.empty else price
                    else:
                        prev = price

                    change     = round(price - prev, 2)
                    change_pct = round((change / prev * 100) if prev else 0, 2)

                    vol = 0
                    if ('Volume', yf_sym) in raw.columns:
                        vol_series = raw['Volume'][yf_sym].dropna()
                        if not vol_series.empty:
                            vol = int(vol_series.iloc[-1])

                    if overwrite or price != self._store.get(internal, {}).get('price'):
                        self._store[internal] = {
                            'price':      round(price, 2),
                            'prev_close': round(prev,  2),
                            'change':     change,
                            'change_pct': change_pct,
                            'volume':     vol,
                        }
                        updated += 1

                except Exception:
                    continue

        return updated

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_price(self, ticker: str) -> float | None:
        with self._lock:
            entry = self._store.get(ticker)
            return entry['price'] if entry else None

    def get_all(self) -> dict:
        with self._lock:
            return {t: dict(v) for t, v in self._store.items()}

    def get_last_update(self) -> float | None:
        return self._last_update

    @property
    def market_open(self) -> bool:
        return _is_market_open()

    @property
    def status(self) -> str:
        return 'open' if _is_market_open() else 'closed'


# Singleton
psx_live = PSXLiveService()
