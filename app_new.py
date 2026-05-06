import os
import json
import numpy as np
import pandas as pd
from flask import Flask
from config import Config
from models import db, login_manager

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.json_encoder = NumpyEncoder

    db.init_app(app)
    login_manager.init_app(app)

    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(api_bp)

    with app.app_context():
        from models.user import User, Watchlist
        db.create_all()

    # Start PSX live price service (yfinance-backed, auto-refreshes every 60 s)
    from psx_live_service import psx_live
    from fast_data_service import fast_data_service as _fds
    from blueprints.api import MARKETS

    _all_tickers = list(MARKETS['PSX']['stocks'].keys())
    _base_prices = {
        t: _fds.market_data['PSX'].get(t, {}).get('base_price', 100.0)
        for t in _all_tickers
    }
    psx_live.start(_all_tickers, _base_prices)

    # Pre-warm screener cache in background so first visitor doesn't wait 60s
    import threading
    def _prewarm():
        import time
        time.sleep(5)  # let Flask + live service start first
        try:
            with app.test_client() as client:
                client.post(
                    '/api/screener',
                    json={'market': 'PSX', 'sort_by': 'marketCap', 'sort_dir': 'desc',
                          'page': 1, 'per_page': 9999, 'include_indicators': True},
                    content_type='application/json'
                )
        except Exception:
            pass  # silently ignore if pre-warm fails

    threading.Thread(target=_prewarm, daemon=True).start()

    return app

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
