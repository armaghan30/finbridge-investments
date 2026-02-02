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

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
