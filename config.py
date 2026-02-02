import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///stockscreener.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CACHE_DURATION = 300  # 5 minutes
    DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    STOCKS_PER_PAGE = 25

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Must be set in production
