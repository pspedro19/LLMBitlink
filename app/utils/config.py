# app/utils/config.py

import os
# app/utils/config.py should have paths like this:
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  # gets the app directory
DATABASE_DIR = BASE_DIR / 'data' / 'database'

DATABASE_PATHS = {
    'activities': DATABASE_DIR / 'activities.xlsx',
    'tourist_spots': DATABASE_DIR / 'tourist_spots.xlsx',
    'restaurants': DATABASE_DIR / 'restaurants.xlsx',
    'nightclubs': DATABASE_DIR / 'nightclubs.xlsx',
    'tourism_packages': DATABASE_DIR / 'tourism_packages.xlsx'
}

# Model paths
MODEL_PATHS = {
    "spacy_model": "es_core_news_sm"
}

# Cache settings
CACHE_CONFIG = {
    "max_size": 1000,
    "ttl": 3600  # 1 hour
}

# API settings
API_CONFIG = {
    "max_recommendations": 10,
    "default_page_size": 20
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}