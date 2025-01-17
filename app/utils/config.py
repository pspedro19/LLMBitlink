# app/utils/config.py

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

# Database paths
DATABASE_PATHS = {
    "activities": DATA_DIR / "database" / "activities.xlsx",
    "tourist_spots": DATA_DIR / "database" / "tourist_spots.xlsx",
    "restaurants": DATA_DIR / "database" / "restaurants.xlsx",
    "nightclubs": DATA_DIR / "database" / "nightclubs.xlsx",
    "tourism_packages": DATA_DIR / "database" / "tourism_packages.xlsx"
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