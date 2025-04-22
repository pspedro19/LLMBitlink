# app/utils/config.py
from typing import Dict, Any
import os
from pathlib import Path

class Config:
    """Configuración para el sistema de recomendaciones."""
    def __init__(self):
        # Rutas base - CORREGIDAS PARA APUNTAR AL DIRECTORIO APP3
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.DATABASE_DIR = self.DATA_DIR / "database"
        
        # Configuración de base de datos
        self.DATABASE_PATHS = {
            "activities": self.DATABASE_DIR / "activities.xlsx",
            "tourist_spots": self.DATABASE_DIR / "tourist_spots.xlsx",
            "restaurants": self.DATABASE_DIR / "restaurants.xlsx",
            "nightclubs": self.DATABASE_DIR / "nightclubs.xlsx",
            "tourism_packages": self.DATABASE_DIR / "tourism_packages.xlsx"
        }
        
        # Configuración de logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Configuración de API
        self.APP_NAME = "Curacao Tourism Recommender"
        self.APP_VERSION = "1.0.0"
        
        # Configuración de procesamiento NLU
        self.DEFAULT_LANGUAGE = "es"
        self.AVAILABLE_LANGUAGES = ["es", "en"]
        
    def get_db_path(self, name: str) -> Path:
        """Obtiene la ruta a un archivo de base de datos específico"""
        if name in self.DATABASE_PATHS:
            return self.DATABASE_PATHS[name]
        raise ValueError(f"Base de datos '{name}' no definida en la configuración")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a un diccionario
        
        Returns:
            Dict[str, Any]: Configuración como diccionario
        """
        # Excluir métodos, solo incluir atributos
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }
        
    def load_from_file(self, file_path: str) -> None:
        """
        Carga configuración desde un archivo
        
        Args:
            file_path (str): Ruta al archivo de configuración
        """
        import json
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
                
            # Actualizar configuración
            for key, value in config_data.items():
                if hasattr(self, key):
                    # Convertir a Path si corresponde
                    if key.endswith('_DIR') or key.endswith('_PATH'):
                        setattr(self, key, Path(value))
                    else:
                        setattr(self, key, value)
                        
        except Exception as e:
            print(f"Error cargando configuración desde {file_path}: {e}")