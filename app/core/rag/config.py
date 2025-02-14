# core/rag/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Configuración simplificada para RAG"""
    
    def __init__(self):
        # Directorio base de la aplicación
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent
        # Directorio padre donde está el .env
        self.PROJECT_ROOT = self.BASE_DIR.parent
        
        # Cargar variables de entorno desde el directorio padre
        self._load_env()
        
        self.INDICES_DIR = self.BASE_DIR / "data/indices"
        
        # FAISS
        self.VECTOR_SIZE = 384  # Dimensión de vectores para miniLM
        self.N_LISTS = 100      # Número de listas para IVF
        
        # Database        
        self.DB_HOST = os.getenv("DB_HOST", "localhost")
        self.DB_PORT = int(os.getenv("DB_PORT", "5432"))
        self.DB_NAME = os.getenv("DB_NAME", "mydatabase")
        self.DB_USER = os.getenv("DB_USER", "myuser")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD", "mypassword")
        
        # Crear directorio de índices
        self.INDICES_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_env(self):
        """Carga las variables de entorno desde el directorio padre"""
        env_path = self.PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            raise FileNotFoundError(f"Archivo .env no encontrado en {env_path}")
    
    @classmethod
    def get_instance(cls):
        return cls()