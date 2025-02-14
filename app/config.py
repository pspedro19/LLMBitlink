import os
from pathlib import Path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/database"

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "your_db_name")
    DB_USER = os.getenv("DB_USER", "your_db_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "your_db_password")
    
    # FAISS
    VECTOR_SIZE = 384  # Dimensión de los vectores (ajustar según el modelo)
    N_LISTS = 100      # Número de listas para IVF
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Paths
    KNOWLEDGE_BASE_PATH = BASE_DIR / "data/knowledge_base"
    CURACAO_INFO_PATHS = [
        KNOWLEDGE_BASE_PATH / "curaçao_information/0_Introduccion",
        KNOWLEDGE_BASE_PATH / "curaçao_information/1_Atracciones",
        KNOWLEDGE_BASE_PATH / "curaçao_information/2_Cultura_e_Historia",
        KNOWLEDGE_BASE_PATH / "curaçao_information/3_Consejos_Practicos"
    ]