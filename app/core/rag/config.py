# core/rag/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Datos de conexión a la Base de Datos
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "your_password"

    # Parámetros para el chunking del documento
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Configuración del modelo de embeddings
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    VECTOR_SIZE: int = 384  # Ajusta según el modelo

    # Configuración del índice FAISS
    FAISS_INDEX_PATH: str = "data/faiss_index.bin"
    N_LISTS: int = 100  # Número de listas para el índice IVF

    # Configuración del pool de conexiones
    MAX_CONNECTIONS: int = 50

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Ejemplo de configuración escalable:
class ScalableConfig(Config):
    SHARD_SIZE: int = 1_000_000   # Cantidad de vectores por shard en FAISS
    CACHE_SIZE: int = 10_000      # Tamaño del caché (ej. con Redis)
