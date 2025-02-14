# app/core/rag/faiss_config.py

import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    name: str
    vector_size: int
    index_path: str
    n_lists: int = 100

class FAISSConfig:
    # Configuración de modelos de embeddings
    MODELS: Dict[str, ModelConfig] = {
        'miniLM': ModelConfig(
            name="all-MiniLM-L6-v2",
            vector_size=384,
            index_path="data/indices/minilm_index.faiss",
        ),
        'mpnet': ModelConfig(
            name="multi-qa-mpnet-base-dot-v1",
            vector_size=768,
            index_path="data/indices/mpnet_index.faiss",
        )
    }

    # Configuración de sincronización
    SYNC_BATCH_SIZE: int = 1000
    AUTO_SYNC: bool = True
    
    # Configuración de búsqueda
    DEFAULT_TOP_K: int = 5
    NPROBE: int = 10

    # Configuración de entrenamiento
    TRAINING_SAMPLE_SIZE: int = 10000
    
    # Configuración de almacenamiento
    INDEX_ROOT_DIR: str = "data/indices"
    BACKUP_DIR: str = "data/indices/backups"
    
    @classmethod
    def get_model_config(cls, model_key: str = 'miniLM') -> ModelConfig:
        """Obtiene la configuración para un modelo específico."""
        return cls.MODELS.get(model_key, cls.MODELS['miniLM'])
    
    @classmethod
    def default_model_config(cls) -> ModelConfig:
        """Retorna la configuración del modelo por defecto (miniLM)."""
        return cls.MODELS['miniLM']