# core/rag/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from core.rag.config import Config
from core.rag.db_manager import DatabaseManager

class EmbeddingsManager:
    def __init__(self, config: Config):
        self.config = config
        # Soporte para múltiples modelos
        self.models = {
            'miniLM': SentenceTransformer('all-MiniLM-L6-v2'),
            'mpnet': SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        }

    def get_model(self, model_name: str):
        return self.models.get(model_name, self.models['miniLM'])

    def generate_embeddings(self, texts: List[str], model_name: str = 'miniLM') -> np.ndarray:
        model = self.get_model(model_name)
        # Se generan embeddings normalizados para similitud coseno
        return model.encode(texts, normalize_embeddings=True)

    def store_embeddings(
        self,
        db: DatabaseManager,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        faiss_ids: List[int],
        model_name: str = 'miniLM'
    ):
        for chunk_id, emb, faiss_id in zip(chunk_ids, embeddings, faiss_ids):
            db.insert_embedding(chunk_id=chunk_id, model_name=model_name, vector=emb, faiss_index_id=faiss_id)
