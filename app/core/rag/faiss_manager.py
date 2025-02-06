# core/rag/faiss_manager.py
import faiss
import numpy as np
from core.rag.config import Config

class FAISSManager:
    def __init__(self, config: Config):
        self.config = config
        self.dimension = config.VECTOR_SIZE
        self.index = self._initialize_index()

    def _initialize_index(self) -> faiss.Index:
        # Se utiliza un índice IVF Flat con métrica L2 (ajustable a cosine si se desea)
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFFlat(quantizer, self.dimension, self.config.N_LISTS)
        if not index.is_trained:
            # Entrenar el índice con datos de entrenamiento sintéticos o reales
            training_data = np.random.random((10000, self.dimension)).astype('float32')
            index.train(training_data)
        return index

    def add_vectors(self, vectors: np.ndarray) -> list:
        start_id = self.index.ntotal
        self.index.add(vectors.astype('float32'))
        return list(range(start_id, start_id + len(vectors)))

    def search(self, query_vector: np.ndarray, k: int = 5):
        distances, ids = self.index.search(query_vector.reshape(1, -1).astype('float32'), k)
        return distances, ids
