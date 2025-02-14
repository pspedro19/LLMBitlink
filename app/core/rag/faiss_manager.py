# core/rag/faiss_manager.py
import faiss
import numpy as np
import os
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class FAISSManager:
    """Gestor simple para índices FAISS."""
    
    def __init__(self, config):
        self.config = config
        self.dimension = 384  # Debe coincidir con la dimensión de los embeddings
        logger.info(f"Inicializando FAISSManager con dimensión {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        if not self.index:
            raise RuntimeError("No se pudo crear el índice FAISS")
        logger.info("Índice FAISS creado exitosamente")

    def verify_vector(self, vector: np.ndarray) -> np.ndarray:
        """Verifica y formatea un vector para FAISS."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        if vector.shape[1] != self.dimension:
            raise ValueError(f"Vector dimensión incorrecta: {vector.shape[1]} != {self.dimension}")
        
        if not np.all(np.isfinite(vector)):
            raise ValueError("El vector contiene valores no finitos (NaN o inf)")
        
        vector = np.ascontiguousarray(vector)
        return vector

    def add_vectors(self, vectors: np.ndarray) -> list:
        """Añade vectores al índice."""
        try:
            vectors = self.verify_vector(vectors)
            logger.info(f"Añadiendo {len(vectors)} vectores")
            
            start_id = self.index.ntotal
            self.index.add(vectors)
            end_id = self.index.ntotal
            
            logger.info(f"Vectores añadidos. IDs: {start_id} a {end_id-1}")
            return list(range(start_id, end_id))
        except Exception as e:
            logger.error(f"Error al añadir vectores: {e}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Busca los k vectores más cercanos."""
        try:
            query_vector = self.verify_vector(query_vector)
            # Verificar que el vector de consulta tenga la forma esperada
            if query_vector.shape != (1, self.dimension):
                raise ValueError(f"query_vector shape es {query_vector.shape}, se esperaba (1, {self.dimension})")
            logger.info(f"Buscando {k} vecinos con vector de consulta de shape {query_vector.shape}")
            
            if self.index.ntotal == 0:
                logger.warning("El índice está vacío")
                return np.array([]), np.array([])
            
            distances, ids = self.index.search(query_vector, min(k, self.index.ntotal))
            logger.info(f"Búsqueda completada: {len(ids[0])} resultados")
            return distances, ids
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            raise

    def get_index_info(self) -> dict:
        """Retorna información sobre el índice."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexFlatL2',
            'status': 'ready'
        }

class PersistentFAISSManager(FAISSManager):
    """FAISSManager con persistencia."""
    
    def __init__(self, config):
        super().__init__(config)
        self.index_path = os.path.join(config.INDICES_DIR, 'faiss_index.bin')
        self._load_existing_index()

    def _load_existing_index(self):
        """Carga un índice existente si está disponible."""
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Cargando índice desde {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Índice cargado: {self.index.ntotal} vectores")
            except Exception as e:
                logger.error(f"Error al cargar índice existente: {e}")
                # Se conserva el índice creado en el constructor base

    def save_index(self):
        """Guarda el índice en disco."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Índice guardado en {self.index_path}")
        except Exception as e:
            logger.error(f"Error al guardar índice: {e}")
            raise

    def add_vectors(self, vectors: np.ndarray) -> list:
        """Añade vectores y guarda el índice."""
        ids = super().add_vectors(vectors)
        self.save_index()
        return ids

    def reset_index(self):
        """Reinicializa el índice FAISS y elimina el índice persistente si existe."""
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                logger.info(f"Índice persistente eliminado: {self.index_path}")
            except Exception as e:
                logger.error(f"Error eliminando el índice persistente: {e}")
                raise
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info("Índice FAISS reinicializado para sincronización completa.")
