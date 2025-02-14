# core/rag/deletion_handler.py

import logging
from typing import List, Set
from uuid import UUID
import numpy as np
from core.rag.sync_manager import SyncManager

logger = logging.getLogger(__name__)

class DeletionHandler:
    def __init__(self, sync_manager: SyncManager):
        self.sync_manager = sync_manager
        self.placeholder_vector = np.zeros(sync_manager.faiss_manager.dimension)
        self.deleted_uuids: Set[str] = set()

    async def mark_deleted(self, embedding_ids: List[UUID]) -> bool:
        """
        Marca embeddings como eliminados y actualiza el índice FAISS.
        
        En lugar de eliminar físicamente los vectores (lo cual alteraría los índices),
        los reemplazamos con vectores de ceros y mantenemos un registro de los eliminados.
        """
        try:
            for emb_id in embedding_ids:
                str_id = str(emb_id)
                if str_id in self.sync_manager.uuid_to_faiss:
                    faiss_id = self.sync_manager.uuid_to_faiss[str_id]
                    
                    # Reemplazar el vector con ceros en el índice FAISS
                    self.sync_manager.faiss_manager.index.remove_ids(np.array([faiss_id]))
                    
                    # Marcar como eliminado
                    self.deleted_uuids.add(str_id)
                    
                    # Actualizar el mapeo
                    del self.sync_manager.uuid_to_faiss[str_id]
            
            # Guardar los cambios
            self.sync_manager.faiss_manager.save_index()
            self.sync_manager._save_mapping()
            
            return True
            
        except Exception as e:
            logger.error(f"Error marking embeddings as deleted: {e}")
            return False

    async def cleanup_deleted(self) -> int:
        """
        Limpia los registros de elementos eliminados y reorganiza el índice si es necesario.
        Retorna el número de registros limpiados.
        """
        try:
            cleaned = len(self.deleted_uuids)
            self.deleted_uuids.clear()
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up deleted embeddings: {e}")
            return 0

    def get_deletion_status(self) -> dict:
        """Retorna el estado actual de las eliminaciones."""
        return {
            'total_deleted': len(self.deleted_uuids),
            'deleted_uuids': list(self.deleted_uuids)
        }