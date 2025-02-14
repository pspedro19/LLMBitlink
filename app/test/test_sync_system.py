# test/test_sync_system.py
import asyncio
import logging
import pytest
from pathlib import Path
import numpy as np
import sys

# En Windows, usa WindowsSelectorEventLoopPolicy para evitar problemas con librerías nativas.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.rag.config import Config
from core.rag.faiss_manager import PersistentFAISSManager  # Usamos la versión persistente (que tiene reset_index)
from core.rag.sync_manager import SyncManager

@pytest.mark.asyncio
async def test_postgres_faiss_sync():
    """Prueba la sincronización entre PostgreSQL y FAISS (con reinicialización completa)."""
    config = Config()
    test_indices_dir = Path(__file__).parent / "test_indices"
    test_indices_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        config.INDICES_DIR = test_indices_dir
        faiss_manager = PersistentFAISSManager(config)
        sync_manager = SyncManager(config, faiss_manager)
        
        initial_info = faiss_manager.get_index_info()
        logger.info(f"Estado inicial del índice: {initial_info}")
        
        # Reinicia el índice y sincroniza todos los embeddings.
        processed, failed = await sync_manager.synchronize()
        logger.info(f"Sincronización completada - Procesados: {processed}, Fallidos: {failed}")
        assert failed == 0, f"No deberían haber fallado embeddings: {failed}"
        
        status = await sync_manager.verify_sync_status()
        logger.info(f"Estado post-sincronización: {status}")
        assert status['postgres_embeddings'] == status['faiss_vectors'], "Discrepancia en números"
        
        # Prueba con un vector de ceros (control)
        zero_vector = np.zeros((1, faiss_manager.dimension), dtype=np.float32)
        logger.info(f"Buscando con vector de ceros: shape {zero_vector.shape}, dtype {zero_vector.dtype}")
        distances_zero, ids_zero = faiss_manager.search(zero_vector, k=1)
        logger.info(f"Resultado búsqueda vector de ceros: Distancias: {distances_zero}, IDs: {ids_zero}")
        
        if processed > 0:
            # Obtener un embedding real de la base de datos y verificarlo
            pool = await sync_manager._get_pool()
            async with pool.acquire() as conn:
                sample = await conn.fetchrow("""
                    SELECT embedding_id, embedding
                    FROM embeddings
                    LIMIT 1
                """)
                if sample:
                    vector = sync_manager._parse_embedding(sample['embedding'])
                    logger.info(f"Query vector shape: {vector.shape}, dtype: {vector.dtype}")
                    logger.info(f"Primeros 5 valores: {vector[0][:5]}")
                    logger.info(f"Is C-contiguous: {vector.flags['C_CONTIGUOUS']}")
                    
                    distances, ids = faiss_manager.search(vector, k=1)
                    logger.info(f"Resultado búsqueda vector de BD: Distancias: {distances}, IDs: {ids}")
                    assert len(ids) > 0, "La búsqueda debería retornar resultados"
                    assert all(d >= 0 for d in distances[0]), "Las distancias deberían ser no negativas"
    
    except Exception as e:
        logger.error(f"Error en prueba: {e}", exc_info=True)
        raise
    
    finally:
        import shutil
        if test_indices_dir.exists():
            shutil.rmtree(test_indices_dir)
            logger.info("Limpieza completada")
