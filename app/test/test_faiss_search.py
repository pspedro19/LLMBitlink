# test/test_faiss_search.py
import asyncio
import logging
import os
from pathlib import Path
import numpy as np
import pytest

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importa la configuración y los gestores
from core.rag.config import Config
from core.rag.faiss_manager import PersistentFAISSManager
from core.rag.sync_manager import SyncManager

@pytest.mark.asyncio
async def test_faiss_search_curazao_data():
    """
    Prueba FAISS usando los datos sincronizados de Curazao (turística, histórica y cultural).
    Se espera que, luego de la sincronización, al realizar una búsqueda con un vector de consulta
    tomado de un registro de la base de datos, se retornen resultados válidos.
    """
    # Establece la variable de entorno para omitir la carga de archivos Excel durante la prueba
    os.environ["SKIP_CSV_LOAD"] = "true"

    # Configuración inicial del entorno de prueba
    config = Config()
    test_indices_dir = Path(__file__).parent / "test_indices"
    test_indices_dir.mkdir(parents=True, exist_ok=True)
    config.INDICES_DIR = test_indices_dir

    try:
        # Usa PersistentFAISSManager para tener acceso al método reset_index
        faiss_manager = PersistentFAISSManager(config)
        sync_manager = SyncManager(config, faiss_manager)
        
        # Sincroniza los embeddings desde la base de datos en FAISS
        processed, failed = await sync_manager.synchronize()
        logger.info(f"Sincronización completada - Procesados: {processed}, Fallidos: {failed}")
        assert failed == 0, "La sincronización presentó errores."
        assert processed > 0, "No se procesaron embeddings para sincronización."
        
        # Realiza una búsqueda de control usando un vector de consulta obtenido de la BD
        pool = await sync_manager._get_pool()
        async with pool.acquire() as conn:
            sample = await conn.fetchrow("""
                SELECT embedding_id, embedding
                FROM embeddings
                LIMIT 1
            """)
            assert sample is not None, "No se encontró ningún embedding en la base de datos."
            
            # Procesa el embedding para obtener el vector de consulta
            query_vector = sync_manager._parse_embedding(sample['embedding'])
            logger.info(f"Query vector shape: {query_vector.shape}, dtype: {query_vector.dtype}")
            logger.info(f"Primeros 5 valores del query vector: {query_vector[0][:5]}")
            
            # Realiza la búsqueda en FAISS
            distances, ids = faiss_manager.search(query_vector, k=5)
            logger.info(f"Resultado búsqueda: IDs={ids}, Distancias={distances}")
            
            # Verifica que se hayan obtenido resultados válidos
            assert ids.size > 0, "La búsqueda no retornó resultados."
            assert np.all(np.isfinite(distances)), "Las distancias contienen valores no finitos."
    
    except Exception as e:
        logger.error(f"Error en la prueba de búsqueda FAISS: {e}", exc_info=True)
        raise
    
    finally:
        # Limpieza: elimina el directorio de índices de prueba
        import shutil
        if test_indices_dir.exists():
            shutil.rmtree(test_indices_dir)
            logger.info("Limpieza completada")
