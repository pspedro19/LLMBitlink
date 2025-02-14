# test/debug_embedding.py
import sys
import os
from pathlib import Path
import asyncio
import logging

# Añadir el directorio raíz al path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from core.rag.config import Config
from core.rag.sync_manager import SyncManager
from core.rag.faiss_manager import PersistentFAISSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_embedding():
    try:
        config = Config()
        faiss_manager = PersistentFAISSManager(config)
        sync_manager = SyncManager(config, faiss_manager)
        
        logger.info("Conectando a la base de datos...")
        pool = await sync_manager._get_pool()
        
        async with pool.acquire() as conn:
            # Obtener un solo embedding para debug
            logger.info("Consultando un embedding de ejemplo...")
            row = await conn.fetchrow("""
                SELECT embedding_id, embedding
                FROM embeddings
                LIMIT 1
            """)
            
            if row:
                logger.info(f"Embedding ID: {row['embedding_id']}")
                logger.info(f"Embedding tipo: {type(row['embedding'])}")
                logger.info(f"Embedding muestra: {str(row['embedding'])[:100]}...")  # Mostrar primeros 100 caracteres
                
                # Intentar parsear
                try:
                    embedding_array = sync_manager._parse_embedding(str(row['embedding']))
                    logger.info(f"Embedding parseado shape: {embedding_array.shape}")
                    logger.info(f"Embedding primeros 5 valores: {embedding_array[:5]}")
                except Exception as e:
                    logger.error(f"Error parseando embedding: {e}")
                    logger.error(f"Contenido completo del embedding: {row['embedding']}")
            else:
                logger.info("No se encontraron embeddings en la base de datos")
    
    except Exception as e:
        logger.error(f"Error durante la depuración: {e}")
        raise
    finally:
        if 'pool' in locals():
            await pool.close()

if __name__ == "__main__":
    asyncio.run(debug_embedding())