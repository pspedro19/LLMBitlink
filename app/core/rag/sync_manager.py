# core/rag/sync_manager.py
import asyncio
import logging
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)

class SyncManager:
    def __init__(self, config, faiss_manager):
        self.config = config
        self.faiss_manager = faiss_manager
        self._auto_sync_task = None
        self.is_syncing = False
        self.pool = None
        
    async def _get_pool(self):
        """Obtiene o crea el pool de conexiones usando el event loop actual."""
        if self.pool is None:
            async def init_connection(conn):
                await conn.set_type_codec(
                    'vector',
                    encoder=lambda v: v,
                    decoder=lambda v: [float(x) for x in v.strip()[1:-1].split(',')],
                    schema='public',
                    format='text'
                )
            self.pool = await asyncpg.create_pool(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT,
                init=init_connection,
                loop=asyncio.get_running_loop()
            )
        return self.pool

    def _parse_embedding(self, embedding) -> np.ndarray:
        """
        Parsea el embedding obtenido desde PostgreSQL y retorna un np.ndarray
        de tipo float32 y con shape (1, 384). Además, verifica que el vector no contenga NaN o inf.
        """
        if isinstance(embedding, (list, np.ndarray)):
            vector = np.array(embedding, dtype=np.float32)
        elif isinstance(embedding, str):
            cleaned = embedding.strip()[1:-1]
            try:
                numbers = [float(x.strip()) for x in cleaned.split(',')]
            except Exception as e:
                raise ValueError(f"Error al convertir embedding a float: {e}")
            vector = np.array(numbers, dtype=np.float32)
        else:
            raise ValueError(f"Tipo de embedding no soportado: {type(embedding)}")
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        if vector.shape[1] != self.faiss_manager.dimension:
            raise ValueError(f"Dimensión incorrecta: {vector.shape[1]} (se esperaba {self.faiss_manager.dimension})")
        
        if not np.all(np.isfinite(vector)):
            raise ValueError("El vector contiene valores no finitos (NaN o inf)")
        
        return vector

    async def _get_all_embeddings(self) -> List[Dict]:
        """Obtiene todos los embeddings de PostgreSQL."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    e.embedding_id,
                    e.embedding,
                    e.faiss_index_id,
                    c.chunk_id,
                    c.content,
                    d.title
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.chunk_id
                JOIN documents d ON c.doc_id = d.doc_id
                ORDER BY e.created_at
            """)
            
            embeddings = []
            for row in rows:
                try:
                    embedding_array = self._parse_embedding(row['embedding'])
                    logger.info(f"Embedding {row['embedding_id']} shape: {embedding_array.shape}")
                    embeddings.append({
                        'embedding_id': row['embedding_id'],
                        'embedding': embedding_array,
                        'faiss_index_id': row['faiss_index_id'],
                        'chunk_id': row['chunk_id'],
                        'content': row['content'],
                        'title': row['title']
                    })
                except Exception as e:
                    logger.error(f"Error procesando embedding {row['embedding_id']}: {e}")
                    continue
            
            return embeddings

    async def _update_chunks_indexed(self, chunk_ids: List[str]):
        """Actualiza la tabla chunks, marcando los registros especificados como indexados (needs_indexing = FALSE)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany("""
                UPDATE chunks 
                SET needs_indexing = FALSE 
                WHERE chunk_id = $1
            """, [(cid,) for cid in chunk_ids])
            logger.info(f"Marcados {len(chunk_ids)} chunks como indexados")

    async def synchronize(self) -> Tuple[int, int]:
        """Sincroniza los embeddings entre PostgreSQL y FAISS de forma completa."""
        if self.is_syncing:
            logger.warning("Ya hay una sincronización en progreso")
            return 0, 0

        self.is_syncing = True
        processed = 0
        failed = 0

        try:
            # Reinicializar el índice FAISS para sincronización completa
            self.faiss_manager.reset_index()

            embeddings = await self._get_all_embeddings()
            if not embeddings:
                logger.info("No hay embeddings para sincronizar")
                return 0, 0

            vectors = np.vstack([emb['embedding'] for emb in embeddings])
            logger.info(f"Preparados {len(vectors)} vectores para FAISS con shape {vectors.shape}")

            try:
                faiss_ids = self.faiss_manager.add_vectors(vectors)
                
                id_mappings = [
                    (emb['embedding_id'], faiss_id)
                    for emb, faiss_id in zip(embeddings, faiss_ids)
                ]
                await self._update_faiss_ids(id_mappings)
                
                # Actualizamos los chunks correspondientes para marcar que ya se indexaron
                chunk_ids = [emb['chunk_id'] for emb in embeddings]
                await self._update_chunks_indexed(chunk_ids)
                
                processed = len(embeddings)
                logger.info(f"Sincronización exitosa: {processed} embeddings procesados")
                
            except Exception as e:
                logger.error(f"Error al procesar embeddings: {e}")
                failed = len(embeddings)
                raise

        except Exception as e:
            logger.error(f"Error durante la sincronización: {e}")
            raise
        finally:
            self.is_syncing = False
            
        return processed, failed

    async def _update_faiss_ids(self, id_mappings: List[Tuple[str, int]]):
        """Actualiza los IDs de FAISS en PostgreSQL."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany("""
                UPDATE embeddings 
                SET faiss_index_id = $2 
                WHERE embedding_id = $1
            """, id_mappings)
            logger.info(f"Actualizados {len(id_mappings)} IDs en PostgreSQL")

    async def verify_sync_status(self) -> dict:
        """Verifica el estado de sincronización."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            pg_count = await conn.fetchval("SELECT COUNT(*) FROM embeddings")
            faiss_count = self.faiss_manager.get_index_info()['total_vectors']
            
            return {
                'postgres_embeddings': pg_count,
                'faiss_vectors': faiss_count,
                'is_synced': pg_count == faiss_count,
                'last_check': datetime.now().isoformat()
            }

    def get_system_status(self) -> dict:
        """Obtiene el estado completo del sistema."""
        return {
            'index_info': self.faiss_manager.get_index_info(),
            'is_syncing': self.is_syncing,
            'auto_sync_running': self._auto_sync_task is not None
        }
