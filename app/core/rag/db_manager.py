# core/rag/db_manager.py
import asyncpg
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from core.rag.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        self.pool = None
        
    async def _get_pool(self):
        """Obtiene o crea el pool de conexiones."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                user=self.config.DB_USER,
                password=self.config.DB_PASSWORD,
                database=self.config.DB_NAME,
                host=self.config.DB_HOST,
                port=self.config.DB_PORT
            )
        return self.pool
        
    async def get_unsynchronized_embeddings(self) -> List[Dict[str, Any]]:
        """Obtiene embeddings que no están sincronizados con FAISS."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT embedding_id, embedding, chunk_id
                FROM embeddings
                WHERE faiss_index_id IS NULL
                ORDER BY created_at
                LIMIT 10000
            """)
            
            return [
                {
                    'embedding_id': row['embedding_id'],
                    'embedding': np.array(row['embedding']),
                    'chunk_id': row['chunk_id']
                }
                for row in rows
            ]
            
    async def update_faiss_ids(self, id_pairs: List[tuple]):
        """Actualiza los IDs de FAISS en la base de datos."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany("""
                UPDATE embeddings
                SET faiss_index_id = $2
                WHERE embedding_id = $1
            """, id_pairs)
            
    async def get_total_embeddings_count(self) -> int:
        """Obtiene el número total de embeddings en la base de datos."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM embeddings")
            
    async def insert_embedding(
        self,
        chunk_id: str,
        model_name: str,
        vector: np.ndarray,
        faiss_index_id: Optional[int] = None
    ):
        """Inserta un nuevo embedding en la base de datos."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO embeddings (chunk_id, model_name, embedding, faiss_index_id)
                VALUES ($1, $2, $3, $4)
            """, chunk_id, model_name, vector.tolist(), faiss_index_id)