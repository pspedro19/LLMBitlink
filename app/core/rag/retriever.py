# core/rag/retriever.py
import time
import logging
import asyncio  # Para poder llamar a funciones asíncronas
import numpy as np
from functools import wraps
from typing import Dict, List, Optional

# No usamos psycopg2 ya que trabajaremos de forma asíncrona con asyncpg

from core.rag.config import Config
from core.rag.db_manager import DatabaseManager
from core.rag.document_processor import DocumentProcessor, DocumentInput
from core.rag.embeddings import EmbeddingsManager
from core.rag.faiss_manager import FAISSManager
from core.rag.metrics import RAGMetrics

logger = logging.getLogger(__name__)

def measure_time(histogram_getter):
    """
    Decorador para medir el tiempo de ejecución de funciones críticas.
    Recibe como parámetro una función (histogram_getter) que, dado 'self',
    retorna el histograma sobre el cual registrar el tiempo transcurrido.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = fn(self, *args, **kwargs)
            elapsed = time.time() - start_time
            histogram = histogram_getter(self)
            histogram.observe(elapsed)
            return result
        return wrapper
    return decorator

class RAGRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.db = DatabaseManager(config)
        self.embeddings_manager = EmbeddingsManager(config)
        self.faiss_manager = FAISSManager(config)
        self.metrics = RAGMetrics()

    @measure_time(lambda self: self.metrics.processing_time)
    def process_document(self, doc_input: DocumentInput) -> str:
        """
        Pipeline de ingesta:
          1. Validación e inserción del documento.
          2. Creación de chunks.
          3. Inserción de chunks (individual o bulk).
          4. Generación de embeddings.
          5. Indexación en FAISS.
          6. Almacenamiento de embeddings en la DB.
        """
        # Validación del documento
        processor = DocumentProcessor(self.config)
        valid_doc = processor.validate_document(doc_input)
        
        logger.info(f"== Iniciando procesamiento del documento: {valid_doc.title} ==")
        doc_id = self.db.insert_document(valid_doc.title, valid_doc.content, valid_doc.metadata)
        
        # Creación de chunks
        chunks = processor.create_chunks(valid_doc.content)
        chunk_ids = self.db.insert_chunks(doc_id, chunks)
        # Alternativamente, se puede usar bulk_insert:
        # self.db.bulk_insert_chunks(doc_id, chunks)
        
        # Generación de embeddings
        texts = [chunk['content'] for chunk in chunks]
        vectors = self.embeddings_manager.generate_embeddings(texts, model_name=valid_doc.model_name)
        
        # Indexación en FAISS
        faiss_ids = self.faiss_manager.add_vectors(vectors)
        
        # Almacenamiento de embeddings en la DB
        self.embeddings_manager.store_embeddings(self.db, chunk_ids, vectors, faiss_ids, model_name=valid_doc.model_name)
        
        self.metrics.document_counter.inc()  # Se incrementa el contador de documentos procesados
        logger.info(f"Documento procesado correctamente: doc_id={doc_id}")
        return doc_id

    async def search_async(
        self, query: str, top_k: int = 5, model_name: str = 'miniLM', filters: Optional[dict] = None
    ) -> List[Dict]:
        """
        Pipeline de búsqueda asíncrona:
          1. Generación del embedding de la query.
          2. Búsqueda en FAISS.
          3. Consulta a la base de datos con filtros (si se requieren).
        """
        start_time = time.time()
        # Generar el vector de la query
        query_vector = self.embeddings_manager.generate_embeddings([query], model_name=model_name)[0]
        distances, faiss_ids = self.faiss_manager.search(query_vector, k=top_k)

        results = []
        # Si la búsqueda en FAISS no encontró vecinos, retorna lista vacía
        if faiss_ids.size == 0:
            logger.warning("El índice está vacío. Retornando resultados vacíos.")
            self.metrics.search_latency.observe(time.time() - start_time)
            return results

        # Obtener el pool de conexiones asíncronas
        pool = await self.db._get_pool()
        async with pool.acquire() as conn:
            for i, faiss_id in enumerate(faiss_ids[0]):
                # Construir la cláusula WHERE y los parámetros de forma dinámica
                where_clauses = [f"e.faiss_index_id = $1"]
                params = [int(faiss_id)]
                if filters:
                    idx = 2
                    for key, value in filters.items():
                        where_clauses.append(f"d.metadata->>'{key}' = ${idx}")
                        params.append(str(value))
                        idx += 1
                query_sql = f"""
                    SELECT 
                        d.doc_id,
                        d.title,
                        d.metadata,
                        c.content,
                        c.chunk_number,
                        e.faiss_index_id
                    FROM embeddings e
                    JOIN chunks c ON e.chunk_id = c.chunk_id
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE {' AND '.join(where_clauses)}
                """
                row = await conn.fetchrow(query_sql, *params)
                if row:
                    row = dict(row)
                    row['similarity'] = float(distances[0][i])
                    results.append(row)
        self.metrics.search_latency.observe(time.time() - start_time)
        return results

    def search(
        self, query: str, top_k: int = 5, model_name: str = 'miniLM', filters: Optional[dict] = None
    ) -> List[Dict]:
        """
        Método síncrono que envuelve la búsqueda asíncrona.
        """
        return asyncio.run(self.search_async(query, top_k, model_name, filters))

    def update_index_from_db(self):
        """
        Actualiza el índice FAISS con embeddings no sincronizados desde la base de datos.
        Si no hay embeddings pendientes y el índice está vacío, se reconstruye el índice
        consultando todos los embeddings existentes en la base de datos.
        """
        unsynced = asyncio.run(self.db.get_unsynchronized_embeddings())
        if unsynced:
            vectors = np.array([item['embedding'] for item in unsynced])
            faiss_ids = self.faiss_manager.add_vectors(vectors)
            id_pairs = [(item['embedding_id'], faiss_id) for item, faiss_id in zip(unsynced, faiss_ids)]
            asyncio.run(self.db.update_faiss_ids(id_pairs))
            logger.info(f"Sincronizados {len(id_pairs)} embeddings nuevos al índice FAISS.")
        else:
            index_info = self.faiss_manager.get_index_info()
            if index_info.get('total_vectors', 0) == 0:
                logger.info("FAISS index vacío. Intentando reconstruirlo con todos los embeddings de la DB.")
                # Se asume que agregas este método en DatabaseManager:
                # async def get_all_embeddings(self) -> List[Dict]:
                all_embeddings = asyncio.run(self.db.get_all_embeddings())
                if all_embeddings:
                    vectors = np.array([item['embedding'] for item in all_embeddings])
                    faiss_ids = self.faiss_manager.add_vectors(vectors)
                    id_pairs = [(item['embedding_id'], faiss_id) for item, faiss_id in zip(all_embeddings, faiss_ids)]
                    asyncio.run(self.db.update_faiss_ids(id_pairs))
                    logger.info(f"Reconstruido el índice FAISS con {len(id_pairs)} embeddings.")
                else:
                    logger.info("No se encontraron embeddings en la base de datos para reconstruir el índice.")
            else:
                logger.info("No hay embeddings pendientes de sincronización.")
