# core/rag/retriever.py
import time
import logging
from functools import wraps
from typing import Dict, List, Optional

from psycopg2.extras import RealDictCursor

from core.rag.config import Config
from core.rag.db_manager import DatabaseManager
from core.rag.document_processor import DocumentProcessor, DocumentInput
from core.rag.embeddings import EmbeddingsManager
from core.rag.faiss_manager import FAISSManager
from core.rag.metrics import RAGMetrics

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.db = DatabaseManager(config)
        self.embeddings_manager = EmbeddingsManager(config)
        self.faiss_manager = FAISSManager(config)
        self.metrics = RAGMetrics()

    def measure_time(self, histogram):
        """
        Decorador para medir el tiempo de ejecución de funciones críticas.
        """
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = fn(*args, **kwargs)
                elapsed = time.time() - start_time
                histogram.observe(elapsed)
                return result
            return wrapper
        return decorator

    @measure_time.__get__(object, object)  # Permite aplicar el decorador en el método
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

    def search(self, query: str, top_k: int = 5, model_name: str = 'miniLM', filters: Optional[dict] = None) -> List[Dict]:
        """
        Pipeline de búsqueda:
          1. Generación del embedding de la query.
          2. Búsqueda en FAISS.
          3. Consulta a la base de datos con filtros (si se requieren).
        """
        start_time = time.time()
        query_vector = self.embeddings_manager.generate_embeddings([query], model_name=model_name)[0]
        distances, faiss_ids = self.faiss_manager.search(query_vector, k=top_k)

        results = []
        conn = self.db.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                for i, faiss_id in enumerate(faiss_ids[0]):
                    where_clauses = ["e.faiss_index_id = %s"]
                    params = [int(faiss_id)]
                    if filters:
                        for key, value in filters.items():
                            where_clauses.append("d.metadata->>%s = %s")
                            params.extend([key, str(value)])
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
                    cur.execute(query_sql, params)
                    row = cur.fetchone()
                    if row:
                        row['similarity'] = float(distances[0][i])
                        results.append(row)
        finally:
            self.db.return_connection(conn)
        self.metrics.search_latency.observe(time.time() - start_time)
        return results
