# core/rag/db_manager.py
from typing import Dict, List
import logging
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor, Json  # <-- Agregamos Json
from psycopg2.pool import ThreadedConnectionPool
from tenacity import retry, stop_after_attempt, wait_exponential

from core.rag.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        # Creamos un pool de conexiones: mínimo 5 y máximo config.MAX_CONNECTIONS
        self.pool = ThreadedConnectionPool(
            minconn=5,
            maxconn=config.MAX_CONNECTIONS,
            dsn=config.db_url
        )

    def get_connection(self):
        return self.pool.getconn()

    def return_connection(self, conn):
        self.pool.putconn(conn)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def insert_document(self, title: str, content: str, metadata: Dict) -> str:
        """
        Inserta un nuevo documento en la tabla 'documents' y retorna el doc_id.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (title, original_content, metadata)
                    VALUES (%s, %s, %s)
                    RETURNING doc_id
                """, (title, content, Json(metadata)))  # <-- Aquí usamos Json(metadata)
                doc_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"Documento insertado con doc_id={doc_id}")
            return doc_id
        except Exception as e:
            conn.rollback()
            logger.error(f"Error insertando documento: {e}")
            raise
        finally:
            self.return_connection(conn)

    def insert_chunks(self, doc_id: str, chunks: List[Dict]) -> List[str]:
        """
        Inserta múltiples chunks de forma individual.
        """
        logger.info(f"Insertando {len(chunks)} chunks para doc_id={doc_id}")
        chunk_ids = []
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    cur.execute("""
                        INSERT INTO chunks 
                        (doc_id, content, chunk_number, start_char, end_char, page_number, token_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING chunk_id
                    """, (
                        doc_id,
                        chunk['content'],
                        i,
                        chunk.get('start'),
                        chunk.get('end'),
                        chunk.get('page_number'),
                        chunk.get('token_count')
                    ))
                    chunk_ids.append(cur.fetchone()[0])
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error insertando chunks: {e}")
            raise
        finally:
            self.return_connection(conn)
        return chunk_ids

    def bulk_insert_chunks(self, doc_id: str, chunks: List[Dict]):
        """
        Inserción masiva de chunks para optimizar el rendimiento.
        """
        logger.info(f"Bulk insert de {len(chunks)} chunks para doc_id={doc_id}")
        query = """
            INSERT INTO chunks 
            (doc_id, content, chunk_number, start_char, end_char)
            VALUES (%s, %s, %s, %s, %s)
        """
        params = [(doc_id, c['content'], i, c.get('start'), c.get('end')) for i, c in enumerate(chunks)]
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                execute_batch(cur, query, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error en bulk insert de chunks: {e}")
            raise
        finally:
            self.return_connection(conn)

    def insert_embedding(self, chunk_id: str, model_name: str, vector, faiss_index_id: int):
        """
        Inserta un embedding asociado a un chunk en la tabla 'embeddings'.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO embeddings
                    (chunk_id, model_name, embedding, faiss_index_id)
                    VALUES (%s, %s, %s, %s)
                """, (chunk_id, model_name, vector.tolist(), faiss_index_id))
            conn.commit()
            logger.debug(f"Insertado embedding (faiss_index_id={faiss_index_id}) para chunk_id={chunk_id}.")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error insertando embedding: {e}")
            raise
        finally:
            self.return_connection(conn)
