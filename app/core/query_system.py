# /core/query_system.py
import asyncio
import numpy as np
from core.rag.config import Config
from core.rag.faiss_manager import PersistentFAISSManager
from core.rag.sync_manager import SyncManager

# Importa el modelo para generar embeddings (Sentence Transformers)
from sentence_transformers import SentenceTransformer
# Importa la pipeline de Hugging Face para Question Answering
from transformers import pipeline

class QuerySystem:
    def __init__(self, config: Config = None):
        """
        Inicializa el sistema de consulta:
         - Configuración y gestión de FAISS y PostgreSQL.
         - Modelo para generar embeddings de la pregunta.
         - Pipeline para generar respuesta a partir de un contexto.
        """
        self.config = config or Config()
        self.faiss_manager = PersistentFAISSManager(self.config)
        self.sync_manager = SyncManager(self.config, self.faiss_manager)
        
        # Carga un modelo preentrenado para generar embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Inicializa la pipeline para Question Answering
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convierte un texto (por ejemplo, la consulta en lenguaje natural) en un vector.
        """
        embedding = self.embedding_model.encode(text)
        return np.array(embedding, dtype=np.float32).reshape(1, -1)
    
    async def synchronize(self):
        """
        Ejecuta la sincronización completa (actualiza FAISS con los embeddings pendientes en PostgreSQL).
        """
        await self.sync_manager.synchronize()
    
    async def retrieve_context(self, indices) -> str:
        """
        A partir de los índices (o posiciones) obtenidos de FAISS, consulta en PostgreSQL
        los contenidos asociados y los concatena en un único contexto.
        
        NOTA: Esta implementación asume que los índices corresponden a posiciones en un
        listado ordenado de chunks. Deberás ajustar la consulta según la relación entre
        los índices de FAISS y tus registros reales.
        """
        pool = await self.sync_manager._get_pool()
        context_chunks = []
        
        async with pool.acquire() as conn:
            for pos in indices[0]:
                row = await conn.fetchrow(
                    "SELECT content FROM chunks ORDER BY chunk_number LIMIT 1 OFFSET $1",
                    pos
                )
                if row:
                    context_chunks.append(row["content"])
        return " ".join(context_chunks)
    
    async def query(self, question: str, k: int = 5) -> dict:
        """
        Realiza una consulta en lenguaje natural:
         1. Convierte la pregunta en un vector.
         2. Busca los k chunks más relevantes en FAISS.
         3. Recupera el contexto asociado a esos chunks desde PostgreSQL.
         4. Usa la pipeline de QA para generar una respuesta basada en la pregunta y el contexto.
         
        Retorna un diccionario con la pregunta, el contexto recuperado, la respuesta y detalles de la búsqueda.
        """
        # Paso 1: Generar embedding para la consulta
        query_vector = self.embed_text(question)
        
        # Paso 2: Buscar en FAISS
        distances, indices = self.faiss_manager.search(query_vector, k=k)
        
        # Paso 3: Recuperar el contexto asociado a los chunks encontrados
        context = await self.retrieve_context(indices)
        
        # Paso 4: Generar respuesta usando la pipeline de QA
        qa_input = {"question": question, "context": context}
        answer = self.qa_pipeline(qa_input)
        
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "distances": distances.tolist(),
            "indices": indices.tolist()
        }

# Permite ejecutar el módulo directamente para pruebas
if __name__ == "__main__":
    async def main():
        qs = QuerySystem()
        # Es importante sincronizar antes de realizar la consulta para tener actualizados los embeddings
        await qs.synchronize()
        consulta = "¿Qué información turística me puedes brindar sobre Curazao?"
        resultado = await qs.query(consulta, k=3)
        print("Consulta:", resultado["question"])
        print("Contexto recuperado:", resultado["context"])
        print("Respuesta generada:", resultado["answer"])
        print("Detalles de búsqueda - Distancias:", resultado["distances"], "Indices:", resultado["indices"])
    
    asyncio.run(main())
