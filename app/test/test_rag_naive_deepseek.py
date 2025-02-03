import os
import re
import glob
import logging
import argparse
import pickle
import numpy as np
import pdfplumber
import requests  # Cambiamos openai por requests para llamadas API
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepseekRAG")

class DeepseekRAG:
    def __init__(self, document_dir, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 deepseek_model="deepseek-chat",
                 embedding_cache="embeddings.pkl",
                 deepseek_api_key=None,
                 password=None):
        """
        RAG adaptado para Deepseek:
        - Manejo mejorado de PDFs
        - Búsqueda eficiente con FAISS
        - Generación de respuestas usando Deepseek API
        - Cache de embeddings
        
        Args:
            document_dir: Directorio con documentos PDF.
            model_name: Modelo de embeddings de SentenceTransformer.
            deepseek_model: Modelo de Deepseek a utilizar.
            embedding_cache: Archivo para cache de embeddings.
            deepseek_api_key: API Key de Deepseek.
            password: Contraseña para PDFs protegidos (opcional).
        """
        self.document_dir = document_dir
        self.password = password
        self.embedding_cache = embedding_cache
        self.deepseek_model = deepseek_model
        self.api_base_url = "https://api.deepseek.com/v1"  # URL de la API de Deepseek

        # Configurar la API Key
        self.deepseek_api_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.deepseek_api_key:
            raise ValueError("Se requiere una API Key de Deepseek. Proporciónala como argumento o en la variable de entorno DEEPSEEK_API_KEY.")

        # Configurar el modelo de embeddings
        self.embedding_model = SentenceTransformer(model_name)

        # Cargar documentos y calcular embeddings
        self.documents = self._load_documents()
        self._compute_embeddings()

    def _load_documents(self):
        """Carga documentos PDF con preprocesamiento de texto."""
        documents = []
        pattern = os.path.join(self.document_dir, "**", "*.pdf")
        file_list = glob.glob(pattern, recursive=True)

        if not file_list:
            logger.warning("No se encontraron PDFs. Usando datos de ejemplo.")
            return [
                ("doc1", "Curazao es una isla caribeña con playas hermosas y cultura vibrante."),
                ("doc2", "Se recomienda visitar museos, disfrutar de la gastronomía local y realizar actividades acuáticas.")
            ]

        for filepath in file_list:
            try:
                with pdfplumber.open(filepath, password=self.password) as pdf:
                    text = "\n".join([self._clean_text(page.extract_text()) 
                                      for page in pdf.pages if page.extract_text()])
                if text.strip():
                    documents.append((os.path.basename(filepath), text))
                    logger.info(f"Cargado: {filepath} ({len(text)} caracteres)")
                else:
                    logger.warning(f"Texto vacío en: {filepath}")
            except Exception as e:
                logger.error(f"Error procesando {filepath}: {str(e)}")
        return documents

    def _clean_text(self, text):
        """Realiza una limpieza básica del texto extraído."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        return re.sub(r'[^\x20-\x7E]', '', text)

    def _compute_embeddings(self):
        """Calcula o carga embeddings desde caché usando FAISS."""
        if os.path.exists(self.embedding_cache):
            logger.info("Cargando embeddings desde caché...")
            with open(self.embedding_cache, "rb") as f:
                cache_data = pickle.load(f)
                self.doc_embeddings = cache_data['embeddings']
                self.faiss_index = faiss.deserialize_index(cache_data['faiss_index'])
            return

        logger.info("Calculando nuevos embeddings...")
        texts = [doc[1] for doc in self.documents]
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings.append(self.embedding_model.encode(batch, convert_to_tensor=True).cpu().numpy())
        self.doc_embeddings = np.concatenate(embeddings)
        
        self.faiss_index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))

        with open(self.embedding_cache, "wb") as f:
            pickle.dump({
                'embeddings': self.doc_embeddings,
                'faiss_index': faiss.serialize_index(self.faiss_index)
            }, f)

    def retrieve(self, query, top_k=3):
        """
        Recupera documentos relevantes usando FAISS.
        Args:
            query: Consulta de búsqueda.
            top_k: Número de documentos a recuperar.
        Returns:
            Lista de tuplas (nombre del documento, texto, score).
        """
        top_k = min(top_k, len(self.documents))
        if top_k <= 0:
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().numpy().astype(np.float32)
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = [(self.documents[i][0], self.documents[i][1], scores[0][j]) 
                   for j, i in enumerate(indices[0])]
        return results

    def generate_response(self, query, retrieved_docs, max_tokens=500):
        """
        Genera una respuesta utilizando la API de Deepseek.
        Args:
            query: Consulta original.
            retrieved_docs: Documentos recuperados.
            max_tokens: Número máximo de tokens para la respuesta.
        """
        if not retrieved_docs:
            return "No se encontró información relevante."

        context = "\n\n".join([f"Documento {doc[0]}:\n{doc[1]}" for doc in retrieved_docs])
        prompt = f"""Responde de manera útil y precisa a la siguiente consulta usando solo el contexto proporcionado.

Consulta: {query}

Contexto:
{context}

Respuesta:"""

        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.deepseek_model,
                "messages": [
                    {"role": "system", "content": "Eres un asistente turístico experto."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }

            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"Error en la API: {response.status_code} - {response.text}")
                return "Error al generar la respuesta."

        except Exception as e:
            logger.error(f"Error en generación de respuesta: {str(e)}")
            return "Hubo un error al generar la respuesta."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema RAG con Deepseek")
    parser.add_argument("--document_dir", default="curacao_information",
                        help="Directorio con documentos PDF")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Número de documentos a recuperar")
    parser.add_argument("--cache_file", default="embeddings.pkl",
                        help="Archivo de caché para embeddings")
    parser.add_argument("--deepseek_api_key", default=None,
                        help="API Key de Deepseek (variable: DEEPSEEK_API_KEY)")
    args = parser.parse_args()

    rag = DeepseekRAG(
        document_dir=args.document_dir,
        embedding_cache=args.cache_file,
        deepseek_api_key=args.deepseek_api_key
    )

    query = "¿Qué actividades turísticas puedo realizar en Curazao?"
    retrieved_docs = rag.retrieve(query, top_k=args.top_k)
    print("\nDocumentos recuperados:")
    for doc in retrieved_docs:
        print(f"\n► {doc[0]} (Score: {doc[2]:.4f})")
        print(f"{doc[1][:200]}...")

    response = rag.generate_response(query, retrieved_docs)
    print("\nRespuesta generada:")
    print(response)