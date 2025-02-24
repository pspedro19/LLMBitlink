import os
import re
import glob
import logging
import argparse
import pickle
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama  # Nueva dependencia

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImprovedNaiveRAG")

class ImprovedNaiveRAG:
    def __init__(self, document_dir, 
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 local_model_path="lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/model.q8_0.gguf",
                 embedding_cache="embeddings.pkl",
                 n_ctx=2048,  # Tamaño del contexto del modelo
                 n_gpu_layers=0,  # Capas a ejecutar en GPU (0 = solo CPU)
                 password=None):
        """
        Args modificados:
            local_model_path: Ruta al modelo GGUF
            n_ctx: Tamaño del contexto del modelo
            n_gpu_layers: Capas a ejecutar en GPU (0=solo CPU)
            password: Contraseña para PDFs protegidos (por defecto None)
        """
        self.document_dir = document_dir
        self.embedding_cache = embedding_cache
        self.password = password

        # Cargar modelo local
        self.llm = Llama(
            model_path=local_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

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
                    text = "\n".join(
                        [self._clean_text(page.extract_text()) for page in pdf.pages if page.extract_text()]
                    )
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
        # Elimina múltiples espacios y saltos de línea
        text = re.sub(r'\s+', ' ', text).strip()
        # Elimina caracteres no imprimibles
        return re.sub(r'[^\x20-\x7E]', '', text)

    def _compute_embeddings(self):
        """Calcula o carga embeddings desde caché usando FAISS para búsquedas eficientes."""
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
            embeddings.append(
                self.embedding_model.encode(batch, convert_to_tensor=True).cpu().numpy()
            )
        self.doc_embeddings = np.concatenate(embeddings)
        # Crear índice FAISS para búsqueda rápida
        self.faiss_index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))

        # Guardar en caché
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

        query_embedding = self.embedding_model.encode(
            [query], convert_to_tensor=True
        ).cpu().numpy().astype(np.float32)
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        results = [
            (self.documents[i][0], self.documents[i][1], scores[0][j])
            for j, i in enumerate(indices[0])
        ]
        return results


    def generate_response(self, query, retrieved_docs, max_tokens=500, max_context_chars_per_doc=1500):
        """
        Genera respuestas en pasos separados para cada documento relevante.

        Args:
            query: Consulta del usuario.
            retrieved_docs: Lista de documentos recuperados [(nombre, contenido, score)].
            max_tokens: Máximo de tokens permitidos en la respuesta.
            max_context_chars_per_doc: Máximo de caracteres del documento a incluir.

        Returns:
            Respuesta combinada con información concisa de cada documento relevante.
        """
        if not retrieved_docs:
            return "No se encontró información relevante."

        respuestas = []
        
        for doc_name, doc_text, score in retrieved_docs:
            # Truncar contenido si es demasiado largo
            text = doc_text[:max_context_chars_per_doc] + " [...]" if len(doc_text) > max_context_chars_per_doc else doc_text

            prompt = f"""### Instrucción:
                Responde de manera clara usando SOLO la información proporcionada en el documento.

                ### Documento: {doc_name}
                {text}

                ### Consulta:
                {query}

                ### Respuesta:
            """
            try:
                response = self.llm(
                    prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=max_tokens,
                    stop=["###"]
                )
                respuesta_generada = response['choices'][0]['text'].strip()
                respuestas.append(f"\n**{doc_name}**:\n{respuesta_generada}")

            except Exception as e:
                logger.error(f"Error en generación con {doc_name}: {str(e)}")
                respuestas.append(f"\n**{doc_name}**: Error generando respuesta.")

        # Unir todas las respuestas generadas en una sola
        return "\n".join(respuestas)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema RAG con modelo local")
    parser.add_argument("--document_dir", default="curacao_information",
                        help="Directorio con documentos PDF")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Número de documentos a recuperar")
    parser.add_argument("--cache_file", default="embeddings.pkl",
                        help="Archivo de caché para embeddings")
    parser.add_argument("--model_path", 
                        default="C:/Users/Nabucodonosor/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                        help="Ruta al modelo GGUF")

    args = parser.parse_args()

    rag = ImprovedNaiveRAG(
        document_dir=args.document_dir,
        embedding_cache=args.cache_file,
        local_model_path=args.model_path,
        n_ctx=4096,
        n_gpu_layers=20
    )

    query = "What tourist activities can I do in Curaçao?"
    retrieved_docs = rag.retrieve(query, top_k=args.top_k)
    
    print("\nDocumentos recuperados:")
    for doc in retrieved_docs:
        print(f"\n► {doc[0]} (Score: {doc[2]:.4f})")
        print(f"{doc[1][:200]}...")

    response = rag.generate_response(query, retrieved_docs)
    print("\nRespuesta generada:")
    print(response)
