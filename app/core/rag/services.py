# app/core/rag/services.py

from fastapi import HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
import pickle
import re
from pathlib import Path
import glob
import openai
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG-Services")

class DocumentResponse(BaseModel):
    document_name: str
    content: str
    score: float

class QueryResponse(BaseModel):
    query: str
    documents: List[DocumentResponse]
    response: str

class RAGService:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 local_model_path: str = "/app/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
                 embedding_cache: str = "data/embeddings.pkl",
                 n_ctx: int = 4096,
                 n_gpu_layers: int = 0):  # Cambiado a 0 para forzar CPU
        
        self.embedding_cache = Path(embedding_cache)
        self.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        self.use_openai = False

        # Inicializar modelos
        self.embedding_model = SentenceTransformer(model_name)
        
        # Intentar cargar el modelo local primero
        try:
            from llama_cpp import Llama
            logger.info(f"Intentando cargar modelo local desde: {local_model_path}")
            self.llm = Llama(
                model_path=local_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            logger.info("Modelo local cargado exitosamente")
            self.use_openai = False
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo local: {str(e)}")
            logger.info("Utilizando OpenAI como alternativa")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("No se encontró la clave API de OpenAI")
                raise ValueError("Se requiere OPENAI_API_KEY cuando el modelo local no está disponible")
            
            # Configurar OpenAI
            self.openai_client = openai.Client(api_key=openai_api_key)
            self.use_openai = True
        
        # Inicializar FAISS
        self.documents = []
        self.faiss_index = None
        self._load_or_create_index()

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        return re.sub(r'[^\x20-\x7E]', '', text)

    def _load_or_create_index(self):
        if self.embedding_cache.exists():
            logger.info("Loading cached embeddings...")
            with open(self.embedding_cache, "rb") as f:
                cache_data = pickle.load(f)
                self.documents = cache_data['documents']
                self.faiss_index = faiss.deserialize_index(cache_data['faiss_index'])
                logger.info(f"Loaded {len(self.documents)} documents from cache")
        else:
            logger.info("No cached embeddings found. Creating new index...")
            self._create_empty_index()

    def _create_empty_index(self):
        dummy_embedding = self.embedding_model.encode(["dummy text"])
        self.faiss_index = faiss.IndexFlatIP(dummy_embedding.shape[1])

    def bulk_load_documents(self, base_path: str = "data/knowledge_base/curaçao_information") -> dict:
        """
        Carga todos los documentos PDF de los directorios especificados.
        
        Args:
            base_path: Ruta base donde se encuentran los directorios de documentos
        
        Returns:
            dict: Resultado del proceso de carga
        """
        base_path = Path(base_path)
        if not base_path.exists():
            raise HTTPException(status_code=400, detail=f"Directory {base_path} not found")

        # Definir subdirectorios
        subdirs = [
            "0_Introduccion",
            "1_Atracciones",
            "2_Cultura_e_Historia",
            "3_Consejos_Practicos"
        ]

        total_docs = 0
        processed_docs = 0
        failed_docs = []
        embeddings_list = []

        # Reset current documents and index
        self.documents = []
        self._create_empty_index()

        for subdir in subdirs:
            dir_path = base_path / subdir
            if not dir_path.exists():
                logger.warning(f"Directory {dir_path} not found, skipping...")
                continue

            pdf_files = list(dir_path.glob("**/*.pdf"))
            total_docs += len(pdf_files)

            for pdf_path in pdf_files:
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        text = "\n".join([
                            self._clean_text(page.extract_text())
                            for page in pdf.pages
                            if page.extract_text()
                        ])

                    if text.strip():
                        # Generar embedding
                        embedding = self.embedding_model.encode([text])[0]
                        embeddings_list.append(embedding)
                        
                        # Almacenar documento
                        rel_path = pdf_path.relative_to(base_path)
                        self.documents.append((str(rel_path), text))
                        processed_docs += 1
                        
                        logger.info(f"Processed: {rel_path} ({len(text)} chars)")
                    else:
                        failed_docs.append((str(pdf_path), "No text extracted"))
                        logger.warning(f"No text extracted from {pdf_path}")

                except Exception as e:
                    failed_docs.append((str(pdf_path), str(e)))
                    logger.error(f"Error processing {pdf_path}: {str(e)}")

        if processed_docs > 0:
            # Convertir lista de embeddings a array numpy
            embeddings_array = np.vstack(embeddings_list)
            
            # Añadir al índice FAISS
            self.faiss_index.add(embeddings_array.astype(np.float32))
            
            # Guardar en caché
            self._save_index()
            
            logger.info(f"Successfully processed {processed_docs} documents")
            
        return {
            "total_documents": total_docs,
            "processed_documents": processed_docs,
            "failed_documents": failed_docs,
            "cache_path": str(self.embedding_cache)
        }

    def _save_index(self):
        with open(self.embedding_cache, "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'faiss_index': faiss.serialize_index(self.faiss_index)
            }, f)
        logger.info(f"Index saved to {self.embedding_cache}")

    def query(self, query_text: str, top_k: int = 3) -> QueryResponse:
        if not self.documents:
            raise HTTPException(status_code=400, detail="No documents loaded in the index")

        # Obtener embedding de la consulta
        query_embedding = self.embedding_model.encode([query_text]).astype(np.float32)
        
        # Buscar documentos similares
        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Preparar documentos recuperados
        retrieved_docs = [
            DocumentResponse(
                document_name=self.documents[i][0],
                content=self.documents[i][1][:1500] + "..." if len(self.documents[i][1]) > 1500 else self.documents[i][1],
                score=float(scores[0][idx])
            )
            for idx, i in enumerate(indices[0])
        ]

        # Generar respuesta con LLM
        context = "\n\n".join([
            f"Document: {doc.document_name}\n{doc.content}"
            for doc in retrieved_docs
        ])

        prompt = f"""### Instruction:
            Based on the following documents, answer the query concisely and accurately.
            Provide specific information and examples when available.

            {context}

            ### Query:
            {query_text}

            ### Response:
        """

        # Usar OpenAI si el modelo local no está disponible
        if self.use_openai:
            try:
                logger.info("Usando OpenAI para generar respuesta")
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error al usar OpenAI: {str(e)}")
                answer = f"Error al generar respuesta: {str(e)}"
        else:
            # Usar el modelo local
            try:
                response = self.llm(
                    prompt,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=500,
                    stop=["###"]
                )
                answer = response['choices'][0]['text'].strip()
            except Exception as e:
                logger.error(f"Error al usar el modelo local: {str(e)}")
                answer = f"Error al generar respuesta: {str(e)}"

        return QueryResponse(
            query=query_text,
            documents=retrieved_docs,
            response=answer
        )