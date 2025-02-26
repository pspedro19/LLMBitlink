# app/core/tourism/agents/agent_rag.py

import logging
from typing import Dict, Any, Optional, List

# Importar el servicio RAG que contiene la función query
from app.core.rag.services import RAGService

# Configurar logging
logger = logging.getLogger(__name__)

class RAGAgent:
    """
    Agente que utiliza el servicio RAG para responder consultas
    basadas en documentos almacenados.
    """
    
    def __init__(self):
        """Inicializar el agente conectándolo al servicio RAG."""
        try:
            self.rag_service = RAGService()
            logger.info("RAG Agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG Agent: {str(e)}")
            raise

    async def process_query(self, query_text: str, top_k: int = 3):
        try:
            logger.info(f"Processing RAG query: {query_text}")
            
            # Llamar directamente a la función query() del servicio RAG
            response = self.rag_service.query(query_text=query_text, top_k=top_k)
            
            # Solo incluir la respuesta generada, sin los documentos
            result = {
                "query": response.query,
                "response": response.response,
                "documents": []  # No incluir los documentos en la respuesta
            }
            print('\n----------------------------------------')
            print(result['query'])
            print(result['response'])
            print('----------------------------------------\n')
            logger.info(f"Query processed successfully with {len(response.documents)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG processing: {str(e)}")
            return {
                "query": query_text,
                "response": f"Error retrieving information: {str(e)}",
                "documents": []
            }