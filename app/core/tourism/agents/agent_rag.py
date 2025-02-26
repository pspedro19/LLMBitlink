# app/core/tourism/agents/agent_rag.py
from .base import BaseAgent, ChatState
import logging

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    def __init__(self, retriever):
        """
        Se espera que 'retriever' sea una instancia de RAGRetriever que implemente
        el método search(query, top_k) de forma similar a la lógica usada en /rag/query.
        """
        super().__init__()
        self.retriever = retriever

    async def process(self, state: ChatState) -> ChatState:
        """
        Ejecuta exclusivamente la lógica de consulta (query) del sistema RAG.
        Toma la consulta del usuario, la procesa y actualiza el estado con el contexto recuperado.
        """
        query = state["user_input"]
        top_k = 3  # Puedes ajustar el número de resultados a recuperar

        try:
            # Invoca la lógica de búsqueda de documentos
            results = self.retriever.search(query, top_k=top_k)
            # Si se recuperan resultados, se genera un contexto combinando la información
            if results:
                # Ejemplo: se concatena el título y parte del contenido de cada resultado
                context = "\n\n".join(
                    [f"Documento {res['title']}:\n{res['content']}" for res in results]
                )
            else:
                context = "No se encontró información relevante para tu consulta."
            state["rag_context"] = context
            state["recommendations"] = [context]
        except Exception as e:
            logger.error(f"Error en RAGAgent al procesar la consulta: {e}")
            state["rag_context"] = "Error al recuperar información."
        return state
