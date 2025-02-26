# app/core/tourism/agents/agent_nlp.py
from typing import Dict, Any
from .base import BaseAgent, ChatState
from app.utils.logger import get_logger

# Importar la función y el modelo de request del text formatter
from app.core.recommender.text_formatter import get_text_formatted_recommendations, TextFormattedRequest

class NLPAgent(BaseAgent):
    """
    Agente NLP simplificado que toma el input del usuario, lo utiliza para construir
    un TextFormattedRequest y llama a la función get_text_formatted_recommendations para
    obtener recomendaciones en formato de texto plano.
    
    La respuesta se almacena en el estado bajo las claves 'formatted_recommendations' y 'response'.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
    
    async def process(self, state: ChatState) -> ChatState:
        """
        Procesa el input del usuario para generar recomendaciones formateadas.
        
        Pasos:
          1. Se toma el contenido original de 'user_input'.
          2. Se construye un TextFormattedRequest con ese contenido.
          3. Se llama asíncronamente a get_text_formatted_recommendations.
          4. Se extrae el texto resultante y se actualiza el estado.
        
        En caso de error, se registra y se asigna un mensaje de error.
        """
        # Tomar el input original (sin modificar) para la recomendación
        user_text = state.get("user_input", "")
        
        try:
            # Construir el request para obtener recomendaciones
            request = TextFormattedRequest(text=user_text)
            response = await get_text_formatted_recommendations(request)
            
            # Extraer el contenido en formato de texto desde el response (PlainTextResponse)
            if hasattr(response, 'body'):
                formatted_text = response.body.decode('utf-8')
            else:
                formatted_text = str(response)
            
            # Actualizar el estado con la respuesta obtenida
            state["formatted_recommendations"] = formatted_text
            state["response"] = formatted_text  # Opcional: asignar al campo principal 'response'
            self.logger.debug(f"Recomendaciones formateadas: {formatted_text[:100]}...")
        
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones formateadas: {e}")
            error_msg = "Error generating recommendations."
            state["formatted_recommendations"] = error_msg
            state["response"] = error_msg
        
        return state
