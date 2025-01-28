"""Utilidad para formatear respuestas HTML."""

def _format_chat_response(response: str) -> str:
    """
    Formatea una respuesta conversacional en HTML.
    
    Args:
        response: Texto de la respuesta
    """
    return f"""
    <div class="message message-bot">
        <div class="message-content">
            <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">
                {response}
            </p>
        </div>
    </div>
    """

def _format_mixed_response(chat_response: str, recommendations_str: str) -> str:
    """
    Formatea una respuesta que combina chat y recomendaciones.
    
    Args:
        chat_response: Texto conversacional
        recommendations_str: HTML de recomendaciones
    """
    return f"""
    <div class="mixed-response">
        <div class="message message-bot">
            <div class="message-content">
                <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">
                    {chat_response}
                </p>
            </div>
        </div>
        <div class="recommendations-section">
            {recommendations_str}
        </div>
    </div>
    """

def _format_error_response(error_type: str) -> str:
    """
    Formatea mensajes de error según el tipo.
    
    Args:
        error_type: Tipo de error ocurrido
    """
    error_messages = {
        "openai": """
            Lo siento, estoy teniendo problemas para procesar tu solicitud en este momento.
            ¿Te gustaría ver algunas de nuestras recomendaciones populares mientras tanto?
        """,
        "recommendation": """
            Disculpa, no pude encontrar recomendaciones específicas para tu solicitud.
            Permíteme sugerirte algunas alternativas interesantes.
        """,
        "mixed": """
            Parece que hubo un problema procesando tu solicitud completa.
            ¿Podrías especificar qué parte te interesa más: las recomendaciones o la información general?
        """,
        "general": """
            Lo siento, hubo un problema al procesar tu solicitud.
            ¿Podrías reformularla de otra manera?
        """
    }
    
    message = error_messages.get(error_type, error_messages["general"])
    return f"""
    <div class="message message-bot error-message">
        <div class="message-content">
            <p style="font-size: 1.2em; color: #333; margin-bottom: 16px;">
                {message}
            </p>
        </div>
    </div>
    """