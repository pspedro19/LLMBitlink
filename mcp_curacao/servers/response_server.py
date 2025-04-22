# mcp_curacao/servers/response_server.py
from mcp.server.fastmcp import FastMCP
import os
import json
import random
import logging
import sys
import importlib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("response-server")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RESPONSE_SERVER_ID, 
    OPENAI_API_KEY, 
    ANTHROPIC_API_KEY, 
    get_llm_provider
)

# Importar plantillas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.templates import ResponseTemplates

# Inicializar servidor MCP
mcp = FastMCP(RESPONSE_SERVER_ID)

# Inicializar plantillas
templates = ResponseTemplates()

# Determinar proveedor LLM disponible
LLM_PROVIDER = get_llm_provider()
logger.info(f"Proveedor LLM activo: {LLM_PROVIDER}")

# Inicializar clientes LLM
openai_client = None
claude_client = None

if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
    try:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("Cliente OpenAI inicializado correctamente")
    except Exception as e:
        logger.error(f"Error inicializando OpenAI: {e}")
        LLM_PROVIDER = "templates"

if LLM_PROVIDER == "claude" and ANTHROPIC_API_KEY:
    try:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Cliente Claude inicializado correctamente")
    except Exception as e:
        logger.error(f"Error inicializando Claude: {e}")
        LLM_PROVIDER = "templates"

@mcp.tool()
def generate_response(
    query: str,
    user_context: dict,
    recommendations: list = None,
    rag_info: dict = None,
    use_llm: bool = True
) -> dict:
    """
    Genera una respuesta natural basada en la información disponible.
    
    Args:
        query: Consulta del usuario
        user_context: Contexto del usuario
        recommendations: Recomendaciones disponibles
        rag_info: Información del sistema RAG
        use_llm: Si usar LLM o plantillas
        
    Returns:
        Respuesta generada con metadatos
    """
    # Determinar qué información falta
    preferences = user_context.get("preferences", {})
    missing_info = []
    required_fields = ["interests", "duration", "budget"]
    
    for field in required_fields:
        if field not in preferences or not preferences[field]:
            missing_info.append(field)
    
    # Si falta información, generar respuesta para obtenerla
    if missing_info:
        if "interests" in missing_info:
            return {
                "response": templates.get_interests_question(),
                "missing_info": missing_info,
                "used_llm": False
            }
        elif "duration" in missing_info:
            return {
                "response": templates.get_duration_question(),
                "missing_info": missing_info,
                "used_llm": False
            }
        elif "budget" in missing_info:
            return {
                "response": templates.get_budget_question(preferences.get("duration", 0)),
                "missing_info": missing_info,
                "used_llm": False
            }
    
    # Si tenemos recomendaciones, generar respuesta con ellas
    if recommendations and len(recommendations) > 0:
        if use_llm and LLM_PROVIDER != "templates":
            # Usar LLM para generar respuesta
            llm_response = _generate_llm_response(
                query=query,
                user_context=user_context,
                recommendations=recommendations,
                rag_info=rag_info
            )
            
            return {
                "response": llm_response,
                "missing_info": [],
                "used_llm": True
            }
        else:
            # Usar plantillas
            formatted_response = templates.format_recommendations(recommendations, rag_info)
            return {
                "response": formatted_response,
                "missing_info": [],
                "used_llm": False
            }
    
    # Si no hay recomendaciones pero tenemos todos los datos requeridos
    if not missing_info and not recommendations:
        return {
            "response": templates.get_processing_message(),
            "missing_info": [],
            "used_llm": False
        }
    
    # Respuesta fallback
    return {
        "response": templates.get_fallback_message(),
        "missing_info": missing_info,
        "used_llm": False
    }

def _generate_llm_response(query, user_context, recommendations, rag_info):
    """Genera una respuesta usando LLM"""
    if LLM_PROVIDER == "openai" and openai_client:
        # Crear prompt para OpenAI
        messages = [
            {
                "role": "system", 
                "content": """Eres un asistente turístico especializado en Curaçao.
                Proporciona respuestas amigables, conversacionales y útiles.
                Usa la información de las recomendaciones e información RAG para dar 
                respuestas precisas y relevantes."""
            }
        ]
        
        # Añadir contexto de usuario
        user_prefs = user_context.get("preferences", {})
        preferences_text = f"""
        Información del usuario:
        - Intereses: {', '.join(user_prefs.get('interests', []))}
        - Duración de estancia: {user_prefs.get('duration', 'No especificada')} días
        - Presupuesto: ${user_prefs.get('budget', 'No especificado')}
        """
        
        messages.append({"role": "system", "content": preferences_text})
        
        # Añadir recomendaciones
        rec_text = "Recomendaciones personalizadas:\n"
        for i, rec in enumerate(recommendations, 1):
            rec_text += f"{i}. {rec.get('name', '')}: {rec.get('description', '')[:100]}...\n"
            if "rag_info" in rec:
                rec_text += f"   Información adicional: {rec.get('rag_info', '')[:200]}...\n"
        
        messages.append({"role": "system", "content": rec_text})
        
        # Añadir consulta del usuario
        messages.append({"role": "user", "content": query})
        
        # Generar respuesta
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error con OpenAI: {e}")
            return templates.format_recommendations(recommendations, rag_info)
    
    elif LLM_PROVIDER == "claude" and claude_client:
        try:
            # Construir mensaje para Claude
            prompt = f"""Eres un asistente turístico especializado en Curaçao.
            Proporciona respuestas amigables, conversacionales y útiles.
            
            Información del usuario:
            - Intereses: {', '.join(user_context.get('preferences', {}).get('interests', []))}
            - Duración de estancia: {user_context.get('preferences', {}).get('duration', 'No especificada')} días
            - Presupuesto: ${user_context.get('preferences', {}).get('budget', 'No especificado')}
            
            Recomendaciones personalizadas:
            """
            
            for i, rec in enumerate(recommendations, 1):
                prompt += f"{i}. {rec.get('name', '')}: {rec.get('description', '')[:100]}...\n"
                if "rag_info" in rec:
                    prompt += f"   Información adicional: {rec.get('rag_info', '')[:200]}...\n"
            
            prompt += f"\nConsulta del usuario: {query}\n\nPor favor, responde amigablemente y usa la información proporcionada."
            
            response = claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error con Claude: {e}")
            return templates.format_recommendations(recommendations, rag_info)
    
    # Fallback a plantillas
    return templates.format_recommendations(recommendations, rag_info)

@mcp.tool()
def generate_follow_up_questions(interests: list) -> list:
    """
    Genera preguntas de seguimiento para mantener la conversación.
    
    Args:
        interests: Intereses del usuario
        
    Returns:
        Lista de posibles preguntas de seguimiento
    """
    return templates.get_follow_up_questions(interests)

if __name__ == "__main__":
    logger.info(f"Servidor MCP de respuestas iniciado: {RESPONSE_SERVER_ID}")
    mcp.serve()