# mcp_curacao/servers/orchestrator_server.py
from mcp.server.fastmcp import FastMCP
import uuid
import time
import logging
import sys
import os
import re

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("orchestrator-server")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ORCHESTRATOR_SERVER_ID

# Inicializar servidor MCP
mcp = FastMCP(ORCHESTRATOR_SERVER_ID)

# Almacenamiento de contexto de usuario
user_contexts = {}

@mcp.tool()
def create_user_session() -> dict:
    """
    Crea una nueva sesión de usuario.
    
    Returns:
        Información de la sesión creada
    """
    session_id = str(uuid.uuid4())
    user_contexts[session_id] = {
        "created_at": time.time(),
        "last_updated": time.time(),
        "preferences": {},
        "recommendations": [],
        "message_history": [],
        "conversation_state": "new_conversation"
    }
    
    logger.info(f"Nueva sesión creada: {session_id}")
    
    return {
        "session_id": session_id,
        "created_at": user_contexts[session_id]["created_at"]
    }

@mcp.tool()
def update_user_preferences(session_id: str, preferences: dict) -> dict:
    """
    Actualiza las preferencias del usuario.
    
    Args:
        session_id: ID de la sesión del usuario
        preferences: Preferencias a actualizar
        
    Returns:
        Preferencias actualizadas
    """
    if session_id not in user_contexts:
        return {
            "status": "error",
            "message": "Sesión no encontrada"
        }
    
    # Actualizar solo las preferencias proporcionadas
    for key, value in preferences.items():
        if value is not None:  # Solo actualizar si hay valor
            user_contexts[session_id]["preferences"][key] = value
    
    user_contexts[session_id]["last_updated"] = time.time()
    
    # Actualizar el estado de la conversación
    if "conversation_state" in user_contexts[session_id]:
        _update_conversation_state(session_id)
    
    logger.info(f"Preferencias actualizadas para sesión {session_id}: {preferences}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "preferences": user_contexts[session_id]["preferences"]
    }

@mcp.tool()
def extract_preferences_from_message(message: str, previous_preferences: dict = None) -> dict:
    """
    Extrae preferencias del usuario a partir de un mensaje.
    
    Args:
        message: Mensaje del usuario
        previous_preferences: Preferencias previas para contexto
        
    Returns:
        Preferencias detectadas
    """
    preferences = {}
    
    # Detectar intereses
    interest_keywords = {
        "cultural": ["cultura", "museo", "historia", "arte", "patrimonio"],
        "natural": ["playa", "naturaleza", "parque", "buceo", "snorkel"],
        "family": ["familia", "niños", "diversión", "actividades para niños"],
        "gastronomy": ["comida", "restaurante", "gastronomía", "comer", "cocina"],
        "nightlife": ["fiesta", "discoteca", "club", "bar", "noche", "música"]
    }
    
    interests = []
    for interest, keywords in interest_keywords.items():
        for keyword in keywords:
            if keyword.lower() in message.lower():
                interests.append(interest)
                break
    
    if interests:
        preferences["interests"] = list(set(interests))  # Eliminar duplicados
    
    # Enhanced duration detection
    duration_patterns = [
        r'(\d+)\s*(día|días|dias|day|days)',  # With explicit word
        r'(?:durante|for|por)\s*(\d+)',  # With preposition
        r'(?:estancia|estadía|stay)[^\d]*(\d+)'  # Stay followed by number
    ]
    
    for pattern in duration_patterns:
        duration_match = re.search(pattern, message.lower())
        if duration_match:
            preferences["duration"] = float(duration_match.group(1))
            break
    
    # Enhanced budget detection
    budget_patterns = [
        r'(\d+)\s*(dólares|dolares|dollars|\$|euros|euro)',  # With currency
        r'(?:presupuesto|budget)[^\d]*(\d+)',  # Budget followed by number
        r'(?:gastar|spend)[^\d]*(\d+)'  # Spend followed by number
    ]
    
    for pattern in budget_patterns:
        budget_match = re.search(pattern, message.lower())
        if budget_match:
            preferences["budget"] = float(budget_match.group(1))
            break
    
    # Single number detection based on context
    if previous_preferences:
        # If we already have interests and missing duration, and receive just a number
        if ("interests" in previous_preferences and 
            "duration" not in previous_preferences and
            re.match(r'^\d+$', message.strip())):
            preferences["duration"] = float(message.strip())
            
        # If we have interests + duration but no budget, and receive just a number
        elif ("interests" in previous_preferences and 
              "duration" in previous_preferences and 
              "budget" not in previous_preferences and
              re.match(r'^\d+$', message.strip())):
            preferences["budget"] = float(message.strip())
    
    # Detectar ubicaciones
    location_keywords = ["willemstad", "punda", "otrobanda", "curaçao", "curacao"]
    locations = []
    for location in location_keywords:
        if location.lower() in message.lower():
            locations.append(location)
    
    if locations:
        preferences["locations"] = locations
    
    logger.info(f"Preferencias extraídas del mensaje: {preferences}")
    return preferences

@mcp.tool()
def determine_missing_information(session_id: str) -> list:
    """
    Determina qué información falta para generar recomendaciones.
    
    Args:
        session_id: ID de la sesión del usuario
        
    Returns:
        Lista de campos que faltan
    """
    if session_id not in user_contexts:
        return ["session_not_found"]
    
    preferences = user_contexts[session_id]["preferences"]
    
    # Campos requeridos para recomendaciones completas
    required_fields = ["interests", "duration", "budget"]
    
    missing = []
    for field in required_fields:
        if field not in preferences or not preferences[field]:
            missing.append(field)
    
    logger.info(f"Campos faltantes para sesión {session_id}: {missing}")
    return missing

@mcp.tool()
def store_recommendations(session_id: str, recommendations: list) -> dict:
    """
    Almacena las recomendaciones generadas para el usuario.
    
    Args:
        session_id: ID de la sesión del usuario
        recommendations: Lista de recomendaciones
        
    Returns:
        Estado de la operación
    """
    if session_id not in user_contexts:
        return {
            "status": "error",
            "message": "Sesión no encontrada"
        }
    
    user_contexts[session_id]["recommendations"] = recommendations
    user_contexts[session_id]["last_updated"] = time.time()
    
    logger.info(f"Almacenadas {len(recommendations)} recomendaciones para sesión {session_id}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "recommendation_count": len(recommendations)
    }

@mcp.tool()
def enrich_recommendations(recommendations: list, rag_results: dict) -> list:
    """
    Enriquece las recomendaciones con información del RAG.
    
    Args:
        recommendations: Lista de recomendaciones
        rag_results: Resultados del sistema RAG
        
    Returns:
        Recomendaciones enriquecidas
    """
    if not recommendations or not rag_results:
        return recommendations
    
    # Obtener chunks del RAG
    chunks = rag_results.get("chunks", [])
    if not chunks:
        return recommendations
    
    # Diccionario para almacenar información relevante por lugar
    place_info = {}
    
    # Extraer información relevante de los chunks
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        # Buscar menciones de lugares en el texto RAG
        for rec in recommendations:
            place_name = rec.get("name", "").lower()
            if place_name and place_name in chunk_text.lower():
                if place_name not in place_info:
                    place_info[place_name] = []
                place_info[place_name].append(chunk_text)
    
    # Enriquecer recomendaciones
    enriched_recommendations = []
    for rec in recommendations:
        enriched_rec = rec.copy()
        
        # Añadir información RAG si existe
        place_name = rec.get("name", "").lower()
        if place_name in place_info:
            # Tomar solo el primer fragmento más relevante
            enriched_rec["rag_info"] = place_info[place_name][0]
        
        enriched_recommendations.append(enriched_rec)
    
    logger.info(f"Enriquecidas {len(enriched_recommendations)} recomendaciones con información RAG")
    return enriched_recommendations

@mcp.tool()
def add_message_to_history(session_id: str, sender: str, message: str) -> dict:
    """
    Añade un mensaje al historial de la conversación.
    
    Args:
        session_id: ID de la sesión del usuario
        sender: Quien envía el mensaje ('user' o 'assistant')
        message: Contenido del mensaje
        
    Returns:
        Estado de la operación
    """
    if session_id not in user_contexts:
        return {
            "status": "error",
            "message": "Sesión no encontrada"
        }
    
    # Añadir mensaje al historial
    user_contexts[session_id]["message_history"].append({
        "sender": sender,
        "message": message,
        "timestamp": time.time()
    })
    
    # Actualizar timestamp
    user_contexts[session_id]["last_updated"] = time.time()
    
    return {
        "status": "success",
        "session_id": session_id,
        "history_length": len(user_contexts[session_id]["message_history"])
    }

@mcp.tool()
def get_user_context(session_id: str) -> dict:
    """
    Obtiene el contexto completo del usuario.
    
    Args:
        session_id: ID de la sesión del usuario
        
    Returns:
        Contexto completo del usuario
    """
    if session_id not in user_contexts:
        return {
            "status": "error",
            "message": "Sesión no encontrada"
        }
    
    return user_contexts[session_id]

@mcp.tool()
def get_conversation_state(session_id: str) -> str:
    """
    Determina el estado actual de la conversación.
    
    Args:
        session_id: ID de la sesión del usuario
        
    Returns:
        Estado de la conversación
    """
    if session_id not in user_contexts:
        return "new_conversation"
    
    if "conversation_state" in user_contexts[session_id]:
        return user_contexts[session_id]["conversation_state"]
    
    # Determinar el estado basado en preferencias
    _update_conversation_state(session_id)
    return user_contexts[session_id]["conversation_state"]

@mcp.tool()
def handle_special_request(session_id: str, request_type: str) -> dict:
    """
    Maneja solicitudes especiales como itinerarios.
    
    Args:
        session_id: ID de la sesión del usuario
        request_type: Tipo de solicitud especial
        
    Returns:
        Resultado de la solicitud
    """
    if session_id not in user_contexts:
        return {"status": "error", "message": "Sesión no encontrada"}
    
    if request_type == "itinerary":
        preferences = user_contexts[session_id]["preferences"]
        recommendations = user_contexts[session_id].get("recommendations", [])
        
        if not recommendations or not preferences.get("duration"):
            return {
                "status": "insufficient_data",
                "message": "Necesito recomendaciones y duración para crear un itinerario"
            }
        
        # Agrupar por tipo
        grouped_recommendations = {
            "activities": [],
            "spots": [],
            "restaurants": [],
            "nightlife": []
        }
        
        for rec in recommendations:
            rec_type = rec.get("type", "")
            if rec_type in ["activity"]:
                grouped_recommendations["activities"].append(rec)
            elif rec_type in ["spot", "attraction"]:
                grouped_recommendations["spots"].append(rec)
            elif rec_type in ["restaurant", "cafe"]:
                grouped_recommendations["restaurants"].append(rec)
            elif rec_type in ["nightlife", "bar", "club"]:
                grouped_recommendations["nightlife"].append(rec)
        
        return {
            "status": "success",
            "request_type": "itinerary",
            "duration": preferences.get("duration", 0),
            "grouped_recommendations": grouped_recommendations
        }
    
    return {"status": "error", "message": "Tipo de solicitud desconocido"}

def _update_conversation_state(session_id):
    """
    Actualiza el estado de la conversación en función de las preferencias.
    
    Args:
        session_id: ID de la sesión del usuario
    """
    if session_id not in user_contexts:
        return
    
    preferences = user_contexts[session_id]["preferences"]
    
    # Comprobar qué información falta
    if "interests" not in preferences or not preferences["interests"]:
        user_contexts[session_id]["conversation_state"] = "asking_interests"
    elif "duration" not in preferences or not preferences["duration"]:
        user_contexts[session_id]["conversation_state"] = "asking_duration"
    elif "budget" not in preferences or not preferences["budget"]:
        user_contexts[session_id]["conversation_state"] = "asking_budget"
    else:
        user_contexts[session_id]["conversation_state"] = "providing_recommendations"

if __name__ == "__main__":
    logger.info(f"Servidor MCP Orquestador iniciado: {ORCHESTRATOR_SERVER_ID}")
    mcp.serve()