# app/api/routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.recommender.engine import RecommendationEngine
from core.nlu.processor import NLUProcessor
from conversation.manager import ConversationManager
from rag.manager import RAGManager
from utils.logger import get_logger

logger = get_logger(__name__)

# Modelos de datos
class Query(BaseModel):
    text: str
    language: Optional[str] = "auto"
    session_id: Optional[str] = None

class Preferences(BaseModel):
    interests: Optional[List[str]] = None
    locations: Optional[List[str]] = None
    budget: Optional[float] = None
    duration: Optional[int] = None
    group_size: Optional[int] = None
    with_children: Optional[bool] = None
    special_occasion: Optional[str] = None

class SessionMessage(BaseModel):
    session_id: str
    message: str

class SessionConfirmation(BaseModel):
    session_id: str
    confirmed: bool

# Crear router
router = APIRouter(prefix="/api", tags=["recommendations"])

# Inicializar componentes
recommender = RecommendationEngine()
nlu_processor = NLUProcessor()
conversation_manager = ConversationManager()
rag_manager = RAGManager()

@router.post("/query")
async def process_query(query: Query):
    """Procesa una consulta directa y devuelve recomendaciones"""
    try:
        # Procesar consulta con NLU
        nlu_result = nlu_processor.process_query(query.text, query.language)
        
        # Mejorar preferencias con RAG
        enhanced_preferences = rag_manager.enhance_preferences(
            nlu_result["preferences"],
            query.text
        )
        
        # Generar recomendaciones
        recommendations = recommender.generate_recommendations(enhanced_preferences)
        
        # Añadir información adicional con RAG para cada recomendación
        if recommendations["status"] == "success":
            for rec in recommendations["recommendations"]:
                additional_info = rag_manager.get_additional_info(rec)
                if additional_info:
                    rec["additional_info"] = additional_info
        
        return {
            "query": nlu_result,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/create")
async def create_session():
    """Crea una nueva sesión de conversación"""
    try:
        session_id = conversation_manager.create_session()
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Error creando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/message")
async def process_session_message(message: SessionMessage):
    """Procesa un mensaje dentro de una sesión de conversación"""
    try:
        result = conversation_manager.process_message(
            message.session_id,
            message.message
        )
        
        # Si está listo para recomendaciones, generarlas
        if result.get("ready_for_recommendations") and result.get("needs_confirmation") is False:
            session_state = conversation_manager.get_session_state(message.session_id)
            preferences = session_state.get("preferences", {})
            
            # Generar recomendaciones
            recommendations = recommender.generate_recommendations(preferences)
            result["recommendations"] = recommendations
        
        return result
        
    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/confirm")
async def confirm_session_preferences(confirmation: SessionConfirmation):
    """Confirma las preferencias en una sesión"""
    try:
        result = conversation_manager.confirm_preferences(
            confirmation.session_id,
            confirmation.confirmed
        )
        
        # Si está confirmado, generar recomendaciones
        if result.get("ready_for_recommendations"):
            session_state = conversation_manager.get_session_state(confirmation.session_id)
            preferences = session_state.get("preferences", {})
            
            # Generar recomendaciones
            recommendations = recommender.generate_recommendations(preferences)
            result["recommendations"] = recommendations
        
        return result
        
    except Exception as e:
        logger.error(f"Error confirmando preferencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/direct")
async def get_direct_recommendations(preferences: Preferences):
    """Genera recomendaciones directamente a partir de preferencias explícitas"""
    try:
        # Convertir modelo Pydantic a diccionario
        prefs_dict = preferences.dict(exclude_none=True)
        
        # Generar recomendaciones
        recommendations = recommender.generate_recommendations(prefs_dict)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones directas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Obtiene información de una sesión"""
    try:
        state = conversation_manager.get_session_state(session_id)
        return {"session_id": session_id, "state": state}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo información de sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))