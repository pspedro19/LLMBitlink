from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from core.analyzer.nlp_processor import ImprovedNLPProcessor
from core.recommender.recommendation_engine import RecommendationEngine
from core.recommender.formatter import HTMLFormatter
import logging
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

# Configuraciones centralizadas
DEFAULT_PREFERENCES = {
    "trip_duration": 3,
    "group_size": 2,
    "interests": ["cultural", "sightseeing"],
    "locations": ["willemstad"],
    "activity_types": []
}

class NLPRequest(BaseModel):
    text: str = Field(..., description="Texto en lenguaje natural con las preferencias de viaje")
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("El texto no puede estar vacío")
        return v.strip()

class ProcessingError(Exception):
    """Error personalizado para fallos en el procesamiento"""
    pass

class RecommendationError(Exception):
    """Error personalizado para fallos en las recomendaciones"""
    pass

class FullRecommendationService:
    def __init__(self):
        self.nlp_processor = ImprovedNLPProcessor()
        self.recommendation_engine = RecommendationEngine()
        self.html_formatter = HTMLFormatter()

    async def process_full_recommendation(self, text: str) -> str:
        """
        Procesa el texto, extrae preferencias, genera recomendaciones y formatea en HTML
        
        Args:
            text (str): Texto en lenguaje natural
            
        Returns:
            str: Contenido HTML con las recomendaciones
            
        Raises:
            HTTPException: Si hay errores en el procesamiento
        """
        try:
            # 1. Extraer preferencias usando el NLP processor
            logger.info(f"Iniciando extracción de preferencias para texto: {text[:100]}...")
            try:
                preferences = self.nlp_processor.extract_preferences(text)
                logger.info("Preferencias extraídas exitosamente")
            except Exception as e:
                logger.error(f"Error en extracción de preferencias: {str(e)}")
                raise ProcessingError(f"Error procesando preferencias: {str(e)}")
            
            # 2. Validar y completar preferencias requeridas
            preferences = self._validate_and_complete_preferences(preferences)
            logger.info(f"Preferencias validadas y completadas: {preferences}")
            
            # 3. Obtener recomendaciones usando el motor existente
            try:
                recommendation_request = {
                    "query": text,
                    "preferences": preferences
                }
                
                recommendations = self.recommendation_engine.get_recommendations(
                    recommendation_request["query"],
                    recommendation_request["preferences"]
                )
                
                if recommendations["status"] != "success":
                    raise RecommendationError(
                        recommendations.get("message", "Error generando recomendaciones")
                    )
                    
                logger.info("Recomendaciones generadas exitosamente")
                
            except RecommendationError as re:
                logger.error(f"Error en motor de recomendaciones: {str(re)}")
                raise HTTPException(
                    status_code=404 if "no results" in str(re).lower() else 500,
                    detail=str(re)
                )
            except Exception as e:
                logger.error(f"Error inesperado en motor de recomendaciones: {str(e)}")
                raise HTTPException(status_code=500, detail="Error interno del servidor")

            # 4. Formatear resultados a HTML
            try:
                html_content = self.html_formatter.format_to_html(
                    recommendations["recommendations"],
                    preferences
                )
                logger.info("HTML generado exitosamente")
                return html_content
                
            except Exception as e:
                logger.error(f"Error formateando HTML: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Error generando la vista HTML"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error no manejado en el servicio: {str(e)}")
            raise HTTPException(status_code=500, detail="Error interno del servidor")

    def _validate_and_complete_preferences(self, preferences: dict) -> dict:
        """
        Valida y completa las preferencias extraídas usando valores por defecto
        
        Args:
            preferences (dict): Preferencias extraídas por NLP
            
        Returns:
            dict: Preferencias validadas y completadas
        """
        validated = preferences.copy()
        
        # Aplicar valores por defecto
        for key, default_value in DEFAULT_PREFERENCES.items():
            if key not in validated or not validated[key]:
                validated[key] = default_value
                logger.debug(f"Usando valor por defecto para {key}: {default_value}")
                
        return validated

# Inicializar el servicio
full_recommendation_service = FullRecommendationService()

# Función para el endpoint
async def get_full_recommendations(request: NLPRequest) -> HTMLResponse:
    """
    Endpoint que combina procesamiento NLP y generación de HTML
    
    Args:
        request (NLPRequest): Texto en lenguaje natural
        
    Returns:
        HTMLResponse: Página HTML con recomendaciones
    """
    try:
        html_content = await full_recommendation_service.process_full_recommendation(request.text)
        return HTMLResponse(content=html_content)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error en endpoint de recomendaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))