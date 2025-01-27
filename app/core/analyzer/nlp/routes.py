from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import os
from utils.logger import get_logger
from .models import NLPRequest, NLPResponse
from .processor import OpenAINLPProcessor

# Configurar logger
logger = get_logger(__name__)

# Crear router
router = APIRouter()

# Dependency para obtener el procesador NLP
def get_nlp_processor() -> OpenAINLPProcessor:
    """Dependency injection para el procesador NLP"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key no configurada"
        )
    return OpenAINLPProcessor(api_key)

@router.post("/recommendations/nlp", 
            response_model=NLPResponse,
            summary="Get NLP Recommendations",
            description="Procesa texto en lenguaje natural para extraer preferencias de viaje")
async def process_nlp_request(
    request: NLPRequest,
    processor: OpenAINLPProcessor = Depends(get_nlp_processor)
) -> Dict[str, Any]:
    """
    Procesa texto en lenguaje natural para extraer preferencias de viaje
    
    Args:
        request (NLPRequest): Texto a procesar
        
    Returns:
        NLPResponse: Preferencias estructuradas extraídas del texto
        
    Raises:
        HTTPException: Si hay un error en el procesamiento
    """
    try:
        # Validar entrada
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="El texto no puede estar vacío"
            )
            
        # Procesar texto
        logger.info(f"Procesando texto: {request.text[:100]}...")
        result = processor.process_text(request.text)
        
        # Validar resultado
        if not result or "preferences" not in result:
            raise HTTPException(
                status_code=500,
                detail="Error procesando el texto"
            )
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en el endpoint NLP: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor"
        )