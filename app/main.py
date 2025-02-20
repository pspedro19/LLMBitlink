from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from core.recommender.recommendation_engine import RecommendationEngine
from core.analyzer.nlp.routes import router as nlp_router
from core.recommender.full_service import get_full_recommendations, NLPRequest
from core.analyzer.nlp.intent_analyzer import IntentAnalyzer, IntentType, IntentResult
from core.recommender.enhanced_recommender import EnhancedRecommender
from fastapi.responses import HTMLResponse
from core.recommender.formatter import HTMLFormatter
import os
from utils.logger import get_logger
from utils.openai_helper import OpenAIHelper

from core.rag.services import RAGService, DocumentResponse, QueryResponse

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"API Key configured: {'Yes' if OPENAI_API_KEY else 'No'}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tourism Recommendations API")
engine = RecommendationEngine()

# Inicializar OpenAI helper
openai_helper = OpenAIHelper(OPENAI_API_KEY)

# Modelos Pydantic para RAG
class DocumentResponse(BaseModel):
    document_name: str = Field(..., description="Name of the document")
    content: str = Field(..., description="Content of the document")
    score: float = Field(..., description="Relevance score")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Original query")
    documents: List[DocumentResponse] = Field(..., description="Retrieved documents")
    response: str = Field(..., description="Generated response")

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Query in natural language")
    top_k: Optional[int] = Field(3, description="Number of documents to retrieve")

class Preferences(BaseModel):
    interests: List[str] = Field(..., description="List of travel interests")
    locations: List[str] = Field(..., description="Preferred locations")
    budget_per_day: Optional[float] = Field(None, description="Daily budget in USD")
    trip_duration: int = Field(..., description="Trip duration in days")
    group_size: int = Field(..., description="Number of travelers")
    activity_types: List[str] = Field(..., description="Preferred activity types")
    specific_sites: Optional[List[str]] = Field(None, description="Specific sites to visit")
    cuisine_preferences: Optional[List[str]] = Field(None, description="Food preferences")

class RecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    preferences: Preferences

class ChatRequest(BaseModel):
    text: str = Field(..., description="Natural language input text")
    
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nlp_router)

# Sistema de caché simple para respuestas
class ChatCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                self.hits += 1
                return entry['response']
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, response: str):
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.items(), key=lambda x: x[1]['timestamp'])
            del self.cache[oldest[0]]
        
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now()
        }


response_cache = ChatCache()
   
 
class Config:
    schema_extra = {
        "example": {
            "interests": ["cultural", "history", "architecture"],
            "locations": ["Punda", "Otrobanda"],
            "budget_per_day": 200.0,
            "trip_duration": 4,
            "group_size": 2,
            "activity_types": ["walking_tour", "museum_visits"],
            "specific_sites": ["Queen Emma Bridge"],
            "cuisine_preferences": ["local"]
        }
    }


# Inicializar servicios
try:
    engine = RecommendationEngine()
    openai_helper = OpenAIHelper(os.getenv("OPENAI_API_KEY"))
    
    # Inicializar RAG Service con configuración específica
    rag_service = RAGService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        local_model_path="/app/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        embedding_cache="/app/data/embeddings.pkl",
        n_ctx=4096,
        n_gpu_layers=20
    )
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Error initializing services: {str(e)}")
    raise


# Endpoints RAG
@app.post("/rag/documents/upload")
async def upload_document(force_reload: bool = False):
    """
    Carga todos los documentos de los directorios de conocimiento y crea el índice FAISS.
    """
    try:
        base_path = "/app/data/knowledge_base/curaçao_information"
        
        # Verificar que el directorio existe
        if not os.path.exists(base_path):
            raise HTTPException(
                status_code=400,
                detail=f"Base directory not found: {base_path}"
            )
        
        # Si force_reload es False y ya existe caché, retornar info
        if not force_reload and os.path.exists(rag_service.embedding_cache):
            return {
                "message": "Documents already loaded in cache",
                "cache_path": str(rag_service.embedding_cache),
                "document_count": len(rag_service.documents)
            }
        
        # Cargar documentos
        result = rag_service.bulk_load_documents(base_path)
        
        return {
            "message": "Documents processed successfully",
            "details": result
        }
        
    except Exception as e:
        logger.error(f"Error in bulk document loading: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing documents: {str(e)}"
        )


@app.post("/rag/query", response_model=QueryResponse)
async def query_documents(request: RAGQueryRequest):
    """Query the RAG system with natural language"""
    try:
        return rag_service.query(request.query, request.top_k)
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommendations/")
def get_recommendations(request: RecommendationRequest) -> JSONResponse:
    """Get personalized tourism recommendations"""
    try:
        # Get recommendations
        response = engine.get_recommendations(
            request.query, 
            request.preferences.dict()
        )
        
        if response["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=response["error"]
            )
            
        if response["status"] == "no_results":
            return JSONResponse(
                status_code=404,
                content={
                    "status": "no_results",
                    "message": response["message"],
                    "metadata": response["metadata"]
                }
            )
        
        # Format recommendations
        formatted_response = {
            "status": "success",
            "recommendations": [
                {
                    "name": rec["name"].upper(),
                    "type": rec["type"],
                    "location": rec["location"],
                    "cost": float(rec.get("cost", 0)),
                    "rating": {
                        "value": float(rec.get("rating", 0)),
                        "display": "★" * int(float(rec.get("rating", 0))) + 
                                 "☆" * (5 - int(float(rec.get("rating", 0))))
                    },
                    "description": rec.get("description", ""),
                    "relevance_score": rec["_scores"]["total"]
                }
                for rec in response["recommendations"]
            ],
            "metadata": {
                **response["metadata"],
                "currency": "USD",
                "timestamp": datetime.now().isoformat()
            },
            "validation": response["validation"]
        }
        
        return JSONResponse(content=formatted_response)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/recommendations/html")
def get_html_recommendations(request: RecommendationRequest) -> HTMLResponse:
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
            
        response = engine.get_recommendations(
            request.query,
            request.preferences.dict()
        )
        
        if response["status"] != "success":
            raise HTTPException(
                status_code=404 if response["status"] == "no_results" else 500,
                detail=response.get("message", "Error generating recommendations")
            )
            
        formatter = HTMLFormatter()
        html_content = formatter.format_to_html(
            response["recommendations"],
            request.preferences.dict()
        )
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Error processing HTML recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommendations/html-pro")
async def get_enhanced_html_recommendations(request: RecommendationRequest) -> HTMLResponse:
    """
    Enhanced endpoint that combines NER and OpenAI to provide rich, contextual recommendations
    """
    try:
        if not OPENAI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured"
            )
            
        # Get base recommendations using existing logic
        response = engine.get_recommendations(
            request.query,
            request.preferences.dict()
        )
        
        if response["status"] != "success":
            raise HTTPException(
                status_code=404 if response["status"] == "no_results" else 500,
                detail=response.get("message", "Error generating recommendations")
            )
            
        # Format base recommendations to HTML
        formatter = HTMLFormatter()
        base_html_content = formatter.format_to_html(
            response["recommendations"],
            request.preferences.dict()
        )
        
        # Initialize enhanced recommender
        enhanced_recommender = EnhancedRecommender(openai_helper)
        
        # Process and enhance the recommendations
        enhanced_html = await enhanced_recommender.process_recommendations(
            html_content=base_html_content,
            user_query=request.query,
            preferences=request.preferences.dict()
        )
        
        return HTMLResponse(content=enhanced_html)
        
    except Exception as e:
        logger.error(f"Error processing enhanced HTML recommendation request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommendations/full")
async def full_recommendations_endpoint(request: NLPRequest) -> HTMLResponse:
    """
    Endpoint que combina el análisis de texto en lenguaje natural y la generación de HTML
    """
    return await get_full_recommendations(request)

@app.post("/recommendations/chat")
async def chat_recommendations(request: ChatRequest) -> HTMLResponse:
    """
    Endpoint mejorado que maneja recomendaciones conversacionales con análisis avanzado de intenciones.
    """
    try:
        logger.info(f"Procesando solicitud de chat: {request.text}")
        
        # Verificar caché
        cache_key = f"chat_{request.text.lower().strip()}"
        cached_response = response_cache.get(cache_key)
        if cached_response:
            return HTMLResponse(content=cached_response)
        
        # Analizar intención con el nuevo analizador
        intent_analyzer = IntentAnalyzer()
        intent_result = intent_analyzer.analyze_intent(request.text)
        
        # Corregir esta línea
        logger.info(
            f"Intención detectada: {intent_result.primary_intent.value}, "
            f"confianza: {intent_result.confidence}"
        )
        
        # Manejar según el tipo de intención
        if intent_result.primary_intent == IntentType.IDENTITY:
            response = await _handle_identity_question(request.text)
            
        elif intent_result.primary_intent == IntentType.GREETING:
            response = await _handle_greeting(request.text)
            
        elif intent_result.primary_intent == IntentType.MIXED:
            response = await _handle_mixed_intent(request.text)
            
        elif intent_result.primary_intent == IntentType.RECOMMENDATION:
            response = await _handle_recommendation(request.text)
            
        else:
            response = await _handle_general_query(request.text)
        
        # Guardar en caché si es apropiado
        if _should_cache_response({
            'primary_intent': intent_result.primary_intent,
            'confidence': intent_result.confidence
        }):
            response_cache.set(cache_key, response.body.decode('utf-8'))
        
        return response
        
    except Exception as e:
        logger.error(f"Error en chat_recommendations: {str(e)}", exc_info=True)
        return HTMLResponse(content=_format_error_response("general"))
    
async def _handle_identity_question(text: str) -> HTMLResponse:
    """Maneja preguntas sobre la identidad del asistente."""
    try:
        system_message = (
            "Eres un guía turístico experto y amigable de Curazao. Al presentarte, "
            "menciona que eres un asistente virtual especializado en ayudar a los "
            "visitantes a descubrir las maravillas de Curazao, incluyendo su cultura, "
            "gastronomía, actividades y lugares de interés. Mantén un tono profesional "
            "pero cálido."
        )
        
        response = openai_helper.generate_tour_guide_response(text, system_message)
        return HTMLResponse(content=_format_chat_response(response))
    except Exception as e:
        logger.error(f"Error en manejo de pregunta de identidad: {str(e)}")
        return HTMLResponse(content=_format_error_response("openai"))

async def _handle_greeting(text: str) -> HTMLResponse:
    """Maneja saludos iniciales."""
    try:
        system_message = (
            "Eres un guía turístico amigable de Curazao. "
            "Responde al saludo y pregunta sobre sus intereses. "
            "Mantén un tono conversacional y ofrece tu ayuda para descubrir la isla."
        )
        
        response = openai_helper.generate_tour_guide_response(text, system_message)
        return HTMLResponse(content=_format_chat_response(response))
    except Exception as e:
        logger.error(f"Error en manejo de saludo: {str(e)}")
        return HTMLResponse(content=_format_error_response("openai"))

async def _handle_mixed_intent(text: str) -> HTMLResponse:
    """Maneja casos que combinan múltiples intenciones."""
    try:
        # Intentar obtener recomendaciones
        try:
            recommendations = await get_full_recommendations(NLPRequest(text=text))
            recommendations_str = recommendations.body.decode('utf-8')
            has_recommendations = "No se encontraron recomendaciones" not in recommendations_str
        except Exception:
            has_recommendations = False
            recommendations_str = ""
        
        # Generar respuesta conversacional
        system_message = (
            "Eres un guía turístico experto en Curazao. "
            f"{'Complementa las recomendaciones con ' if has_recommendations else 'Proporciona '}"
            "información relevante y mantén un tono conversacional."
        )
        
        chat_response = openai_helper.generate_tour_guide_response(text, system_message)
        
        if has_recommendations:
            return HTMLResponse(content=_format_mixed_response(chat_response, recommendations_str))
        else:
            return HTMLResponse(content=_format_chat_response(chat_response))
            
    except Exception as e:
        logger.error(f"Error en manejo de intención mixta: {str(e)}")
        return HTMLResponse(content=_format_error_response("mixed"))

async def _handle_recommendation(text: str) -> HTMLResponse:
    """Maneja solicitudes específicas de recomendaciones."""
    try:
        recommendations = await get_full_recommendations(NLPRequest(text=text))
        recommendations_str = recommendations.body.decode('utf-8')
        
        if "No se encontraron recomendaciones" in recommendations_str:
            return await _generate_alternative_recommendations(text)
            
        return HTMLResponse(content=recommendations_str)
        
    except Exception as e:
        logger.error(f"Error procesando recomendación: {str(e)}")
        return HTMLResponse(content=_format_error_response("recommendation"))

async def _handle_general_query(text: str) -> HTMLResponse:
    """Maneja consultas generales o no clasificadas."""
    try:
        system_message = (
            "Eres un guía turístico experto en Curazao. "
            "Proporciona información útil y relevante sobre la isla, "
            "y pregunta más detalles si es necesario para entender "
            "mejor los intereses del usuario."
        )
        
        response = openai_helper.generate_tour_guide_response(text, system_message)
        return HTMLResponse(content=_format_chat_response(response))
    except Exception as e:
        logger.error(f"Error en consulta general: {str(e)}")
        return HTMLResponse(content=_format_error_response("general"))

async def _generate_alternative_recommendations(text: str) -> HTMLResponse:
    """Genera recomendaciones alternativas cuando no hay coincidencias exactas."""
    try:
        response = openai_helper.generate_tour_guide_response(
            text,
            "Eres un guía turístico experto en Curazao. No encontramos recomendaciones " +
            "exactas para esta solicitud. Sugiere alternativas relevantes y explica " +
            "por qué podrían ser interesantes para el usuario."
        )
        return HTMLResponse(content=_format_chat_response(response))
    except Exception as e:
        logger.error(f"Error generando alternativas: {str(e)}")
        return HTMLResponse(content=_format_error_response("openai"))

def _format_chat_response(response: str) -> str:
    """
    Formatea la respuesta del chat en HTML para mantener consistencia visual.
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
    Formatea mensajes de error en HTML.
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

def _should_cache_response(intent_info: Dict[str, Any]) -> bool:
    """
    Determina si una respuesta debe ser cacheada basado en la información de la intención.
    """
    # No cachear respuestas con baja confianza
    if intent_info['confidence'] < 0.5:
        return False
    
    # No cachear ciertos tipos de intenciones
    if intent_info['primary_intent'] in {IntentType.GREETING, IntentType.IDENTITY}:
        return False
    
    # Cachear recomendaciones y respuestas a preguntas específicas
    if intent_info['primary_intent'] in {IntentType.RECOMMENDATION, IntentType.SPECIFIC_QUESTION}:
        return True
    
    # Para casos mixtos, cachear solo si la confianza es alta
    if intent_info['primary_intent'] == IntentType.MIXED:
        return intent_info['confidence'] > 0.7
    
    return False

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """API health check endpoint"""
    return {
        "status": "healthy",
        "rag_service": "active" if rag_service else "not initialized",
        "recommendation_engine": "active" if engine else "not initialized",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)