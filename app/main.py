import json
import os
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from dotenv import load_dotenv
import logging
import spacy
from app.real_estate_analyzer import RealEstateAnalyzer

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Inicializar FastAPI
app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración de medios
MEDIA_DIR = os.getenv('MEDIA_DIR', '/app/images')
if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR, exist_ok=True)
    # raise RuntimeError(f"El directorio de imágenes no existe en: {MEDIA_DIR}")

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Definir el modelo de Claude a usar
CLAUDE_MODEL = "claude-3-sonnet-20240229"

# Validar ANTHROPIC_API_KEY
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

# Inicializar Anthropic
anthropic_client = Anthropic(
    api_key=api_key
)

# Modelo Pydantic para mensajes
class ChatMessage(BaseModel):
    user_input: str
    response_format: Optional[str] = "html"
    context: Optional[Dict[str, Any]] = None

# Funciones disponibles para análisis inmobiliario
PROPERTY_ANALYSIS_SYSTEM_PROMPT = """
You are a real estate assistant specialized in analyzing property data. Your task is to:
1. Understand user queries about properties
2. Extract relevant search criteria (location, price, features)
3. Generate natural responses about properties
4. Present property information in a clear, structured format

When presenting properties:
- Focus on key details: location, price, size, features
- Highlight unique selling points
- Include relevant market context
- Make recommendations based on user preferences

Respond in Spanish unless specifically requested otherwise.
"""

def initialize_analyzer():
    try:
        nlp_models = [
            spacy.load("es_core_news_md"),
            spacy.load("en_core_web_md")
        ]
        
        db_path = os.getenv("DB_PATH", "chat-Interface/db.sqlite3")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
            
        analyzer = RealEstateAnalyzer(
            db_path=db_path,
            nlp_models=nlp_models,
            log_path="real_estate_logs"
        )
        
        logger.info("Analyzer initialized successfully")
        return analyzer
        
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {str(e)}", exc_info=True)
        return None

analyzer = initialize_analyzer()

def build_html_response(data: Dict[str, Any], claude_response: str) -> str:
    """Construye la respuesta HTML combinando el análisis de Claude y los resultados"""
    if "error" in data:
        return f"""
        <div class="message message-bot error-message">
            <div class="message-content">
                <p>Lo siento, hubo un problema: {data['error']}</p>
            </div>
        </div>
        """

    # Añadir respuesta de Claude
    response_html = f"""
    <div class="message message-bot">
        <div class="message-content">
            <p>{claude_response}</p>
        </div>
    </div>
    """

    # Construir grid de propiedades
    properties_html = ""
    for prop in data.get('properties', []):
        properties_html += f"""
        <div class="property-card">
            <img src="{prop.get('image', '/media/default.jpg')}" 
                 alt="{prop.get('property_type', 'Propiedad')}" 
                 class="property-image">
            <div class="property-content">
                <div class="property-price">
                    {prop.get('price', 'Precio no disponible')} USD
                </div>
                <div class="property-location">
                    <i class="fas fa-map-marker-alt"></i> 
                    {prop.get('location', '')}, {prop.get('city_name', '')}, 
                    {prop.get('province_name', '')}, {prop.get('country_name', '')}
                </div>
                <div class="property-features">
                    <span><i class="fas fa-bed"></i> {prop.get('avg_rooms', 'N/A')} hab</span>
                    <span><i class="fas fa-bath"></i> {prop.get('avg_bedrooms', 'N/A')} baños</span>
                    <span><i class="fas fa-ruler-combined"></i> {prop.get('square_meters', 'N/A')} m²</span>
                </div>
                <p class="property-description">{prop.get('description', 'Sin descripción disponible')}</p>
                <a href="{prop.get('url', '#')}" class="property-cta" 
                   target="_blank" rel="noopener noreferrer">Ver Detalles</a>
            </div>
        </div>
        """

    return f"""
    {response_html}
    <div class="property-grid">
        {properties_html if properties_html else '''
        <div class="message message-bot">
            <div class="message-content">
                <p>No se encontraron propiedades que coincidan con los criterios especificados.</p>
            </div>
        </div>
        '''}
    </div>
    """

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    """Endpoint principal para procesar consultas de usuarios usando Claude"""
    try:
        if not analyzer:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")

        # Procesar la consulta con el analizador
        analysis_results = analyzer.process_user_query(chat_message.user_input)
        
        if analysis_results["status"] == "error":
            return HTMLResponse(content=build_html_response(
                {"error": analysis_results["error"]}, 
                "Lo siento, hubo un error al procesar tu consulta."
            ))

        # Construir el prompt para Claude
        properties_context = json.dumps(analysis_results["results"], ensure_ascii=False)
        prompt = f"""
        Consulta del usuario: {chat_message.user_input}

        Propiedades encontradas:
        {properties_context}

        Por favor, analiza estas propiedades y genera una respuesta natural que:
        1. Resuma los principales hallazgos
        2. Destaque las características más relevantes
        3. Haga recomendaciones basadas en la consulta del usuario
        4. Mencione rangos de precios y ubicaciones disponibles
        """

        # Obtener respuesta de Claude
        # response = anthropic_client.messages.create(
        #     model="claude-3-sonnet-20240229",
        #     max_tokens=2000,
        #     temperature=0.7,
        #     system=PROPERTY_ANALYSIS_SYSTEM_PROMPT,
        #     messages=[{
        #         "role": "user",
        #         "content": prompt
        #     }]
        # )
        message = await anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            temperature=0.7,
            system="Eres un experto asistente inmobiliario que ayuda a usuarios a encontrar propiedades. Tus respuestas son claras, concisas y naturales.",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Construir y retornar respuesta HTML
        html_response = build_html_response(
            analysis_results,
            message.content[0].text
        )

        if chat_message.response_format == "html":
            return HTMLResponse(content=html_response)
        else:
            return JSONResponse(content={
                "response": html_response,
                "analysis": analysis_results,
                "status": "success"
            })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        error_response = {
            "status": "error",
            "detail": str(e),
            "response": """
                <div class="message message-bot error-message">
                    <div class="message-content">
                        <p>Lo siento, ha ocurrido un error al procesar tu solicitud.</p>
                    </div>
                </div>
            """
        }
        return JSONResponse(
            status_code=500,
            content=error_response
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800)