import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
from dotenv import load_dotenv
import os
import logging
import spacy
from fastapi.middleware.cors import CORSMiddleware
from real_estate_analyzer import RealEstateAnalyzer

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
    raise RuntimeError(f"El directorio de imágenes no existe en: {MEDIA_DIR}")

app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# Inicializar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Modelo Pydantic actualizado para mensajes
class ChatMessage(BaseModel):
    user_input: str
    response_format: Optional[str] = "html"
    context: Optional[Dict[str, Any]] = None

# Funciones disponibles actualizadas para NER
AVAILABLE_FUNCTIONS = {
    "get_property_search": {
        "name": "get_property_search",
        "description": "Search for properties based on various criteria including location, price, and features",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's search query"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "price_range": {
                            "type": "object",
                            "properties": {
                                "min": {"type": "number"},
                                "max": {"type": "number"}
                            }
                        },
                        "property_type": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["query"]
        }
    }
}

# Inicializar RealEstateAnalyzer con modelos de spaCy
# def initialize_analyzer():
#     try:
#         # Cargar modelos de spaCy
#         nlp_models = [
#             spacy.load("es_core_news_md"),
#             spacy.load("en_core_web_md")
#         ]
        
#         db_path = os.getenv("DB_PATH", "chat-Interface/db.sqlite3")
#         if not os.path.exists(db_path):
#             raise FileNotFoundError(f"Database not found at {db_path}")
            
#         return RealEstateAnalyzer(
#             db_path=db_path,
#             nlp_models=nlp_models
#         )
#     except Exception as e:
#         logger.error(f"Failed to initialize analyzer: {str(e)}")
#         return None

def initialize_analyzer():
    try:
        # Cargar modelos de spaCy
        nlp_models = [
            spacy.load("es_core_news_md"),
            spacy.load("en_core_web_md")
        ]
        
        db_path = os.getenv("DB_PATH", "chat-Interface/db.sqlite3")
        logger.info(f"Trying to access database at: {db_path}")
        logger.info(f"Database exists: {os.path.exists(db_path)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Directory contents: {os.listdir('/app/chat-Interface')}")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
            
        # Crear el analizador con los modelos de spaCy
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

def execute_function(analyzer: RealEstateAnalyzer, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta la función solicitada por el LLM usando el NER para procesar la consulta
    """
    if not analyzer:
        return {"error": "Database connection not available"}
        
    try:
        if function_name == "get_property_search":
            # Procesar la consulta usando NER
            analysis_results = analyzer.process_user_query(parameters['query'])
            
            if analysis_results["status"] == "error":
                return {"error": analysis_results["error"]}
            
            # Generar respuesta con GPT
            gpt_response = analyzer.generate_gpt_response(
                parameters['query'], 
                analysis_results
            )
            
            return {
                "properties": analysis_results["results"],
                "analysis": analysis_results,
                "gpt_response": gpt_response
            }
        else:
            raise ValueError(f"Unknown function: {function_name}")
            
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        return {"error": f"Error executing function: {str(e)}"}

def build_html_response(data: Dict[str, Any], gpt_response: str) -> str:
    """
    Construye la respuesta HTML combinando el análisis GPT y los resultados
    """
    if "error" in data:
        return f"""
        <div class="message message-bot error-message">
            <div class="message-content">
                <p>Lo siento, hubo un problema: {data['error']}</p>
            </div>
        </div>
        """

    # Añadir respuesta GPT
    response_html = f"""
    <div class="message message-bot">
        <div class="message-content">
            <p>{gpt_response}</p>
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

# def build_html_response(data: Dict[str, Any], question: str) -> str:
#     if "error" in data:
#         return f"""
#         <div class="message message-bot">
#             <div class="message-content">
#                 <p>Lo siento, hubo un problema al obtener los datos: {data['error']}</p>
#             </div>
#         </div>
#         """

#     properties_html = ""
#     for prop in data.get('properties', []):
#         details_url = prop.get('url', '#')
        
#         properties_html += f"""
#         <div class="property-card">
#             <img src="{prop.get('image', 'default.bmp')}" alt="{prop.get('property_type', 'Propiedad')}" class="property-image">
#             <div class="property-content">
#                 <div class="property-price">{prop.get('price', 'Precio no disponible')} USD</div>
#                 <div class="property-location">
#                     <i class="fas fa-map-marker-alt"></i> {prop.get('location', '')}, {prop.get('city_name', '')}, {prop.get('province_name', '')}, {prop.get('country_name', '')}
#                 </div>
#                 <div class="property-features">
#                     <span><i class="fas fa-bed"></i> {prop.get('promedio_dormitorios', 'N/A')} hab</span>
#                     <span><i class="fas fa-bath"></i> {prop.get('promedio_ambientes', 'N/A')} baños</span>
#                     <span><i class="fas fa-ruler-combined"></i> {prop.get('square_meters', 'N/A')} m²</span>
#                 </div>
#                 <p class="property-description">{prop.get('description', 'Sin descripción disponible')}</p>
#                 <a href="{details_url}" class="property-cta" target="_blank" rel="noopener noreferrer">Ver Detalles</a>
#             </div>
#         </div>
#         """

#     # Si no hay propiedades, devolver mensaje formateado
#     if not properties_html:
#         return """
#         <div class="message message-bot">
#             <div class="message-content">
#                 <p>Lo siento, no encontré propiedades que coincidan con tu búsqueda. ¿Te gustaría intentar con otros criterios?</p>
#             </div>
#         </div>
#         """

#     return f"""
#     <div class="message message-bot">
#         <div class="message-content">
#             <p>Aquí te muestro las propiedades disponibles según tu búsqueda:</p>
#         </div>
#     </div>
#     <div class="property-grid">
#         {properties_html}
#     </div>
#     """

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    """
    Endpoint principal para procesar consultas de usuarios usando NER
    """
    try:
        if not analyzer:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")

        # Consultar a GPT para determinar la función a llamar
        function_selection_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a real estate assistant that helps users find and analyze properties. 
                    Based on the user's message, extract the main search criteria and intent."""
                },
                {"role": "user", "content": chat_message.user_input}
            ],
            functions=list(AVAILABLE_FUNCTIONS.values()),
            function_call="auto"
        )

        # Extraer información de la función a llamar
        message = function_selection_response.choices[0].message
        
        if message.get("function_call"):
            # Ejecutar la función seleccionada
            function_name = message["function_call"]["name"]
            function_params = eval(message["function_call"]["arguments"])
            
            # Añadir el contexto si existe
            if chat_message.context:
                function_params["context"] = chat_message.context
            
            data = execute_function(analyzer, function_name, function_params)
            
            # Construir respuesta HTML
            response_content = build_html_response(
                data, 
                data.get("gpt_response", "")
            )

            # Retornar según formato solicitado
            if chat_message.response_format == "html":
                return HTMLResponse(content=response_content)
            else:
                return JSONResponse(content={
                    "response": response_content,
                    "analysis": data.get("analysis", {}),
                    "status": "success"
                })

        else:
            # Si no se llamó a ninguna función, proporcionar respuesta por defecto
            return JSONResponse(content={
                "response": """
                    <div class="message message-bot">
                        <div class="message-content">
                            <p>Lo siento, no pude entender completamente tu consulta. 
                            ¿Podrías proporcionar más detalles sobre lo que estás buscando?</p>
                        </div>
                    </div>
                """,
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
        
@app.get("/health")
def health_check():
    if not os.path.exists(os.getenv("DB_PATH")):
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Database not found"}
        )
    return {"status": "healthy"}