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
from fastapi.middleware.cors import CORSMiddleware
from .real_estate_analyzer import RealEstateAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configurar la ruta absoluta a la carpeta de imágenes
MEDIA_DIR = os.getenv('MEDIA_DIR', '/app/images')

# Verificar que el directorio existe
print(f"Buscando directorio de imágenes en: {MEDIA_DIR}")
if not os.path.exists(MEDIA_DIR):
    raise RuntimeError(f"El directorio de imágenes no existe en: {MEDIA_DIR}")
else:
    print(f"Directorio de imágenes encontrado en: {MEDIA_DIR}")
    
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar el directorio de imágenes
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
MEDIA_URL = "http://161.35.120.130:8000/media/"
# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize RealEstateAnalyzer with proper path handling
def initialize_analyzer():
    possible_db_paths = [
        os.getenv("DB_PATH", ""),
        "chat-Interface/db.sqlite3"
    ]
    
    for db_path in possible_db_paths:
        try:
            if db_path and os.path.exists(db_path):
                return RealEstateAnalyzer(db_path=db_path)
        except Exception as e:
            logger.warning(f"Failed to initialize with path {db_path}: {str(e)}")
            continue
    
    # If we couldn't connect to the database, log the error but don't crash
    logger.error("Could not connect to any database path")
    return None

analyzer = initialize_analyzer()

# Available functions for the LLM to call
AVAILABLE_FUNCTIONS = {
    "get_location_info": {
        "name": "get_location_info",
        "description": "Get information about properties based on location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to search for"
                }
            },
            "required": ["location"]
        }
    },
    "get_price_analysis": {
        "name": "get_price_analysis",
        "description": "Get price analysis for properties",
        "parameters": {
            "type": "object",
            "properties": {
                "property_type": {
                    "type": "string",
                    "description": "Type of property to analyze"
                }
            }
        }
    },
    "get_property_details": {
        "name": "get_property_details",
        "description": "Get detailed information about properties",
        "parameters": {
            "type": "object",
            "properties": {
                "property_type": {
                    "type": "string",
                    "description": "Type of property to get details for"
                }
            }
        }
    }
}

class ChatMessage(BaseModel):
    user_input: str
    response_format: Optional[str] = "html"

def execute_function(function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the requested function and return the results
    """
    if not analyzer:
        return {"error": "Database connection not available"}
        
    try:
        if function_name == "get_location_info":
            data = analyzer._obtener_datos("", "identificacion_localizacion")
            return {"properties": data.get("info_basica", [])}
            
        elif function_name == "get_price_analysis":
            data = analyzer._obtener_datos("", "analisis_precio")
            return {"price_analysis": data.get("comparativa_precios", [])}
            
        elif function_name == "get_property_details":
            data = analyzer._obtener_datos("", "detalles_propiedad")
            return {"details": data.get("estadisticas_tipos", [])}
            
        else:
            raise ValueError(f"Unknown function: {function_name}")
            
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {str(e)}")
        return {"error": f"Error executing function: {str(e)}"}

# def build_prompt(data: Dict[str, Any], question: str) -> str:
#     """
#     Build the prompt with HTML structure and property data
#     """
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
#         # Construir la URL completa de la imagen
#         image_name = prop.get('image', 'default.bmp')
#         image_url = f"{MEDIA_URL}{image_name}"
        
#         properties_html += f"""
#         <div class="property-card">
#             <img src="{image_url}" alt="{prop.get('property_type', 'Propiedad')}" class="property-image">
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
#                 <a href="{prop.get('url', '')}" class="property-cta" target="_blank">Ver Detalles</a>
#             </div>
#         </div>
#         """

#     return f"""
#     <div class="message message-bot">
#         <div class="message-content">
#             <p>Aquí te muestro las propiedades disponibles según tu búsqueda:</p>
#         </div>
#     </div>
#     <d class="property-grid">
#         {properties_html if properties_html else '''
#         <div class="message message-bot">
#             <div class="message-content">
#                 <p>Lo siento, en este momento no hay propiedades disponibles que coincidan con tus criterios. 
#                 ¿Te gustaría que te notifique cuando tengamos nuevas opciones?</p>
#             </div>
#         </div>
#         '''}
#     </div>
        
    
#     """

def build_prompt(data: Dict[str, Any], question: str) -> str:
    if "error" in data:
        return f"""
        <div class="message message-bot">
            <div class="message-content">
                <p>Lo siento, hubo un problema al obtener los datos: {data['error']}</p>
            </div>
        </div>
        """

    properties_html = ""
    for prop in data.get('properties', []):
        
        # Limpia y formatea la URL
        details_url = prop.get('url', '#')
        
        # properties_html += f"""
        # <div class="property-card">
        #     <img src="{image_url}" alt="{prop.get('property_type', 'Propiedad')}" class="property-image">
        #     <div class="property-content">
        #         <div class="property-price">{prop.get('price', 'Precio no disponible')} USD</div>
        #         <div class="property-location">
        #             <i class="fas fa-map-marker-alt"></i> {prop.get('location', '')}, {prop.get('city_name', '')}, {prop.get('province_name', '')}, {prop.get('country_name', '')}
        #         </div>
        #         <div class="property-features">
        #             <span><i class="fas fa-bed"></i> {prop.get('promedio_dormitorios', 'N/A')} hab</span>
        #             <span><i class="fas fa-bath"></i> {prop.get('promedio_ambientes', 'N/A')} baños</span>
        #             <span><i class="fas fa-ruler-combined"></i> {prop.get('square_meters', 'N/A')} m²</span>
        #         </div>
        #         <p class="property-description">{prop.get('description', 'Sin descripción disponible')}</p>
        #         <a href="{details_url}" class="property-cta" target="_blank" rel="noopener noreferrer">Ver Detalles</a>
        #     </div>
        # </div>
        # """
        properties_html += f"""
        <div class="property-card">
            <img src="{prop.get('image', 'default.bmp')}" alt="{prop.get('property_type', 'Propiedad')}" class="property-image">
            <div class="property-content">
                <div class="property-price">{prop.get('price', 'Precio no disponible')} USD</div>
                <div class="property-location">
                    <i class="fas fa-map-marker-alt"></i> {prop.get('location', '')}, {prop.get('city_name', '')}, {prop.get('province_name', '')}, {prop.get('country_name', '')}
                </div>
                <div class="property-features">
                    <span><i class="fas fa-bed"></i> {prop.get('promedio_dormitorios', 'N/A')} hab</span>
                    <span><i class="fas fa-bath"></i> {prop.get('promedio_ambientes', 'N/A')} baños</span>
                    <span><i class="fas fa-ruler-combined"></i> {prop.get('square_meters', 'N/A')} m²</span>
                </div>
                <p class="property-description">{prop.get('description', 'Sin descripción disponible')}</p>
                <a href="{details_url}" class="property-cta" target="_blank" rel="noopener noreferrer">Ver Detalles</a>
            </div>
        </div>
        """

    return f"""
    <div class="message message-bot">
        <div class="message-content">
            <p>Aquí te muestro las propiedades disponibles según tu búsqueda:</p>
        </div>
    </div>
    <div class="property-grid">
        {properties_html if properties_html else '''
        <div class="message message-bot">
            <div class="message-content">
                <p>Lo siento, en este momento no hay propiedades disponibles que coincidan con tus criterios.</p>
            </div>
        </div>
        '''}
    </div>
    """

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    """
    Chat endpoint that handles both HTML and JSON responses with function calling
    """
    try:
        # First, ask the LLM which function to call based on the user's message
        function_selection_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a real estate assistant that helps users find and analyze properties. Based on the user's message, determine which function would be most appropriate to call."},
                {"role": "user", "content": chat_message.user_input}
            ],
            functions=list(AVAILABLE_FUNCTIONS.values()),
            function_call="auto"
        )

        # Extract the function call information
        message = function_selection_response.choices[0].message
        
        if message.get("function_call"):
            # Execute the selected function
            function_name = message["function_call"]["name"]
            function_params = eval(message["function_call"]["arguments"])
            data = execute_function(function_name, function_params)
        else:
            # If no function was called, provide a default response
            data = {"properties": []}
            
        # Build prompt with the retrieved data
        response_content = build_prompt(data, chat_message.user_input)
            
        # Return response based on requested format
        if chat_message.response_format == "html":
            return HTMLResponse(content=response_content)
        else:
            return JSONResponse(content={
                "response": response_content,
                "content_type": "html",
                "status": "success"
            })
                
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        error_response = {
            "status": "error",
            "detail": str(e),
            "content_type": "html",
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

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <div class="chat-container">
        <div class="message message-bot">
            <div class="message-content">
                <p>¡Bienvenido al Asistente Inmobiliario!</p>
            </div>
        </div>
    </div>
    """

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database_connected": analyzer is not None
    }

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8800))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)