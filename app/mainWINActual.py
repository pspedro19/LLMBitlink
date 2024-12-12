import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from dotenv import load_dotenv
import os
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

engine = 'gpt-3.5-turbo'

# Pydantic models
class ChatMessage(BaseModel):
    user_input: str
    response_format: Optional[str] = "html"

def build_prompt(data: Dict[str, Any], question: str) -> str:
    """
    Build the prompt with enhanced HTML structure and property data
    """
    conversations = "\n".join([
        f"Input: {conv['input']} -> Output: {conv['output']}" 
        for conv in data.get('conversations', [])
    ])
    
    chunks = "\n".join([
        f"Document {chunk['document_id']}: {chunk['content'][:50]}" 
        for chunk in data.get('chunks', [])
    ])
    
    # Construir el HTML de las propiedades
    properties_html = ""
    for prop in data.get('properties', []):
        properties_html += f"""
        <div class="property-card">
            <img src="{prop.get('image', '/default-property.jpg')}" alt="{prop.get('property_type', 'Propiedad')}" class="property-image">
            <div class="property-content">
                <div class="property-price">{prop.get('price', 'Precio no disponible')} USD</div>
                <div class="property-location">
                    <i class="fas fa-map-marker-alt"></i> {prop.get('location', '')}, {prop.get('city_name', '')}, {prop.get('province_name', '')}, {prop.get('country_name', '')}
                </div>
                <div class="property-features">
                    <span><i class="fas fa-bed"></i> {prop.get('promedio_ambientes', 'N/A')} hab</span>
                    <span><i class="fas fa-bath"></i> {prop.get('promedio_dormitorios', 'N/A')} baños</span>
                    <span><i class="fas fa-ruler-combined"></i> {prop.get('square_meters', 'N/A')} m²</span>
                </div>
                <p class="property-description">{prop.get('description', 'Sin descripción disponible')}</p>
                <a href="{prop.get('url', '#')}" class="property-cta">Ver Detalles</a>
            </div>
        </div>
        """
    
    # Construir la respuesta HTML completa
    html_response = f"""
    
        
        <div class="message message-bot">
            <div class="message-content">
                <p>Aquí te muestro las propiedades disponibles según tu búsqueda:</p>
            </div>
        </div>
        <div class="property-grid">
            {properties_html if properties_html else '''
            <div class="message message-bot">
                <div class="message-content">
                    <p>Lo siento, en este momento no hay propiedades disponibles que coincidan con tus criterios. 
                    ¿Te gustaría que te notifique cuando tengamos nuevas opciones?</p>
                </div>
            </div>
            '''}
        </div>
        
    
    """

    # Construir el prompt final
    prompt = f"""
    # Sistema
    Eres un API endpoint especializado en responder consultas inmobiliarias. Debes usar el siguiente HTML como base para tu respuesta, 
    pero puedes modificar el mensaje inicial según la pregunta del usuario. Mantén el formato y las clases CSS.

    # HTML Base:
    {html_response}

    # Contexto:
    Conversaciones previas:
    {conversations}

    Chunks relevantes:
    {chunks}

    # Pregunta del usuario:
    {question}

    # Instrucciones:
    1. Responde como Max, el experto inmobiliario
    2. Utiliza la información de las propiedades proporcionada
    3. Mantén el formato HTML existente
    4. Personaliza el mensaje inicial según la pregunta del usuario
    5. Asegúrate de que la respuesta sea relevante y útil
    """
    
    return prompt

def basic_html_validation(html_content: str) -> bool:
    """
    Simple HTML validation to ensure basic structure is present
    """
    required_classes = [
        'chat-messages',
        'message message-bot',
        'message-content'
    ]
    
    return all(class_name in html_content for class_name in required_classes)

def fetch_data_from_django():
    """
    Fetch data from Django API
    """
    try:
        django_url = "http://161.35.120.130:8000/get_all_data/"
        response = requests.get(django_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching data from Django: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching data from Django API")

def get_completion_from_openai(prompt: str) -> str:
    """
    Get completion from OpenAI
    """
    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in OpenAI API call")

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    """
    Chat endpoint that handles both HTML and JSON responses
    """
    try:
        # Fetch data from Django
        data = fetch_data_from_django()
        
        # Build prompt
        prompt = build_prompt(data, chat_message.user_input)
        
        # Get response from OpenAI
        response_content = get_completion_from_openai(prompt)
        
        # Basic HTML validation
        if not basic_html_validation(response_content):
            logger.warning("Invalid HTML structure received from OpenAI")
            response_content = f"""
                <div>{response_content}</div>
            """
        
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
                <div class="chat-messages">
                    <div class="message message-bot error-message">
                        <div class="message-content">
                            <p>Lo siento, ha ocurrido un error al procesar tu solicitud.</p>
                        </div>
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
        <div class="chat-messages">
            <div class="message message-bot">
                <div class="message-content">
                    <p>¡Bienvenido al Asistente Inmobiliario!</p>
                </div>
            </div>
        </div>
    </div>
    """

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8800))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)