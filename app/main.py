import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
import requests
import logging
from fastapi import Request
import io
import logging
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por los dominios específicos si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI API
# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

engine = 'gpt-3.5-turbo'

class ChatMessage(BaseModel):
    user_input: str

# Prompt template (unchanged)
prompt_template = """
# Rol
Eres un experto en ventas inmobiliarias llamado Max. Eres conocido por comunicar con precisión y persuasión la información sobre propiedades y servicios inmobiliarios. Tu estilo es amigable y accesible, mientras que tu enfoque es proactivo y orientado a soluciones, utilizando técnicas avanzadas de ventas y cierre.

# Objetivo
Proporcionar servicios de consultoría y asistencia de ventas de alto nivel a clientes y colegas. Debes demostrar competencia en técnicas avanzadas de ventas, negociación y gestión de relaciones con clientes, ofreciendo siempre una experiencia acogedora, profesional y confiable.

# Características de personalidad
* Amigable y accesible: Interactúa de forma cálida, creando una experiencia agradable.
* Profesional y confiable: Ofrece información precisa y actualizada.
* Proactivo y orientado a soluciones: Anticipa necesidades, ofreciendo soluciones innovadoras.
* Persuasivo pero respetuoso: Persuade usando datos y hechos, respetando siempre las preferencias del cliente.

# Contexto:
Conversaciones:
{conversations}

Chunks:
{chunks}

Propiedades:
{properties}

# Pregunta:
{question}

Proporciona una respuesta clara y concisa basada en la información de contexto.
"""

# Función para construir el prompt
def build_prompt(data, question):
    # Ahora solo usamos 'input' y 'output' de las conversaciones, ya que 'user' ha sido eliminado
    conversations = "\n".join([f"Input: {conv['input']} -> Output: {conv['output']}" for conv in data['conversations']])
    chunks = "\n".join([f"Document {chunk['document_id']}: {chunk['content'][:50]}" for chunk in data['chunks']])
    properties = "\n".join([f"{prop['property_type']} en {prop['location']} - {prop['price']} USD" for prop in data['properties']])

    return prompt_template.format(
        conversations=conversations,
        chunks=chunks,
        properties=properties,
        question=question
    )

# Obtener datos de Django API
def fetch_data_from_django():
    try:
        django_url = "http://52.67.197.89:8800/get_all_data/"  # URL de la API de Django
        response = requests.get(django_url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error al obtener los datos de Django: {response.status_code}")
            raise HTTPException(status_code=500, detail="Error fetching data from Django API.")
    except Exception as e:
        logger.error(f"Error en la solicitud a la API de Django: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching data from Django API.")

# Obtener respuesta de OpenAI
def get_completion_from_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except openai.error.OpenAIError as e:
        logger.error(f"Error en la API de OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in OpenAI API call")

# Ruta /chat/ que utiliza RAG con la información de Django y OpenAI
@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    logger.debug(f"Mensaje recibido: {chat_message.user_input}")
    try:
        # Obtener datos de Django API
        data = fetch_data_from_django()

        # Construir el prompt con la información de Django y la entrada del usuario
        prompt = build_prompt(data, chat_message.user_input)

        # Obtener respuesta de OpenAI
        result = get_completion_from_openai(prompt)

        logger.debug(f"Respuesta de OpenAI: {result}")
        return {"response": result}
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OpenAI Integration!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
