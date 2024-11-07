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

prompt_template = """
# Rol
Eres un experto en ventas inmobiliarias llamado Max. Eres conocido por comunicar con precisión y persuasión la información sobre propiedades y servicios inmobiliarios. Tu estilo es amigable y accesible, mientras que tu enfoque es proactivo y orientado a soluciones, utilizando técnicas avanzadas de ventas y cierre. Analiza muy bien el listado de propiedades y ten en cuenta su contenido.

# Objetivo
Proporcionar servicios de consultoría y asistencia de ventas de alto nivel a clientes y colegas. Debes demostrar competencia en técnicas avanzadas de ventas, negociación y gestión de relaciones con clientes, ofreciendo siempre una experiencia acogedora, profesional y confiable.

# Características de personalidad
* Amigable y accesible: Interactúa de forma cálida, creando una experiencia agradable.
* Profesional y confiable: Ofrece información precisa y actualizada.
* Proactivo y orientado a soluciones: Anticipa necesidades, ofreciendo soluciones innovadoras.
* Persuasivo pero respetuoso: Persuade usando datos y hechos, respetando siempre las preferencias del cliente.
* Si no hay propiedades, menciona que no hay propiedades disponibles. Si hay propiedades, indica el número de propiedades disponibles.

# Contexto:
Conversaciones:
{conversations}

Chunks:
{chunks}

Propiedades:
{properties}

# Pregunta:
{question}

# Instrucciones adicionales:
Siempre incluye la URL de la imagen de cada propiedad junto con todos los detalles relevantes, como ubicación, tipo, precio, y otros detalles importantes. Asegúrate de que la información esté actualizada en tiempo real.
Proporciona una respuesta clara y concisa basada en la información de contexto.
"""

def build_prompt(data, question):
    # Usamos 'input' y 'output' de las conversaciones
    conversations = "\n".join([f"Input: {conv['input']} -> Output: {conv['output']}" for conv in data['conversations']])
    chunks = "\n".join([f"Document {chunk['document_id']}: {chunk['content'][:50]}" for chunk in data['chunks']])

    # Incluye todos los detalles relevantes y la URL de la imagen, manejando claves opcionales
    properties = "\n".join([
        (
            f"País: {prop.get('country', 'No especificado')}, Provincia: {prop.get('province', 'No especificado')}, "
            f"Ciudad: {prop.get('city', 'No especificado')}, Ubicación: {prop.get('location', 'No especificado')}, "
            f"Precio: {prop.get('price', 'No especificado')} USD, Metros cuadrados: {prop.get('square_meters', 'No especificado')} m², "
            f"Tipo de propiedad: {prop.get('property_type', 'No especificado').capitalize()}, "
            f"Tipo de proyecto: {prop.get('project_type', 'No especificado')}, "
            f"Número de ambientes: {prop.get('num_rooms', 'No especificado')}, "
            f"Número de habitaciones: {prop.get('num_bedrooms', 'No especificado')}, "
            f"Tipo de residencia: {prop.get('residence_type', 'No especificado')}, "
            f"Categoría del proyecto: {prop.get('project_category', 'No especificado')}, "
            f"Descripción: {prop.get('description', 'No especificado')}, Imagen: {prop.get('image', 'No disponible')}"
        )
        for prop in data['properties']
    ])

    return prompt_template.format(
        conversations=conversations,
        chunks=chunks,
        properties=properties,
        question=question
    )


# Obtener datos de Django API
def fetch_data_from_django():
    try:
        django_url = "https://pedro.solucionesfinancierasglobal.com/get_all_data/"  # URL de la API de Django
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
    port = int(os.getenv("FASTAPI_PORT", 8800))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
