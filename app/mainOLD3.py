import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
import requests
import logging
from fastapi.middleware.cors import CORSMiddleware

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI()

# Configuración de middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia "*" por los dominios específicos si prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verificar y cargar la clave API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY no está configurada. Por favor, verifica tu archivo .env.")
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

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

def truncate_conversations(conversations, max_entries=5):
    """
    Trunca las conversaciones a las últimas `max_entries` entradas.
    """
    return conversations[-max_entries:] if len(conversations) > max_entries else conversations

def truncate_chunks(chunks, max_chunks=5, snippet_length=50):
    """
    Trunca los chunks a los primeros `max_chunks` documentos y los reduce a `snippet_length` caracteres.
    """
    return [
        f"Document {chunk['document_id']}: {chunk['content'][:snippet_length]}"
        for chunk in chunks[:max_chunks]
    ]

def truncate_properties(properties, max_properties=3):
    """
    Limita las propiedades a un máximo de `max_properties`.
    """
    return properties[:max_properties]

def build_prompt(data, question):
    """
    Construye el prompt limitando el tamaño de las secciones para ajustarse al límite de tokens.
    """
    truncated_conversations = truncate_conversations(data.get('conversations', []), max_entries=5)
    truncated_chunks = truncate_chunks(data.get('chunks', []), max_chunks=5, snippet_length=50)
    truncated_properties = truncate_properties(data.get('properties', []), max_properties=3)

    conversations = "\n".join([f"Input: {conv['input']} -> Output: {conv['output']}" for conv in truncated_conversations])
    chunks = "\n".join(truncated_chunks)
    properties = "\n".join([
        (
            f"País: {prop.get('country', 'No especificado')}, Provincia: {prop.get('province', 'No especificado')}, "
            f"Ciudad: {prop.get('city', 'No especificado')}, Ubicación: {prop.get('location', 'No especificado')}, "
            f"Precio: {prop.get('price', 'No especificado')} USD, Metros cuadrados: {prop.get('square_meters', 'No especificado')} m², "
            f"Tipo de propiedad: {prop.get('property_type', 'No especificado').capitalize()}, "
            f"Descripción: {prop.get('description', 'No especificado')}, Imagen: {prop.get('image', 'No disponible')}"
        )
        for prop in truncated_properties
    ])

    return prompt_template.format(
        conversations=conversations,
        chunks=chunks,
        properties=properties,
        question=question
    )

def fetch_data_from_django():
    """
    Recupera datos desde la API de Django.
    """
    try:
        django_url = "http://161.35.120.130:8000/get_all_data/"  # URL de la API de Django
        response = requests.get(django_url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error al obtener los datos de Django: {response.status_code}")
            raise HTTPException(status_code=500, detail="Error fetching data from Django API.")
    except requests.RequestException as e:
        logger.error(f"Error de conexión a la API de Django: {e}")
        raise HTTPException(status_code=500, detail="Error connecting to Django API.")

def count_tokens(prompt):
    """
    Calcula el número aproximado de tokens en un prompt.
    """
    return len(prompt.split())

def get_completion_from_openai(prompt):
    """
    Solicita una respuesta a la API de OpenAI con el prompt proporcionado.
    """
    try:
        response = openai.ChatCompletion.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message["content"]
    except openai.error.InvalidRequestError as e:
        if "context_length_exceeded" in str(e):
            logger.error("El prompt excede el límite de tokens permitido.")
            raise HTTPException(
                status_code=400,
                detail="La pregunta o el contexto es demasiado largo. Por favor, intenta reducir la cantidad de información."
            )
        else:
            logger.error(f"Error en la API de OpenAI: {e}")
            raise HTTPException(status_code=500, detail="Error in OpenAI API call.")

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    logger.info(f"Mensaje recibido: {chat_message.user_input}")
    try:
        data = fetch_data_from_django()
        prompt = build_prompt(data, chat_message.user_input)

        # Contar tokens del prompt
        token_count = count_tokens(prompt)
        logger.info(f"Tokens en el prompt: {token_count}")
        if token_count > 16000:
            raise HTTPException(status_code=400, detail="El prompt generado es demasiado largo.")

        result = get_completion_from_openai(prompt)
        logger.info(f"Respuesta de OpenAI: {result}")
        return {"response": result}

    except Exception as e:
        logger.error(f"Error procesando el mensaje: {e}", exc_info=True)
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
