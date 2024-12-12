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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True, 
   allow_methods=["*"],
   allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

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

# def build_prompt(data, question, max_history=5):
#    # Limitar historial de conversaciones
#    recent_conversations = data['conversations'][-max_history:] if data['conversations'] else []
#    conversations = "\n".join([
#        f"Input: {conv['input']} -> Output: {conv['output']}" 
#        for conv in recent_conversations
#    ])

#    # Filtrar propiedades relevantes por ubicación mencionada en la pregunta
#    relevant_properties = [
#        prop for prop in data['properties']
#        if any(keyword in question.lower() 
#              for keyword in prop.get('location', '').lower().split())
#    ][:3]  # Limitar a 3 propiedades más relevantes

#    properties = "\n".join([
#        f"País: {prop.get('country', 'No especificado')}, "
#        f"Provincia: {prop.get('province', 'No especificado')}, "
#        f"Ciudad: {prop.get('city', 'No especificado')}, "
#        f"Ubicación: {prop.get('location', 'No especificado')}, "
#        f"Precio: {prop.get('price', 'No especificado')} USD, "
#        f"Metros cuadrados: {prop.get('square_meters', 'No especificado')} m², "
#        f"Tipo de propiedad: {prop.get('property_type', 'No especificado').capitalize()}, "
#        f"Tipo de proyecto: {prop.get('project_type', 'No especificado')}, "
#        f"Número de ambientes: {prop.get('num_rooms', 'No especificado')}, "
#        f"Número de habitaciones: {prop.get('num_bedrooms', 'No especificado')}, "
#        f"Tipo de residencia: {prop.get('residence_type', 'No especificado')}, "
#        f"Categoría del proyecto: {prop.get('project_category', 'No especificado')}, "
#        f"Descripción: {prop.get('description', 'No especificado')[:200]}..., "
#        f"Imagen: {prop.get('image', 'No disponible')}"
#        for prop in (relevant_properties or data['properties'][:3])
#    ])

#    return prompt_template.format(
#        conversations=conversations,
#        chunks="",  # Omitir chunks para reducir tokens
#        properties=properties,
#        question=question
#    )

def fetch_data_from_django():
   try:
       django_url = "http://161.35.120.130:8000/get_all_data/"
       response = requests.get(django_url)
       if response.status_code == 200:
           return response.json()
       else:
           logger.error(f"Error al obtener los datos de Django: {response.status_code}")
           raise HTTPException(status_code=500, detail="Error fetching data from Django API.")
   except Exception as e:
       logger.error(f"Error en la solicitud a la API de Django: {str(e)}")
       raise HTTPException(status_code=500, detail="Error fetching data from Django API.")

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

# def build_prompt(data, question, max_history=5, max_properties=3):
#     total_tokens = 0
    
#     # Iniciar con conversaciones más recientes
#     recent_conversations = data['conversations'][-max_history:] if data['conversations'] else []
#     conversations = []
    
#     # Agregar conversaciones mientras no excedan el límite
#     for conv in recent_conversations:
#         conv_text = f"Input: {conv['input']} -> Output: {conv['output']}"
#         if total_tokens + len(conv_text.split()) < 3000:
#             conversations.append(conv_text)
#             total_tokens += len(conv_text.split())
    
#     # Filtrar propiedades relevantes
#     relevant_properties = []
#     keywords = question.lower().split()
    
#     for prop in data['properties']:
#         location = prop.get('location', '').lower()
#         if any(keyword in location for keyword in keywords):
#             relevant_properties.append(prop)
#             if len(relevant_properties) >= max_properties:
#                 break
    
#     # Si no hay propiedades relevantes, tomar las primeras
#     if not relevant_properties:
#         relevant_properties = data['properties'][:max_properties]
    
#     # Construir cadena de propiedades con información truncada
#     properties = "\n".join([
#         f"Ubicación: {prop.get('location', '')}, "
#         f"Precio: {prop.get('price', '')} USD, "
#         f"Metros cuadrados: {prop.get('square_meters', '')} m², "
#         f"Descripción: {prop.get('description', '')[:150]}..."
#         for prop in relevant_properties
#     ])
    
#     return prompt_template.format(
#         conversations="\n".join(conversations),
#         chunks="",
#         properties=properties,
#         question=question
#     )

def build_prompt(data, question):
   # Propiedades base (contexto esencial siempre presente)
   base_properties = [
       f"País: {prop.get('country', 'No especificado')}, "
       f"Ciudad: {prop.get('city', 'No especificado')}, "
       f"Ubicación: {prop.get('location', 'No especificado')}, "
       f"Precio: {prop.get('price', 'No especificado')} USD, "
       f"Metros cuadrados: {prop.get('square_meters', 'No especificado')} m², "
       f"Tipo: {prop.get('property_type', 'No especificado').capitalize()}"
       for prop in data['properties'][:5]  # Limitar a 5 propiedades base
   ]
   properties_context = "\n".join(base_properties)
   
   # Manejo de conversaciones dentro del límite
   total_tokens = len(properties_context.split()) + 1000  # Buffer para prompt_template
   conversations = []
   
   for conv in data['conversations'][-5:]:  # Últimas 5 conversaciones
       conv_text = f"Input: {conv['input']} -> Output: {conv['output']}"
       if total_tokens + len(conv_text.split()) < 3500:
           conversations.append(conv_text)
           total_tokens += len(conv_text.split())
   
   # Propiedades detalladas relevantes a la pregunta
   relevant_properties = [
       prop for prop in data['properties']
       if any(keyword in prop.get('location', '').lower() 
             for keyword in question.lower().split())
   ][:2]  # Máximo 2 propiedades detalladas

   detailed_properties = "\n".join([
       f"Ubicación: {prop.get('location', '')}\n"
       f"Precio: {prop.get('price', '')} USD\n"
       f"Metros cuadrados: {prop.get('square_meters', '')} m²\n"
       f"Tipo de propiedad: {prop.get('property_type', '').capitalize()}\n"
       f"Descripción: {prop.get('description', '')[:200]}...\n"
       f"Imagen: {prop.get('image', 'No disponible')}"
       for prop in (relevant_properties or data['properties'][:2])
   ])

   return prompt_template.format(
       conversations="\n".join(conversations),
       chunks="",
       properties=f"{properties_context}\n\nInformación detallada:\n{detailed_properties}",
       question=question
   )

@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    try:
        data = fetch_data_from_django()
        max_history = 5
        
        while max_history > 0:
            prompt = build_prompt(data, chat_message.user_input, max_history=max_history)
            if len(prompt.split()) <= 4000:
                result = get_completion_from_openai(prompt)
                return {"response": result}
            max_history -= 1
        
        # Si aún excede, usar respuesta mínima
        minimal_prompt = build_prompt(data, chat_message.user_input, max_history=1, max_properties=1)
        result = get_completion_from_openai(minimal_prompt)
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