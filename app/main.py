import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
import json
from memory import EnhancedVectorMemory

# Specify the path to the .env file and load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

# Initialize the FastAPI app
app = FastAPI()

# Set up OpenAI client configuration with environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
engine = 'gpt-3.5-turbo'

# Define the Pydantic model for chat message input validation
class ChatMessage(BaseModel):
    user_input: str

# Prompt for the AI model
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

# Habilidades clave a modelar
1. Comunicación efectiva: Simplifica la complejidad del mercado y describe las propiedades de forma clara.
2. Técnicas avanzadas de venta y cierre: Adapta las estrategias según el contexto y necesidades del cliente, incluyendo manejo de objeciones.
3. Análisis de datos del mercado inmobiliario: Analiza grandes volúmenes de datos para ofrecer recomendaciones detalladas sobre las tendencias del mercado.
4. Gestión eficiente de CRM: Utiliza herramientas CRM para automatizar y optimizar la gestión de relaciones, mejorando la eficiencia y la experiencia del cliente.

# Capacidades tecnológicas
* Integración con herramientas de realidad virtual para tours de propiedades, proporcionando una experiencia inmersiva y detallada.
* Uso de sistemas CRM avanzados para seguimiento y análisis de clientes, facilitando una gestión personalizada.
* Implementación de marketing digital para atraer y retener clientes mediante estrategias orientadas a datos.

# Ejemplos de interacción
- **Consulta inicial:** "Basado en tus preferencias y las tendencias actuales del mercado, te recomiendo considerar propiedades en estas áreas específicas..."
- **Negociación:** "Si estás interesado en hacer una oferta, puedo ayudarte a estructurarla de manera que sea atractiva para el vendedor y se ajuste a tu presupuesto. Además, puedo mostrarte cómo otros clientes han encontrado valor en propuestas similares."

# Notas
* Responde siempre en español latino.
* Sé persuasivo, específico y detallado, usando técnicas de venta como la escasez para crear urgencia.
* Responde únicamente a la consulta específica sin incluir información irrelevante.

Question: {question}  Context: {context}
"""

# Load memory from JSON file
memory_dir = "/data"  # Ruta absoluta dentro del contenedor
memory_path = os.path.join(memory_dir, 'memory.json')
print(f"Memory directory path: {memory_dir}")  # Print the absolute path for debugging
print(f"Memory file path: {memory_path}")  # Print the absolute path for debugging

if not os.path.exists(memory_dir):
    os.makedirs(memory_dir)
    print(f"Created missing directory: {memory_dir}")

if not os.path.exists(memory_path):
    # Create the file if it doesn't exist
    with open(memory_path, 'w') as file:
        json.dump([], file)
    print(f"Created missing memory file at: {memory_path}")

memory = EnhancedVectorMemory(memory_path)

# Function to generate text completions using the OpenAI API
def get_completion(user_input, engine="gpt-3.5-turbo"):
    memory_response = memory.get_closest_memory(user_input)
    enhanced_input = f"{memory_response} {user_input}" if memory_response else user_input
    prompt = prompt_template.format(question=enhanced_input, context="")
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=0.5  # Controls the randomness of the model's output
    )
    return response.choices[0].message["content"]

# Define a POST endpoint for the chat interaction
@app.post("/chat/")
async def chat_with_agent(chat_message: ChatMessage):
    try:
        result = get_completion(chat_message.user_input, engine)
        memory.add_to_memory(chat_message.user_input, result)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define a GET endpoint to verify server operation
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI OpenAI Integration!"}

# If the script is executed as the main script, run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8800, reload=True)
