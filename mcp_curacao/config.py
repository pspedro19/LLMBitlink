# mcp_curacao/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas principales
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Rutas a los datos
EXCEL_DIR = os.path.join(PROJECT_DIR, "app3", "data", "database")
RAG_APP_DIR = os.path.join(PROJECT_DIR, "app2")

# Configuración de base de datos para RAG
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
    "database": os.getenv("DB_NAME", "vector_db")
}

# Configuración del LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Definir qué modelo usar basado en disponibilidad
def get_llm_provider():
    if OPENAI_API_KEY:
        return "openai"
    elif ANTHROPIC_API_KEY:
        return "claude"
    return "templates"

# Nombres de servidores MCP
EXCEL_SERVER_ID = "curacao-excel-server"
RAG_SERVER_ID = "curacao-rag-server"
ORCHESTRATOR_SERVER_ID = "curacao-orchestrator-server"
RESPONSE_SERVER_ID = "curacao-response-server"

# Configuración de recomendaciones
MAX_RECOMMENDATIONS = 5