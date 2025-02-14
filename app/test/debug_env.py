# test/debug_env.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_env():
    # Encontrar el directorio raíz del proyecto
    current_dir = Path(__file__).resolve().parent
    app_dir = current_dir.parent
    project_dir = app_dir.parent
    
    logger.info(f"Buscando archivo .env en las siguientes rutas:")
    logger.info(f"1. Directorio actual: {current_dir}")
    logger.info(f"2. Directorio app: {app_dir}")
    logger.info(f"3. Directorio proyecto: {project_dir}")
    
    # Intentar cargar .env desde el directorio del proyecto
    env_path = project_dir / ".env"
    logger.info(f"\nIntentando cargar .env desde: {env_path}")
    
    if env_path.exists():
        logger.info("¡Archivo .env encontrado!")
        load_dotenv(env_path)
        
        # Mostrar variables cargadas
        logger.info("\nVariables de entorno cargadas:")
        logger.info(f"DB_HOST: {os.getenv('DB_HOST', 'No configurado')}")
        logger.info(f"DB_PORT: {os.getenv('DB_PORT', 'No configurado')}")
        logger.info(f"DB_NAME: {os.getenv('DB_NAME', 'No configurado')}")
        logger.info(f"DB_USER: {os.getenv('DB_USER', 'No configurado')}")
        if os.getenv('DB_PASSWORD'):
            logger.info(f"DB_PASSWORD: {'*' * len(os.getenv('DB_PASSWORD'))}")
        else:
            logger.info("DB_PASSWORD: No configurado")
    else:
        logger.error(f"¡Archivo .env NO encontrado en {env_path}!")
        
        # Mostrar el contenido del directorio
        logger.info("\nContenido del directorio del proyecto:")
        for item in project_dir.iterdir():
            logger.info(f"- {item.name}")

if __name__ == "__main__":
    debug_env()