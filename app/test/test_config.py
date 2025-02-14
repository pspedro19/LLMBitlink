# test/test_config.py
import asyncio
import logging
import pytest
from core.rag.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_config_and_db():
    """Prueba la configuración y la conexión a la base de datos"""
    try:
        config = Config()
        
        # Verificar que se cargaron las variables de entorno
        logger.info("Configuración cargada:")
        logger.info(f"  DB_HOST: {config.DB_HOST}")
        logger.info(f"  DB_PORT: {config.DB_PORT}")
        logger.info(f"  DB_NAME: {config.DB_NAME}")
        logger.info(f"  DB_USER: {config.DB_USER}")
        logger.info(f"  DB_PASSWORD: {'*' * len(config.DB_PASSWORD) if config.DB_PASSWORD else 'No configurado'}")
        
        # Probar conexión a la base de datos
        import asyncpg
        conn = await asyncpg.connect(
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            host=config.DB_HOST,
            port=config.DB_PORT
        )
        
        logger.info("Conexión a la base de datos exitosa!")
        
        # Verificar que existan las tablas necesarias
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        logger.info("Tablas encontradas:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        await conn.close()
        
    except FileNotFoundError as e:
        logger.error(f"Error de configuración: {e}")
        raise
    except asyncpg.InvalidPasswordError:
        logger.error("Error: Credenciales de base de datos inválidas")
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_config_and_db())