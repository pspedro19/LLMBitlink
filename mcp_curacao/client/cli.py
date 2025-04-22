# mcp_curacao/client/cli.py
import asyncio
import argparse
import sys
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_curacao_cli.log"),
    ]
)
logger = logging.getLogger("mcp-cli")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import Colors
from chat_interface import CuracaoTravelAgent

async def main():
    """Función principal del cliente CLI"""
    parser = argparse.ArgumentParser(description='Asistente turístico de Curaçao con MCP+RAG')
    
    parser.add_argument('--debug', action='store_true',
                     help='Activa el modo de depuración (más logs)')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Crear e iniciar el agente
    agent = CuracaoTravelAgent()
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)