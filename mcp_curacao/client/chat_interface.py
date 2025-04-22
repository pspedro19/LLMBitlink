# mcp_curacao/client/chat_interface.py
import asyncio
import os
import uuid
import shutil
import sys
import logging
from datetime import datetime

# Configurar logging silencioso (los logs van a un archivo, no a la consola)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mcp_curacao.log"),
    ]
)
logger = logging.getLogger("mcp-chat")

# Añadir directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EXCEL_SERVER_ID,
    RAG_SERVER_ID,
    ORCHESTRATOR_SERVER_ID,
    RESPONSE_SERVER_ID
)
from utils.helpers import Colors

class CuracaoTravelAgent:
    """Asistente de viajes de Curaçao con integración MCP+RAG"""
    
    def __init__(self):
        """Inicializa el asistente de viajes"""
        # Inicializar variables
        self.session_id = None
        self.clients = {}
        self.term_width, self.term_height = shutil.get_terminal_size()
    

    async def initialize(self):
        """Inicializa las conexiones a los servidores MCP"""
        try:
            # Intentar diferentes importaciones posibles
            try:
                from mcp.client import Client as MCPClient
            except ImportError:
                try:
                    from mcp import Client as MCPClient
                except ImportError:
                    from mcp.server.client import Client as MCPClient
            
            print(f"{Colors.DIM}Conectando a servidores MCP...{Colors.RESET}")
            
            # Resto del código...
            
            # Conectar a cada servidor
            servers = [
                ("excel", EXCEL_SERVER_ID),
                ("rag", RAG_SERVER_ID),
                ("orchestrator", ORCHESTRATOR_SERVER_ID),
                ("response", RESPONSE_SERVER_ID)
            ]
            
            for name, server_id in servers:
                self.clients[name] = MCPClient(server_id)
                await self.clients[name].connect()
                logger.info(f"Conectado a {server_id}")
                print(f"{Colors.DIM}Conectado a {server_id}{Colors.RESET}")
            
            # Crear sesión
            session_result = await self.clients["orchestrator"].call_tool(
                "create_user_session",
                {}
            )
            
            self.session_id = session_result.get("session_id")
            logger.info(f"Sesión creada: {self.session_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error al inicializar: {str(e)}")
            print(f"{Colors.RED}Error al inicializar: {str(e)}{Colors.RESET}")
            return False
    
    async def close(self):
        """Cierra las conexiones a los servidores MCP"""
        for name, client in self.clients.items():
            try:
                await client.disconnect()
            except:
                pass
        logger.info("Conexiones cerradas")
    
    async def process_message(self, message):
        """Procesa un mensaje del usuario y genera una respuesta"""
        logger.info(f"Procesando mensaje: {message}")
        print(f"{Colors.DIM}Procesando mensaje...{Colors.RESET}")
        
        # 1. Extraer preferencias del mensaje
        preferences = await self.clients["orchestrator"].call_tool(
            "extract_preferences_from_message",
            {"message": message}
        )
        
        # 2. Actualizar contexto del usuario
        if preferences:
            await self.clients["orchestrator"].call_tool(
                "update_user_preferences",
                {
                    "session_id": self.session_id,
                    "preferences": preferences
                }
            )
        
        # 3. Obtener contexto actualizado
        user_context = await self.clients["orchestrator"].call_tool(
            "get_user_context",
            {"session_id": self.session_id}
        )
        
        # 4. Determinar información faltante
        missing_info = await self.clients["orchestrator"].call_tool(
            "determine_missing_information",
            {"session_id": self.session_id}
        )
        
        # 5. Si tenemos toda la información, generar recomendaciones
        recommendations = []
        rag_info = {}
        
        if not missing_info:
            # Obtener preferencias para búsqueda
            prefs = user_context.get("preferences", {})
            
            # Buscar recomendaciones en Excel
            print(f"{Colors.DIM}Buscando recomendaciones...{Colors.RESET}")
            excel_results = await self.clients["excel"].call_tool(
                "get_recommendations",
                {
                    "interests": prefs.get("interests", []),
                    "budget": prefs.get("budget"),
                    "duration": prefs.get("duration"),
                    "locations": prefs.get("locations", ["Curaçao"]),
                    "limit": 5
                }
            )
            
            # Si hay recomendaciones, buscar información adicional con RAG
            if excel_results:
                # Construir consulta para RAG basada en las recomendaciones
                places = [rec.get("name", "") for rec in excel_results[:2]]
                if places:
                    query = f"Información sobre {', '.join(places)} en Curaçao"
                    print(f"{Colors.DIM}Consultando RAG: {query}{Colors.RESET}")
                    rag_results = await self.clients["rag"].call_tool(
                        "search_knowledge",
                        {"query": query, "top_k": 3}
                    )
                    
                    # Enriquecer recomendaciones con RAG
                    enriched_results = await self.clients["orchestrator"].call_tool(
                        "enrich_recommendations",
                        {
                            "recommendations": excel_results,
                            "rag_results": rag_results
                        }
                    )
                    
                    recommendations = enriched_results
                    rag_info = rag_results
                else:
                    recommendations = excel_results
            else:
                recommendations = excel_results
        
        # 6. Registrar mensaje en historial
        await self.clients["orchestrator"].call_tool(
            "add_message_to_history",
            {
                "session_id": self.session_id,
                "sender": "user",
                "message": message
            }
        )
        
        # 7. Generar respuesta
        print(f"{Colors.DIM}Generando respuesta...{Colors.RESET}")
        response_data = await self.clients["response"].call_tool(
            "generate_response",
            {
                "query": message,
                "user_context": user_context,
                "recommendations": recommendations,
                "rag_info": rag_info,
                "use_llm": True
            }
        )
        
        # 8. Registrar respuesta en historial
        await self.clients["orchestrator"].call_tool(
            "add_message_to_history",
            {
                "session_id": self.session_id,
                "sender": "assistant",
                "message": response_data.get("response", "")
            }
        )
        
        # 9. Guardar recomendaciones en el contexto
        if recommendations:
            await self.clients["orchestrator"].call_tool(
                "store_recommendations",
                {
                    "session_id": self.session_id,
                    "recommendations": recommendations
                }
            )
        
        return response_data.get("response", "")
    
    def print_welcome(self):
        """Imprime mensaje de bienvenida"""
        title = "Asistente Turístico de Curaçao - Powered by MCP+RAG"
        
        print("\n" + "=" * self.term_width)
        print(f"{Colors.BOLD}{Colors.GREEN}{title.center(self.term_width)}{Colors.RESET}")
        print("=" * self.term_width)
        
        print(f"\n{Colors.CYAN}¡Bienvenido al asistente turístico de Curaçao!{Colors.RESET}")
        print("Puedo ayudarte a planificar tu viaje con recomendaciones personalizadas.")
        print("Cuéntame sobre tus intereses, duración de estancia y presupuesto.")
        
        print("\nEscribe tu consulta y presiona Enter. Escribe '/salir' para terminar.")
        print("=" * self.term_width + "\n")
    
    def format_response(self, response):
        """Formatea la respuesta para mostrar en terminal"""
        lines = []
        for paragraph in response.split('\n'):
            if not paragraph.strip():
                lines.append("")
                continue
            wrapped = []
            current_line = ""
            for word in paragraph.split():
                if len(current_line + " " + word) > self.term_width - 4:
                    wrapped.append(current_line)
                    current_line = word
                else:
                    if not current_line:
                        current_line = word
                    else:
                        current_line += " " + word
            if current_line:
                wrapped.append(current_line)
            lines.extend(wrapped)
        
        formatted = "\n".join(["    " + line for line in lines])
        return formatted
    
    async def run(self):
        """Ejecuta el chat interactivo"""
        # Inicializar componentes
        if not await self.initialize():
            print(f"\n{Colors.RED}Error inicializando el sistema. Verifique los logs para más detalles.{Colors.RESET}\n")
            return
        
        # Mostrar bienvenida
        self.print_welcome()
        
        # Loop principal
        try:
            while True:
                # Actualizar tamaño de terminal
                self.term_width, self.term_height = shutil.get_terminal_size()
                
                # Solicitar entrada
                user_input = input(f"\n{Colors.BOLD}{Colors.GREEN}Tú:{Colors.RESET} ")
                
                # Verificar si es comando de salida
                if user_input.lower() in ['/salir', '/exit', '/quit']:
                    print(f"\n{Colors.GREEN}¡Gracias por usar el asistente turístico de Curaçao! ¡Hasta pronto!{Colors.RESET}\n")
                    break
                
                # Verificar si está vacío
                if not user_input.strip():
                    continue
                
                # Procesar mensaje
                response = await self.process_message(user_input)
                
                # Mostrar respuesta
                print(f"\n{Colors.BOLD}{Colors.BLUE}Asistente:{Colors.RESET}")
                formatted_response = self.format_response(response)
                print(formatted_response)
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            print(f"\n{Colors.RED}Error inesperado: {str(e)}{Colors.RESET}\n")
        finally:
            # Cerrar conexiones
            await self.close()