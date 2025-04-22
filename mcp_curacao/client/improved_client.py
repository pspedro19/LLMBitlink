#!/usr/bin/env python3
"""
Improved client for MCP 1.6.0 compatibility.
Implements proper connection to all MCP servers and handles conversation flow.
"""

import asyncio
import os
import sys
import logging
import json
import shutil
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("improved-client")

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Custom imports
try:
    from config import (
        EXCEL_SERVER_ID,
        RAG_SERVER_ID,
        ORCHESTRATOR_SERVER_ID,
        RESPONSE_SERVER_ID
    )
    from utils.helpers import Colors
except ImportError as e:
    logger.error(f"Failed to import from config or utils: {e}")
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ImprovedClient:
    """Improved client for the Curaçao Tourism Assistant using MCP 1.6.0."""
    
    def __init__(self):
        """Initialize the client."""
        self.session_id = None
        self.servers = {}
        self.server_sessions = {}
        
        # Get terminal size for formatting
        try:
            self.term_width, self.term_height = shutil.get_terminal_size()
        except:
            self.term_width, self.term_height = 80, 24
    
    async def start_server(self, server_id, script_path):
        """
        Start an MCP server subprocess.
        
        Args:
            server_id: Server identifier
            script_path: Path to server script
        
        Returns:
            subprocess.Popen: Server process or None if failed
        """
        try:
            # Check if script exists
            if not os.path.exists(script_path):
                logger.error(f"Server script not found: {script_path}")
                return None
            
            logger.info(f"Starting server {server_id} with script {script_path}")
            
            # Start server process
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for initialization
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server {server_id} failed to start: {stderr}")
                print(f"Server {server_id} failed to start: {stderr}")
                return None
            
            logger.info(f"Started server {server_id}")
            return process
        except Exception as e:
            logger.error(f"Error starting server {server_id}: {e}")
            return None
    
    async def connect_to_server(self, server_id, script_path):
        """
        Connect to an MCP server using stdio.
        
        Args:
            server_id: Server identifier
            script_path: Path to server script
        
        Returns:
            ClientSession or None if failed
        """
        try:
            # Import MCP components
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
            except ImportError as e:
                logger.error(f"Failed to import MCP components: {e}")
                print(f"Error: MCP components not available - {e}")
                print("Try installing MCP 1.6.0: pip install mcp==1.6.0")
                return None
            
            # Start the server first
            server_process = await self.start_server(server_id, script_path)
            if not server_process:
                return None
            
            # Store the process
            self.servers[server_id] = {"process": server_process}
            
            # Create parameters
            params = StdioServerParameters(
                command="python",
                args=[script_path]
            )
            
            # Connect
            logger.info(f"Connecting to {server_id}...")
            read, write = await stdio_client(params)
            session = ClientSession(read, write)
            await session.initialize()
            logger.info(f"Successfully connected to {server_id}")
            
            # Store session
            self.servers[server_id]["session"] = session
            self.server_sessions[server_id] = session
            
            return session
        except Exception as e:
            logger.error(f"Error connecting to server {server_id}: {e}")
            print(f"Connection error: {e}")
            return None
    
    async def initialize(self):
        """Initialize all servers."""
        try:
            print(f"{Colors.DIM}Iniciando y conectando a servidores MCP...{Colors.RESET}")
            
            # Get base directory
            base_dir = Path(__file__).parent.parent
            servers_dir = base_dir / "servers"
            
            # Server configurations
            server_configs = {
                "excel": {
                    "id": EXCEL_SERVER_ID,
                    "path": str(servers_dir / "excel_server.py")
                },
                "rag": {
                    "id": RAG_SERVER_ID,
                    "path": str(servers_dir / "rag_server.py")
                },
                "orchestrator": {
                    "id": ORCHESTRATOR_SERVER_ID,
                    "path": str(servers_dir / "orchestrator_server.py")
                },
                "response": {
                    "id": RESPONSE_SERVER_ID,
                    "path": str(servers_dir / "response_server.py")
                }
            }
            
            # Connect to each server
            sessions = {}
            for name, config in server_configs.items():
                print(f"{Colors.DIM}Conectando a {config['id']}...{Colors.RESET}")
                session = await self.connect_to_server(config["id"], config["path"])
                if session:
                    sessions[name] = session
                    print(f"{Colors.GREEN}Conectado a {config['id']}{Colors.RESET}")
                else:
                    print(f"{Colors.RED}Error conectando a {config['id']}{Colors.RESET}")
                    return False
            
            # Create a user session
            logger.info("Creating user session...")
            try:
                result = await sessions["orchestrator"].call_tool(
                    "create_user_session", {}
                )
                self.session_id = result["session_id"]
                logger.info(f"User session created: {self.session_id}")
                
                print(f"{Colors.DIM}Sesión creada: {self.session_id}{Colors.RESET}")
                return True
            except Exception as e:
                logger.error(f"Failed to create user session: {e}")
                print(f"{Colors.RED}Error al crear sesión de usuario: {str(e)}{Colors.RESET}")
                return False
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            print(f"{Colors.RED}Error de inicialización: {str(e)}{Colors.RESET}")
            return False
    
    async def close(self):
        """Close all connections and terminate servers."""
        # Terminate all server processes
        for server_id, server_data in self.servers.items():
            process = server_data.get("process")
            if process and process.poll() is None:
                try:
                    logger.info(f"Terminating server {server_id}...")
                    process.terminate()
                    await asyncio.sleep(0.5)
                    if process.poll() is None:
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating server {server_id}: {e}")
    
    async def process_message(self, message):
        """
        Process a user message and generate a response.
        
        Args:
            message: User message
        
        Returns:
            str: Assistant response
        """
        try:
            # Add message to history
            logger.info(f"Processing message: {message}")
            
            # 1. Add message to history
            await self.server_sessions["orchestrator"].call_tool(
                "add_message_to_history", {
                    "session_id": self.session_id,
                    "sender": "user",
                    "message": message
                }
            )
            
            # 2. Extract preferences
            preferences = await self.server_sessions["orchestrator"].call_tool(
                "extract_preferences_from_message", {"message": message}
            )
            
            # 3. Update preferences if any were extracted
            if preferences:
                logger.info(f"Extracted preferences: {preferences}")
                await self.server_sessions["orchestrator"].call_tool(
                    "update_user_preferences", {
                        "session_id": self.session_id,
                        "preferences": preferences
                    }
                )
            
            # 4. Check missing information
            missing_info = await self.server_sessions["orchestrator"].call_tool(
                "determine_missing_information", {"session_id": self.session_id}
            )
            
            # 5. Get user context
            user_context = await self.server_sessions["orchestrator"].call_tool(
                "get_user_context", {"session_id": self.session_id}
            )
            
            # 6. Generate recommendations if we have enough information
            recommendations = []
            if not missing_info:
                try:
                    recommendations = await self.server_sessions["excel"].call_tool(
                        "get_recommendations", {
                            "interests": user_context.get("preferences", {}).get("interests", []),
                            "budget": user_context.get("preferences", {}).get("budget"),
                            "duration": user_context.get("preferences", {}).get("duration"),
                            "limit": 5
                        }
                    )
                    logger.info(f"Generated {len(recommendations)} recommendations")
                    
                    # Store recommendations
                    await self.server_sessions["orchestrator"].call_tool(
                        "store_recommendations", {
                            "session_id": self.session_id,
                            "recommendations": recommendations
                        }
                    )
                except Exception as rec_err:
                    logger.error(f"Error getting recommendations: {rec_err}")
            
            # 7. Get RAG information if needed
            rag_info = {}
            if "itinerario" in message.lower() or "plan" in message.lower():
                try:
                    query = f"Información sobre turismo en Curaçao"
                    rag_info = await self.server_sessions["rag"].call_tool(
                        "search_knowledge", {"query": query, "top_k": 3}
                    )
                except Exception as rag_err:
                    logger.error(f"Error getting RAG info: {rag_err}")
            
            # 8. Generate response
            try:
                response_data = await self.server_sessions["response"].call_tool(
                    "generate_response", {
                        "query": message,
                        "user_context": user_context,
                        "recommendations": recommendations,
                        "rag_info": rag_info
                    }
                )
                
                response = response_data.get("response", "Lo siento, no pude generar una respuesta.")
                
                # 9. Add response to history
                await self.server_sessions["orchestrator"].call_tool(
                    "add_message_to_history", {
                        "session_id": self.session_id,
                        "sender": "assistant",
                        "message": response
                    }
                )
                
                return response
            except Exception as resp_err:
                logger.error(f"Error generating response: {resp_err}")
                return f"Lo siento, ocurrió un error al generar la respuesta: {str(resp_err)}"
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}"
    
    def print_welcome(self):
        """Print welcome message."""
        title = "Asistente Turístico de Curaçao - Mejorado con MCP"
        
        print("\n" + "=" * self.term_width)
        print(f"{Colors.BOLD}{Colors.GREEN}{title.center(self.term_width)}{Colors.RESET}")
        print("=" * self.term_width)
        
        print(f"\n{Colors.CYAN}¡Bienvenido al asistente turístico de Curaçao!{Colors.RESET}")
        print("Puedo ayudarte a planificar tu viaje ideal con recomendaciones personalizadas.")
        print("Cuéntame sobre tus intereses, duración de estancia y presupuesto.")
        
        print("\nEscribe tu consulta y presiona Enter. Escribe '/salir' para terminar.")
        print("=" * self.term_width + "\n")
    
    def format_response(self, response):
        """Format response for terminal display."""
        lines = []
        for paragraph in response.split('\n'):
            if not paragraph.strip():
                lines.append("")
                continue
            
            # Wrap text to terminal width
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
        """Run the interactive chat."""
        # Initialize
        if not await self.initialize():
            print(f"\n{Colors.RED}Error inicializando el sistema. Verifique los logs.{Colors.RESET}\n")
            return
        
        # Show welcome message
        self.print_welcome()
        
        # Main loop
        try:
            while True:
                # Update terminal size
                try:
                    self.term_width, self.term_height = shutil.get_terminal_size()
                except:
                    pass
                
                # Get user input
                user_input = input(f"\n{Colors.BOLD}{Colors.GREEN}Tú:{Colors.RESET} ")
                
                # Check for exit command
                if user_input.lower() in ['/salir', '/exit', '/quit']:
                    print(f"\n{Colors.GREEN}¡Gracias por usar el asistente! ¡Hasta pronto!{Colors.RESET}\n")
                    break
                
                # Check if empty
                if not user_input.strip():
                    continue
                
                # Process message
                print(f"\n{Colors.DIM}Procesando tu mensaje...{Colors.RESET}")
                response = await self.process_message(user_input)
                
                # Show response
                print(f"\n{Colors.BOLD}{Colors.BLUE}Asistente:{Colors.RESET}")
                formatted_response = self.format_response(response)
                print(formatted_response)
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n{Colors.RED}Error inesperado: {str(e)}{Colors.RESET}\n")
        finally:
            # Close connections
            await self.close()

async def main():
    """Main function."""
    client = ImprovedClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())