#!/usr/bin/env python3
"""
Simple MCP client that works with MCP 1.6.0.
This client directly connects to your MCP servers using stdio communication.
"""

import asyncio
import os
import sys
import logging
import shutil
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple-client")

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class SimpleClient:
    """Simple client for MCP Curaçao Assistant."""
    
    def __init__(self):
        """Initialize the client."""
        self.session_id = None
        self.server_processes = {}
        self.server_sessions = {}
        
        # Get terminal size
        try:
            self.term_width, self.term_height = shutil.get_terminal_size()
        except:
            self.term_width, self.term_height = 80, 24
    
    async def start_servers(self):
        """Start all MCP servers."""
        print(f"{Colors.BLUE}Starting MCP servers...{Colors.RESET}")
        
        # Get servers directory
        base_dir = Path.cwd()
        servers_dir = base_dir / "mcp_curacao" / "servers"
        
        if not servers_dir.exists():
            print(f"{Colors.RED}Server directory not found at: {servers_dir}{Colors.RESET}")
            return False
        
        # Server files
        server_scripts = [
            ("Excel", servers_dir / "excel_server.py"),
            ("RAG", servers_dir / "rag_server.py"),
            ("Orchestrator", servers_dir / "orchestrator_server.py"),
            ("Response", servers_dir / "response_server.py")
        ]
        
        # Start each server
        for name, script_path in server_scripts:
            if not script_path.exists():
                print(f"{Colors.RED}Server script not found: {script_path}{Colors.RESET}")
                continue
                
            print(f"{Colors.DIM}Starting {name} server...{Colors.RESET}")
            
            try:
                # Start server process
                process = subprocess.Popen(
                    ["python", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for server to start
                await asyncio.sleep(1)
                
                # Check if process is running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"{Colors.RED}Failed to start {name} server: {stderr}{Colors.RESET}")
                    continue
                
                # Store process
                self.server_processes[name] = process
                print(f"{Colors.GREEN}✓ {name} server started{Colors.RESET}")
                
            except Exception as e:
                logger.error(f"Error starting {name} server: {e}")
                print(f"{Colors.RED}Error starting {name} server: {e}{Colors.RESET}")
        
        # Check if all servers started
        if len(self.server_processes) < 4:
            print(f"{Colors.YELLOW}Warning: Not all servers started. Some features may not work.{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}All servers started successfully!{Colors.RESET}")
        
        return len(self.server_processes) > 0
    
    async def connect_to_servers(self):
        """Connect to all running servers using MCP."""
        print(f"{Colors.BLUE}Connecting to MCP servers...{Colors.RESET}")
        
        try:
            # Import MCP components
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            print(f"{Colors.RED}Failed to import MCP: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Try installing MCP 1.6.0: pip install mcp==1.6.0{Colors.RESET}")
            return False
        
        # Get servers directory
        base_dir = Path.cwd()
        servers_dir = base_dir / "mcp_curacao" / "servers"
        
        # Server files
        server_scripts = [
            ("Excel", servers_dir / "excel_server.py"),
            ("RAG", servers_dir / "rag_server.py"),
            ("Orchestrator", servers_dir / "orchestrator_server.py"),
            ("Response", servers_dir / "response_server.py")
        ]
        
        # Connect to each server
        for name, script_path in server_scripts:
            if not name in self.server_processes:
                print(f"{Colors.YELLOW}Skipping connection to {name} (server not running){Colors.RESET}")
                continue
                
            print(f"{Colors.DIM}Connecting to {name} server...{Colors.RESET}")
            
            try:
                # Create parameters
                params = StdioServerParameters(
                    command="python",
                    args=[str(script_path)]
                )
                
                # Connect to server
                read, write = await stdio_client(params)
                session = ClientSession(read, write)
                await session.initialize()
                
                # Store session
                self.server_sessions[name] = session
                print(f"{Colors.GREEN}✓ Connected to {name} server{Colors.RESET}")
                
            except Exception as e:
                logger.error(f"Error connecting to {name} server: {e}")
                print(f"{Colors.RED}Error connecting to {name} server: {e}{Colors.RESET}")
        
        # Check connections
        if not self.server_sessions:
            print(f"{Colors.RED}Failed to connect to any servers{Colors.RESET}")
            return False
            
        if "Orchestrator" not in self.server_sessions:
            print(f"{Colors.RED}Failed to connect to Orchestrator server (required){Colors.RESET}")
            return False
        
        # Create user session
        try:
            result = await self.server_sessions["Orchestrator"].call_tool(
                "create_user_session", {}
            )
            self.session_id = result["session_id"]
            print(f"{Colors.GREEN}User session created: {self.session_id}{Colors.RESET}")
        except Exception as e:
            logger.error(f"Failed to create user session: {e}")
            print(f"{Colors.RED}Failed to create user session: {e}{Colors.RESET}")
            return False
        
        return True
    
    async def process_message(self, message):
        """Process a user message and generate a response."""
        logger.info(f"Processing message: {message}")
        
        try:
            # Add message to history
            await self.server_sessions["Orchestrator"].call_tool(
                "add_message_to_history", {
                    "session_id": self.session_id,
                    "sender": "user",
                    "message": message
                }
            )
            
            # Extract preferences
            preferences = await self.server_sessions["Orchestrator"].call_tool(
                "extract_preferences_from_message", {"message": message}
            )
            
            if preferences:
                logger.info(f"Extracted preferences: {preferences}")
                await self.server_sessions["Orchestrator"].call_tool(
                    "update_user_preferences", {
                        "session_id": self.session_id,
                        "preferences": preferences
                    }
                )
            
            # Get user context
            user_context = await self.server_sessions["Orchestrator"].call_tool(
                "get_user_context", {"session_id": self.session_id}
            )
            
            # Generate recommendations if needed
            recommendations = []
            if "Excel" in self.server_sessions:
                try:
                    preferences = user_context.get("preferences", {})
                    recommendations = await self.server_sessions["Excel"].call_tool(
                        "get_recommendations", {
                            "interests": preferences.get("interests", []),
                            "budget": preferences.get("budget", 0),
                            "duration": preferences.get("duration", 0),
                            "limit": 5
                        }
                    )
                except Exception as e:
                    logger.error(f"Error getting recommendations: {e}")
            
            # Generate response
            if "Response" in self.server_sessions:
                response_data = await self.server_sessions["Response"].call_tool(
                    "generate_response", {
                        "query": message,
                        "user_context": user_context,
                        "recommendations": recommendations
                    }
                )
                
                response = response_data.get("response", "")
                
                # Add response to history
                await self.server_sessions["Orchestrator"].call_tool(
                    "add_message_to_history", {
                        "session_id": self.session_id,
                        "sender": "assistant",
                        "message": response
                    }
                )
                
                return response
            else:
                # Fallback if Response server not available
                if not user_context.get("preferences", {}).get("interests"):
                    return "¿Qué tipo de actividades te interesarían en Curaçao? Por ejemplo, ¿prefieres experiencias culturales, playas, gastronomía o vida nocturna?"
                elif not user_context.get("preferences", {}).get("duration"):
                    return "¿Por cuántos días planeas visitar Curaçao? Esto me ayudará a organizar mejor las recomendaciones."
                elif not user_context.get("preferences", {}).get("budget"):
                    return "¿Tienes un presupuesto aproximado para tu viaje? Esto me ayudará a sugerirte opciones adecuadas."
                else:
                    return "Gracias por la información. Estoy procesando tus preferencias para ofrecerte las mejores recomendaciones."
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}"
    
    def print_welcome(self):
        """Print welcome message."""
        title = "Asistente Turístico de Curaçao - MCP Cliente Simple"
        
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
    
    async def close(self):
        """Close connections and terminate server processes."""
        for name, session in self.server_sessions.items():
            try:
                await session.close()
                logger.info(f"Closed {name} session")
            except:
                pass
        
        for name, process in self.server_processes.items():
            try:
                print(f"{Colors.DIM}Stopping {name} server...{Colors.RESET}")
                process.terminate()
                await asyncio.sleep(0.5)
                if process.poll() is None:
                    process.kill()
                logger.info(f"Terminated {name} server")
            except:
                pass
    
    async def run(self):
        """Run the interactive client."""
        # Start servers
        if not await self.start_servers():
            print(f"{Colors.RED}Failed to start required servers. Exiting.{Colors.RESET}")
            return
        
        # Connect to servers
        if not await self.connect_to_servers():
            print(f"{Colors.RED}Failed to connect to required servers. Exiting.{Colors.RESET}")
            await self.close()
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
                print(f"{Colors.DIM}Procesando tu consulta...{Colors.RESET}")
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
            # Close connections and terminate servers
            await self.close()

async def main():
    """Main function."""
    client = SimpleClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())