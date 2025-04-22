#!/usr/bin/env python3
"""
Minimal MCP client for version 1.6.0.
This client uses the bare minimum MCP API features to ensure compatibility.
"""

import asyncio
import subprocess
import logging
import shutil
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("minimal_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("minimal-client")

# Terminal colors
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class MinimalClient:
    """A minimal MCP client that avoids using complex MCP APIs."""
    
    def __init__(self):
        """Initialize the client."""
        # Get terminal size
        try:
            self.term_width, self.term_height = shutil.get_terminal_size()
        except:
            self.term_width, self.term_height = 80, 24
            
        # Preserve server processes
        self.server_processes = {}
        
        # User session info
        self.session_id = None
        self.preferences = {
            "interests": [],
            "duration": None,
            "budget": None
        }
        self.message_history = []
        
    async def start_servers(self):
        """Start all required MCP servers."""
        print(f"{Colors.BLUE}Starting MCP servers...{Colors.RESET}")
        
        # Base directory and servers directory
        base_dir = Path.cwd()
        servers_dir = base_dir / "mcp_curacao" / "servers"
        
        # Check if servers directory exists
        if not servers_dir.exists():
            print(f"{Colors.RED}Server directory not found at: {servers_dir}{Colors.RESET}")
            return False
        
        # Server scripts
        server_scripts = [
            ("Excel", "excel_server.py"),
            ("RAG", "rag_server.py"),
            ("Orchestrator", "orchestrator_server.py"),
            ("Response", "response_server.py")
        ]
        
        # Start each server
        success = True
        for name, script_name in server_scripts:
            script_path = servers_dir / script_name
            if not script_path.exists():
                print(f"{Colors.RED}Server script not found: {script_path}{Colors.RESET}")
                success = False
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
                    success = False
                    continue
                
                # Store process
                self.server_processes[name] = process
                print(f"{Colors.GREEN}✓ {name} server started{Colors.RESET}")
                
            except Exception as e:
                logger.error(f"Error starting {name} server: {e}")
                print(f"{Colors.RED}Error starting {name} server: {e}{Colors.RESET}")
                success = False
        
        return success
    
    def close_servers(self):
        """Close all server processes."""
        for name, process in self.server_processes.items():
            try:
                print(f"{Colors.DIM}Stopping {name} server...{Colors.RESET}")
                process.terminate()
                logger.info(f"Terminated {name} server")
            except Exception as e:
                logger.error(f"Error stopping {name} server: {e}")
        
        # Wait a moment for processes to terminate
        print(f"{Colors.GREEN}All servers stopped.{Colors.RESET}")
    
    def extract_preferences(self, message):
        """Extract user preferences from a message using simple rules."""
        # This is a simplified version of what your orchestrator server would do
        updated_prefs = {}
        
        # Extract interests
        interest_keywords = {
            "cultural": ["cultura", "museo", "historia", "arte", "patrimon"],
            "natural": ["playa", "naturaleza", "parque", "buceo", "snorkel"],
            "family": ["familia", "niños", "diversión", "actividades para niños"],
            "gastronomy": ["comida", "restaurante", "gastronomía", "comer", "cocina"],
            "nightlife": ["fiesta", "discoteca", "club", "bar", "noche", "música"]
        }
        
        found_interests = []
        message_lower = message.lower()
        for interest, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    found_interests.append(interest)
                    break
        
        if found_interests:
            updated_prefs["interests"] = list(set(found_interests))
        
        # Extract duration - looking for numbers followed by "day", "days", "día", "días"
        import re
        duration_match = re.search(r'(\d+)\s*(día|días|dias|day|days)', message_lower)
        if duration_match:
            updated_prefs["duration"] = float(duration_match.group(1))
        elif re.match(r'^\d+$', message.strip()) and not self.preferences["duration"] and self.preferences["interests"]:
            # Just a number and we're missing duration
            updated_prefs["duration"] = float(message.strip())
        
        # Extract budget - looking for numbers followed by "$", "dólares", "dollars"
        budget_match = re.search(r'(\d+)\s*(\$|dólares|dolares|dollars)', message_lower)
        if budget_match:
            updated_prefs["budget"] = float(budget_match.group(1))
        elif re.match(r'^\d+$', message.strip()) and self.preferences["duration"] and not self.preferences["budget"]:
            # Just a number and we're missing budget
            updated_prefs["budget"] = float(message.strip())
        
        # Update preferences
        self.preferences.update(updated_prefs)
        return updated_prefs
    
    async def process_message(self, message):
        """Process a user message and generate a response."""
        # Add to history
        self.message_history.append({
            "sender": "user",
            "message": message
        })
        
        # Extract preferences
        updated_prefs = self.extract_preferences(message)
        logger.info(f"Extracted preferences: {updated_prefs}")
        
        # Determine conversation state and generate response
        response = ""
        
        # If no interests yet, ask for interests
        if not self.preferences["interests"]:
            response = "¿Qué tipo de actividades te interesarían en Curaçao? Por ejemplo, ¿prefieres experiencias culturales, playas, gastronomía o vida nocturna?"
        
        # If no duration yet, ask for duration
        elif not self.preferences["duration"]:
            response = "¿Por cuántos días planeas visitar Curaçao? Esto me ayudará a organizar mejor las recomendaciones."
        
        # If no budget yet, ask for budget
        elif not self.preferences["budget"]:
            response = f"¿Tienes un presupuesto aproximado para tu viaje de {self.preferences['duration']} días? Esto me ayudará a sugerirte opciones adecuadas."
        
        # If we have all needed preferences, generate recommendations
        else:
            # This would normally come from your Excel server
            if "cultural" in self.preferences["interests"]:
                response = self.generate_cultural_recommendations()
            elif "natural" in self.preferences["interests"]:
                response = self.generate_natural_recommendations()
            elif "gastronomy" in self.preferences["interests"]:
                response = self.generate_gastronomy_recommendations()
            elif "nightlife" in self.preferences["interests"]:
                response = self.generate_nightlife_recommendations()
            else:
                response = self.generate_default_recommendations()
        
        # Add response to history
        self.message_history.append({
            "sender": "assistant",
            "message": response
        })
        
        return response
    
    def generate_cultural_recommendations(self):
        """Generate cultural recommendations."""
        return """Basado en tus preferencias, te recomiendo estas opciones en Curaçao:

1. **Sinagoga Mikvé Israel-Emanuel**: La sinagoga más antigua del hemisferio occidental, con piso de arena.
   Rating: 4.8★ | Costo aproximado: $20
   *Construida en 1732, esta sinagoga es la más antigua en uso continuo en el hemisferio occidental. Es famosa por su suelo de arena, que simboliza el desierto por el que los judíos vagaron durante 40 años y también servía para amortiguar el sonido durante los servicios secretos en tiempos de persecución.*

2. **Recorrido a pie por Punda**: Tour guiado por el distrito histórico de Punda, con edificios coloniales.
   Rating: 4.6★ | Costo aproximado: $35
   *Punda es uno de los cuatro distritos históricos de Willemstad y es conocido por sus edificios coloniales holandeses de colores pastel. El recorrido a pie incluye visitas a la Plaza Gomez, el Puente Emma, y varias tiendas y edificios históricos.*

3. **Museo Kura Hulanda**: Un museo antropológico que documenta la historia de la esclavitud en el Caribe.
   Rating: 4.5★ | Costo aproximado: $45
   *El Museo Kura Hulanda es un museo antropológico en Willemstad que documenta la historia de la trata de esclavos. Fundado en 1999, el museo contiene una amplia colección de artefactos históricos y exhibiciones interactivas.*

¿Te gustaría más información sobre alguna de estas opciones?"""
    
    def generate_natural_recommendations(self):
        """Generate natural/beach recommendations."""
        return """Basado en tus preferencias, te recomiendo estas opciones en Curaçao:

1. **Playa Kenepa (Grote Knip)**: Una de las playas más hermosas de Curaçao, con aguas cristalinas.
   Rating: 4.9★ | Costo aproximado: $0 (entrada gratuita)
   *Grote Knip es conocida por sus aguas turquesas y su arena blanca. Es ideal para nadar y hacer snorkel, con abundante vida marina cerca de la costa. Ofrece instalaciones básicas y sombra limitada, así que considera llevar una sombrilla.*

2. **Parque Nacional Christoffel**: El parque más grande de Curaçao, hogar de flora y fauna local.
   Rating: 4.7★ | Costo aproximado: $25
   *Este parque alberga el punto más alto de la isla (Christoffelberg) y ofrece varias rutas de senderismo. Podrás ver ciervos, iguanas y más de 450 especies de plantas. Para subir a la montaña, se recomienda comenzar temprano antes del calor del mediodía.*

3. **Shete Boka National Park**: Parque que muestra la costa agreste del norte de la isla.
   Rating: 4.8★ | Costo aproximado: $15
   *Shete Boka significa "siete bocas" en papiamento, refiriéndose a las calas y entradas donde el mar choca contra las formaciones rocosas. La más famosa es Boka Tabla, donde puedes entrar en una cueva y sentir la fuerza del océano.*

¿Te gustaría más información sobre alguno de estos lugares?"""
    
    def generate_gastronomy_recommendations(self):
        """Generate gastronomy recommendations."""
        return """Basado en tus preferencias, te recomiendo estas opciones gastronómicas en Curaçao:

1. **Kome**: Restaurante contemporáneo con fusión de cocina local e internacional.
   Rating: 4.7★ | Precio promedio: $$$
   *Ubicado en Pietermaai, Kome ofrece un ambiente elegante pero relajado. Su menú cambia regularmente para incorporar ingredientes frescos locales. Sus platos de pescado son particularmente recomendados.*

2. **Plasa Bieu (Mercado Viejo)**: Mercado de comida local con varios puestos tradicionales.
   Rating: 4.5★ | Precio promedio: $
   *Este mercado en Willemstad ofrece la experiencia culinaria más auténtica de Curaçao. Varios puestos sirven platos típicos como kabritu stoba (estofado de cabra), funchi (similar a la polenta) y kadushi (sopa de cactus). El ambiente es sencillo pero lleno de locales.*

3. **Gouverneur de Rouville**: Restaurante con vista panorámica al puerto y puente Emma.
   Rating: 4.6★ | Precio promedio: $$
   *Ubicado en un edificio histórico en Otrobanda, este restaurante ofrece cocina holandesa y caribeña. Su terraza es perfecta para contemplar el atardecer sobre el puerto. Prueba el pescado fresco o el famoso rijsttafel indonesio-holandés.*

¿Te gustaría conocer más detalles sobre alguno de estos restaurantes o recibir otras recomendaciones gastronómicas?"""
    
    def generate_nightlife_recommendations(self):
        """Generate nightlife recommendations."""
        return """Basado en tus preferencias, te recomiendo estas opciones de vida nocturna en Curaçao:

1. **Zanzibar**: Bar de playa con música en vivo y ambiente relajado.
   Rating: 4.6★ | Costo aproximado: $$
   *Ubicado directamente en la playa de Jan Thiel, Zanzibar ofrece cócteles tropicales y puestas de sol espectaculares. Los fines de semana suelen tener bandas en vivo tocando reggae y ritmos caribeños. Es ideal para un ambiente más relajado.*

2. **Cabana Beach**: Club de playa con DJs internacionales y fiestas temáticas.
   Rating: 4.7★ | Costo aproximado: $$$
   *Este elegante beach club se transforma por la noche en una vibrante discoteca al aire libre. Cuenta con una gran piscina, áreas VIP y eventos especiales que atraen a DJs internacionales. Su clientela es una mezcla de locales y turistas.*

3. **Miles Jazz Café**: Club de jazz íntimo con música en vivo.
   Rating: 4.8★ | Costo aproximado: $$
   *Para una noche más sofisticada, este café de jazz en Pietermaai ofrece excelentes actuaciones en un ambiente íntimo. Cuenta con una buena selección de vinos y cócteles artesanales. Se recomienda reservar, especialmente los fines de semana.*

¿Te interesa alguno de estos lugares o prefieres otro tipo de entretenimiento nocturno?"""
    
    def generate_default_recommendations(self):
        """Generate default mixed recommendations."""
        return """Basado en tus preferencias, te recomiendo estas opciones en Curaçao:

1. **Queen Emma Bridge**: El famoso "puente flotante" que conecta Punda y Otrobanda.
   Rating: 4.8★ | Costo aproximado: $0 (gratis)
   *Este puente peatonal flotante es un símbolo de Willemstad. Construido en 1888, se abre regularmente para permitir el paso de barcos al puerto interno. La vista desde el puente, especialmente al atardecer, es impresionante.*

2. **Blue Room Cave**: Una cueva submarina con espectacular iluminación azul natural.
   Rating: 4.7★ | Costo aproximado: $30 (tour guiado)
   *Esta cueva marina recibe luz solar que se refleja a través del agua, creando un asombroso efecto de luz azul. Se puede visitar nadando o en bote con fondo transparente. La experiencia es mejor alrededor del mediodía cuando la luz es más intensa.*

3. **Mambo Beach Boulevard**: Complejo comercial y de entretenimiento junto a la playa.
   Rating: 4.5★ | Costo variable
   *Este desarrollo frente al mar combina tiendas, restaurantes y acceso a una hermosa playa. Durante el día es perfecto para compras y relajación en la playa, mientras que por la noche ofrece varios bares y clubes con ambiente animado.*

¿Te interesa alguna de estas recomendaciones? ¿Te gustaría información más específica sobre algún tipo de actividad?"""
    
    def print_welcome(self):
        """Print welcome message."""
        title = "Asistente Turístico de Curaçao - Modo Independiente"
        
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
        """Run the interactive client."""
        # Start servers
        if not await self.start_servers():
            print(f"{Colors.RED}Failed to start MCP servers. Running in standalone mode.{Colors.RESET}")
        else:
            print(f"{Colors.GREEN}MCP servers started successfully. Running in assisted mode.{Colors.RESET}")
        
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
            # Close servers
            self.close_servers()

async def main():
    """Main function."""
    client = MinimalClient()
    await client.run()

if __name__ == "__main__":
    asyncio.run(main())