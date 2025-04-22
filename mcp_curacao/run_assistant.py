#!/usr/bin/env python3
"""
Curaçao Tourism Assistant - Enhanced Version with Server Integration
This assistant helps visitors plan their trip to Curaçao with personalized recommendations
by leveraging RAG and Excel servers for dynamic information retrieval.
"""

import asyncio
import subprocess
import sys
import os
import shutil
import signal
import time
import logging
import random
import re
import json
import aiohttp
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("curacao-assistant")

# Terminal colors
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class CuracaoAssistant:
    """Enhanced Curaçao Tourism Assistant with server integration for dynamic information retrieval."""
    
    def __init__(self):
        """Initialize the assistant."""
        # Server processes and endpoints
        self.server_processes = {}
        self.server_ports = {
            "Excel": 5001,
            "RAG": 5002,
            "Orchestrator": 5003,
            "Response": 5004
        }
        self.base_url = "http://localhost"
        
        # Get terminal size
        try:
            self.term_width, self.term_height = shutil.get_terminal_size()
        except:
            self.term_width, self.term_height = 80, 24
        
        # User data
        self.preferences = {
            "interests": [],
            "duration": None,
            "budget": None,
            "family_size": None,
            "has_children": None,
            "children_ages": []
        }
        self.confirmed_preferences = {}  # Track which preferences have been confirmed
        self.session = {}  # Session context for the conversation
        self.message_history = []
        
        # Conversation state
        self.conversation_stage = "greeting"
        self.user_name = None
        self.current_hour = datetime.now().hour  # For time-sensitive greetings
        
        # Topics to potentially explore - will be populated dynamically
        self.available_topics = []
        
        # Prevents repeatedly asking for the same information
        self.itinerary_offered = False
        self.booking_info_offered = False
        
        # For tracking conversation frustration levels
        self.frustration_level = 0
        
    async def start_servers(self):
        """Start all MCP servers in the background."""
        print(f"{Colors.BLUE}Iniciando servidores MCP...{Colors.RESET}")
        
        # Get base directory and try multiple potential server locations
        base_dir = Path.cwd()
        
        # Try multiple possible server locations
        possible_server_dirs = [
            base_dir / "servers",  # Direct child
            base_dir / "mcp_curacao" / "servers",  # In mcp_curacao subdir
            Path(__file__).parent / "servers",  # Relative to this script
            Path(__file__).parent.parent / "mcp_curacao" / "servers",  # Parent dir
            base_dir / "LLMBitlink" / "mcp_curacao" / "servers"  # Specific path from logs
        ]
        
        servers_dir = None
        for dir_path in possible_server_dirs:
            if dir_path.exists() and dir_path.is_dir():
                logger.info(f"Found server directory at: {dir_path}")
                servers_dir = dir_path
                break
        
        if not servers_dir:
            print(f"{Colors.RED}¡Directorio de servidores no encontrado! Buscando en: {possible_server_dirs}{Colors.RESET}")
            return False
        
        print(f"{Colors.GREEN}Encontrado directorio de servidores en: {servers_dir}{Colors.RESET}")
        
        # Server configuration
        server_configs = [
            ("Excel", servers_dir / "excel_server.py"),
            ("RAG", servers_dir / "rag_server.py"),
            ("Orchestrator", servers_dir / "orchestrator_server.py"),
            ("Response", servers_dir / "response_server.py")
        ]
        
        # Start each server
        for name, script_path in server_configs:
            if not script_path.exists():
                print(f"{Colors.RED}Script no encontrado: {script_path}{Colors.RESET}")
                continue
            
            print(f"{Colors.DIM}Iniciando servidor {name}...{Colors.RESET}")
            
            try:
                process = subprocess.Popen(
                    ["python", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait briefly for the server to start
                await asyncio.sleep(1)
                
                # Check if still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"{Colors.RED}Error al iniciar servidor {name}: {stderr}{Colors.RESET}")
                    continue
                    
                self.server_processes[name] = process
                print(f"{Colors.GREEN}Servidor {name} iniciado correctamente{Colors.RESET}")
                
            except Exception as e:
                logger.error(f"Error starting {name} server: {e}")
                print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        
        # Wait a bit for all servers to be ready
        await asyncio.sleep(2)
        return len(self.server_processes) > 0
    
    async def query_rag(self, query, topic=None):
        """Query the RAG server for information."""
        endpoint = f"{self.base_url}:{self.server_ports['RAG']}/query"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "topic": topic,
                    "context": self.session
                }
                
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "No se encontró información relevante.")
                    else:
                        logger.error(f"RAG query failed with status: {response.status}")
                        return "Lo siento, no puedo acceder a esa información en este momento."
        except Exception as e:
            logger.error(f"Error querying RAG server: {e}")
            return f"Lo siento, tuve un problema al buscar esa información: {str(e)}"
    
    async def query_excel(self, filter_criteria=None):
        """Query the Excel server for structured data."""
        endpoint = f"{self.base_url}:{self.server_ports['Excel']}/query"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query_type": "recommendations",
                    "filters": filter_criteria or self.preferences,
                    "context": self.session
                }
                
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    else:
                        logger.error(f"Excel query failed with status: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error querying Excel server: {e}")
            return []
    
    async def get_orchestrated_response(self, message, intent):
        """Get an orchestrated response combining RAG and Excel data."""
        endpoint = f"{self.base_url}:{self.server_ports['Orchestrator']}/respond"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "message": message,
                    "intent": intent,
                    "preferences": self.preferences,
                    "context": self.session,
                    "history": self.message_history[-5:] if len(self.message_history) > 0 else [],
                    "conversation_stage": self.conversation_stage
                }
                
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "No pude generar una respuesta apropiada.")
                    else:
                        logger.error(f"Orchestrator request failed with status: {response.status}")
                        return "Lo siento, hay un problema con mi sistema de respuesta."
        except Exception as e:
            logger.error(f"Error communicating with Orchestrator server: {e}")
            # Fallback to a basic response
            return self.get_fallback_response(intent)
    
    def get_fallback_response(self, intent):
        """Generate a fallback response when servers are unavailable."""
        if intent == "greeting":
            return "¡Hola! Soy tu asistente de viaje para Curaçao. ¿En qué puedo ayudarte hoy? Puedo recomendarte lugares según tus intereses, crear itinerarios y brindarte información para reservas."
        
        elif intent == "help":
            return "Puedo ayudarte a planificar tu viaje a Curaçao recomendándote lugares según tus intereses, crear itinerarios personalizados y darte información para reservas. ¿Qué te gustaría saber?"
        
        elif self.conversation_stage == "collecting_preferences":
            missing_prefs = []
            if not self.preferences["interests"]:
                missing_prefs.append("intereses (playas, cultura, gastronomía, etc.)")
            if not self.preferences["duration"]:
                missing_prefs.append("duración de tu estancia")
            if not self.preferences["budget"]:
                missing_prefs.append("presupuesto aproximado")
            
            if missing_prefs:
                return f"Para ayudarte mejor, necesito conocer tu {' y '.join(missing_prefs)}. ¿Puedes compartir esta información conmigo?"
            else:
                return "Gracias por la información. ¿Hay algo específico que te gustaría conocer sobre Curaçao?"
        
        else:
            return "Lo siento, estoy teniendo problemas para acceder a mi sistema de información. ¿Puedes intentar reformular tu pregunta?"
    
    def extract_preferences(self, message):
        """Extract user preferences from message text more effectively."""
        updated = {}
        message_lower = message.lower()
        
        # Try to extract user name
        name_patterns = [
            r'me llamo (\w+)',
            r'soy (\w+)',
            r'mi nombre es (\w+)'
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, message_lower)
            if name_match:
                self.user_name = name_match.group(1).capitalize()
                break
        
        # Extract interests with broader keyword recognition
        interest_keywords = {
            "cultural": ["cultura", "museo", "historia", "arte", "patrimon", "monumento", "cultural", "arquitectura", "histórico"],
            "natural": ["playa", "naturaleza", "parque", "buceo", "snorkel", "acuatic", "natural", "oceano", "mar", "playa"],
            "family": ["familia", "niños", "diversión", "actividades para niños", "pequeños", "familiar", "hijos", "somos"],
            "gastronomy": ["comida", "restaurante", "gastronomía", "comer", "cocina", "platos", "gastronomico", "gastro", "culinaria"],
            "nightlife": ["fiesta", "discoteca", "club", "bar", "noche", "música", "bailar", "nocturno", "diversión"]
        }
        
        interests = []
        for interest, keywords in interest_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    interests.append(interest)
                    break
        
        if interests:
            # Update interests without duplicates
            current_interests = set(self.preferences["interests"])
            for interest in interests:
                if interest not in current_interests:
                    current_interests.add(interest)
            
            self.preferences["interests"] = list(current_interests)
            updated["interests"] = interests
        
        # Extract family size
        family_size_matches = re.search(r'somos (\d+)(?:\s*personas)?', message_lower)
        if family_size_matches:
            self.preferences["family_size"] = int(family_size_matches.group(1))
            updated["family_size"] = self.preferences["family_size"]
        
        family_ref_matches = re.search(r'con mi familia', message_lower)
        if family_ref_matches and not self.preferences["family_size"]:
            self.preferences["family_size"] = "múltiple"  # Unknown size but it's a family
            updated["family_size"] = self.preferences["family_size"]
            
        # Detect children
        children_mention = re.search(r'(?:hijos?|niños?|pequeños?|bebés?)', message_lower)
        if children_mention:
            self.preferences["has_children"] = True
            updated["has_children"] = True
            
            # Try to extract children ages
            children_ages = re.findall(r'(\d+)\s*años', message_lower)
            if children_ages:
                self.preferences["children_ages"] = [int(age) for age in children_ages]
                updated["children_ages"] = self.preferences["children_ages"]
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s*(día|días|dias|day|days)', message_lower)
        if duration_match:
            self.preferences["duration"] = float(duration_match.group(1))
            updated["duration"] = self.preferences["duration"]
        elif re.match(r'^\d+$', message.strip()) and not self.preferences["duration"] and self.preferences["interests"]:
            # Just a number and we're missing duration
            self.preferences["duration"] = float(message.strip())
            updated["duration"] = self.preferences["duration"]
        
        # Extract budget with broader pattern matching
        budget_patterns = [
            r'(\d+)\s*(?:usd|dólares?|dolares?|euros?|€|\$)',
            r'presupuesto\s*(?:de|es|:)?\s*(\d+)',
            r'gastar?\s*(?:hasta)?\s*(\d+)',
            r'(?:tengo|tenemos)\s*(\d+)\s*(?:para el viaje|para gastar|para vacacionar)'
        ]
        
        for pattern in budget_patterns:
            budget_match = re.search(pattern, message_lower)
            if budget_match:
                self.preferences["budget"] = float(budget_match.group(1))
                updated["budget"] = self.preferences["budget"]
                break
        
        # Update the confirmed preferences
        for key, value in updated.items():
            if key not in self.confirmed_preferences:
                self.confirmed_preferences[key] = False  # New preference, not confirmed yet
        
        # Update conversation stage based on preferences
        if self.conversation_stage == "greeting" and updated:
            self.conversation_stage = "collecting_preferences"
        
        if (self.preferences["interests"] and self.preferences["duration"] and 
            self.preferences["budget"] and all(self.confirmed_preferences.values())):
            self.conversation_stage = "providing_recommendations"
        
        return updated
    
    def detect_intent(self, message):
        """Detect user intent from message with improved patterns and frustration detection."""
        message_lower = message.lower()
        
        # Check for frustration signals first
        frustration_patterns = [
            r'\b(ya te dije|te lo dije|como te dije|repetir|otra vez|no entiendes|no comprendes)\b',
            r'\b(habla.*humano|más humano|sin contexto|no robot|natural|conversar)\b',
            r'\b(estás mal|error|incorrecto|equivocado|qué dices)\b',
        ]
        
        for pattern in frustration_patterns:
            if re.search(pattern, message_lower):
                self.frustration_level += 1
                logger.info(f"Frustration level increased to {self.frustration_level}")
                # If the frustration is directly about communication style
                if re.search(r'\b(habla.*humano|más humano|sin contexto|no robot|natural|conversar)\b', message_lower):
                    return "improve_communication"
                break
        
        # Check for greetings
        if re.search(r'\b(hola|buenos dias|buenas tardes|buenas noches|saludos|hello|hi)\b', message_lower):
            return "greeting"
            
        # Check for information requests about specific topics
        topics = {
            "historia": [r'\b(historia|pasado|origen|fundación|colonial|colonización)\b'],
            "geografia": [r'\b(geografía|geografia|ubicación|ubicacion|isla|territorio|mapa)\b'],
            "cultura": [r'\b(cultura|tradición|tradicion|costumbres|cultural|identidad)\b'],
            "idioma": [r'\b(idioma|lengua|hablan|dialecto|papiamento|idiomas)\b'],
            "moneda": [r'\b(moneda|dinero|cambio|divisa|florín|dolar|euro|efectivo)\b'],
            "clima": [r'\b(clima|tiempo|temperatura|lluvia|soleado|calor|humedad|temporada)\b'],
            "playas": [r'\b(playas|playa|arena|mar|nadar|snorkel|buceo|caribe)\b'],
            "turismo": [r'\b(turismo|visitar|atracciones|lugares|sitios|ver|visita)\b'],
            "transporte": [r'\b(transporte|mover|moverse|auto|coche|alquilar|taxi|bus)\b']
        }
        
        # Check for topic-specific queries
        for topic, patterns in topics.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    self.session["current_topic"] = topic
                    return "topic_query"
        
        # Check for general information request
        if re.search(r'\b(información|informacion|info|datos|dime|cuéntame|cuentame|explicar|explica)\b', message_lower):
            return "information_request"
            
        # Check for request for alternatives
        if re.search(r'\b(otros|otras|alternativas|diferentes|más|mas|otro|otra|tipo|explorar)\b', message_lower):
            return "alternatives"
            
        # Check for thanks - could be closing the conversation
        if re.search(r'\b(gracias|agradezco|thanks|thank you)\b', message_lower):
            return "thanks"
            
        # Check for help
        if re.search(r'\b(ayuda|ayúdame|help|qué puedes hacer|que puedes hacer|cómo funciona|como funciona)\b', message_lower):
            return "help"
            
        # Check for negative responses
        if re.search(r'^\s*(no|nope|negativo|ni|nah|para nada)\s*$', message_lower):
            return "negative"
            
        # Check for positive confirmation
        if re.search(r'^\s*(sí|si|claro|por supuesto|afirmativo|exacto|correcto|ok|okay)\s*$', message_lower):
            return "confirm"
            
        # Check for itinerary/planning requests - broader pattern
        if re.search(r'\b(itinerario|itineriari|agenda|plan|planificación|planificacion|programa|días|dias|organizar|organización|cronograma|recomendación|recomendaciones)\b', message_lower):
            return "itinerary"
        
        # Check for booking information requests - broader pattern
        if re.search(r'\b(reservas?|reservación|reservacion|reservar|booking|reserve|hospedaje|alojamiento|hotel|vuelo|transfer|contacto|teléfono|telefono|dirección|direccion|costo)\b', message_lower):
            return "booking_info"
            
        # Specific question mode
        if re.search(r'^(?:qué|que|cómo|como|dónde|donde|cuál|cual|cuánto|cuanto|cuándo|cuando).*\?$', message_lower):
            return "specific_question"
        
        # Default to unknown
        return "unknown"
    
    def get_greeting(self):
        """Get appropriate greeting based on time of day and conversation state."""
        # Time-based greeting
        if self.current_hour < 12:
            time_greeting = "¡Buenos días"
        elif self.current_hour < 18:
            time_greeting = "¡Buenas tardes"
        else:
            time_greeting = "¡Buenas noches"
        
        # Add name if known
        if self.user_name:
            time_greeting += f", {self.user_name}"
        
        time_greeting += "!"
        
        # Different greetings depending on familiarity
        if self.message_history:  # Not the first message
            greetings = [
                f"{time_greeting} ¿En qué puedo ayudarte ahora con tu viaje a Curaçao?",
                f"¡Hola de nuevo! ¿Seguimos planificando tu viaje a Curaçao?",
                f"{time_greeting} ¿Continuamos con los planes para tu viaje?"
            ]
        else:  # First message
            greetings = [
                f"{time_greeting} Soy tu asistente para planear tu viaje a Curaçao. Puedo recomendarte lugares, crear un itinerario y ayudarte con información para reservas. ¿Qué tipo de viaje estás planeando?",
                f"{time_greeting} Estoy aquí para ayudarte a planificar un viaje increíble a Curaçao. Cuéntame, ¿viajas solo, en pareja o en familia? ¿Qué te gustaría experimentar en la isla?",
                f"{time_greeting} Soy tu asistente para Curaçao. Para empezar, ¿puedes contarme por cuántos días viajarás, con quién y qué tipo de experiencias buscas?"
            ]
        
        return random.choice(greetings)
    
    def get_confirmation_message(self):
        """Generate a message to confirm gathered preferences."""
        if not self.preferences["interests"] and not self.preferences["duration"] and not self.preferences["budget"]:
            return "Para empezar a planificar tu viaje a Curaçao, ¿puedes contarme qué tipo de experiencias te interesan? Por ejemplo, playas, cultura, gastronomía..."
        
        # Map interest codes to readable names
        interest_names = {
            "cultural": "cultura e historia",
            "natural": "playas y naturaleza",
            "family": "actividades familiares",
            "gastronomy": "gastronomía",
            "nightlife": "vida nocturna"
        }
        
        # Parts of the confirmation message
        parts = []
        
        # Format interests in a readable way
        if self.preferences["interests"]:
            readable_interests = [interest_names.get(interest, interest) for interest in self.preferences["interests"]]
            if len(readable_interests) == 1:
                interests_text = f"te interesa {readable_interests[0]}"
            elif len(readable_interests) == 2:
                interests_text = f"te interesan {readable_interests[0]} y {readable_interests[1]}"
            else:
                interests_text = f"te interesan {', '.join(readable_interests[:-1])} y {readable_interests[-1]}"
            parts.append(interests_text)
        
        # Add duration if available
        if self.preferences["duration"]:
            days_text = "día" if self.preferences["duration"] == 1 else "días"
            parts.append(f"planeas estar {int(self.preferences['duration'])} {days_text}")
        
        # Add budget if available
        if self.preferences["budget"]:
            parts.append(f"cuentas con un presupuesto de ${int(self.preferences['budget'])}")
        
        # Add family information if available
        if self.preferences["family_size"]:
            if isinstance(self.preferences["family_size"], int):
                family_text = f"viajarás con tu familia de {self.preferences['family_size']} personas"
            else:
                family_text = "viajarás en familia"
            parts.append(family_text)
        
        # Add children information if available
        if self.preferences["has_children"]:
            if self.preferences["children_ages"]:
                ages = [str(age) for age in self.preferences["children_ages"]]
                if len(ages) == 1:
                    children_text = f"con un niño de {ages[0]} años"
                else:
                    children_text = f"con niños de {', '.join(ages[:-1])} y {ages[-1]} años"
                parts.append(children_text)
            else:
                parts.append("con niños")
        
        # Construct the confirmation message
        if parts:
            confirmation = "He notado que " + " y ".join(parts) + ". ¿Es correcta esta información?"
            return confirmation
        else:
            # If somehow we have no parts (shouldn't happen given the initial check)
            return "¿Puedes darme más detalles sobre tus planes para Curaçao? Me ayudará a darte mejores recomendaciones."
    
    def get_missing_info_question(self):
        """Get question for missing information in a conversational way."""
        # If we don't have interests yet
        if not self.preferences["interests"]:
            questions = [
                "Para empezar, ¿qué te gustaría experimentar en Curaçao? ¿Playas, cultura, gastronomía, vida nocturna?",
                "¿Qué tipo de actividades te interesarían más en tu viaje? ¿Disfrutar de las playas, conocer la cultura local, probar la gastronomía...?",
                "Curaçao tiene mucho que ofrecer. ¿Qué te atrae más de la isla? ¿Sus playas paradisíacas, su rica cultura, su deliciosa comida?"
            ]
            return random.choice(questions)
        
        # If we don't have confirmation on interests but have them
        if self.preferences["interests"] and "interests" in self.confirmed_preferences and not self.confirmed_preferences["interests"]:
            return self.get_confirmation_message()
        
        # If we don't know about family composition
        if not self.preferences["family_size"] and "family" in self.preferences["interests"]:
            questions = [
                "Mencionaste actividades familiares. ¿Cuántas personas viajarán y hay niños en el grupo? Esto me ayudará a recomendar actividades adecuadas.",
                "Para las actividades familiares, ¿podrías decirme cuántas personas son y si viajan con niños? Así puedo ajustar mejor las recomendaciones.",
                "¿Me podrías contar más sobre tu grupo familiar? ¿Cuántas personas son y hay niños? Esto es importante para sugerirte las mejores actividades."
            ]
            return random.choice(questions)
        
        # If we don't have duration
        if not self.preferences["duration"]:
            questions = [
                "¿Por cuántos días planeas visitar Curaçao? Esto me ayudará a organizar mejor las recomendaciones.",
                "¿Cuánto tiempo estarás en Curaçao? Así podré sugerirte un itinerario adecuado a tu estancia.",
                "Para crear el plan ideal, ¿cuántos días durará tu visita a Curaçao?"
            ]
            return random.choice(questions)
        
        # If we don't have budget
        if not self.preferences["budget"]:
            days = int(self.preferences["duration"])
            questions = [
                f"¿Con qué presupuesto aproximado cuentas para tus {days} días en Curaçao? Esto me ayudará a recomendarte opciones adecuadas.",
                f"Para afinar las recomendaciones, ¿podrías indicarme tu presupuesto para los {days} días? Así evitaremos opciones fuera de rango.",
                f"¿Cuál es tu presupuesto aproximado para tu estancia de {days} días? Me permitirá sugerirte experiencias que se ajusten a él."
            ]
            return random.choice(questions)
        
        # If we have all basic info but not all is confirmed
        if not all(self.confirmed_preferences.values()):
            return self.get_confirmation_message()
        
        # If we have all the basic info, ask if they want specific recommendations
        return "Gracias por la información. ¿Te gustaría que te sugiera algunas actividades específicas o prefieres un itinerario completo para tu viaje?"
    
    async def generate_recommendations(self, category=None):
        """Generate recommendations based on user preferences, using the Excel server when possible."""
        # Try to get recommendations from Excel server
        filters = self.preferences.copy()
        if category:
            filters["category"] = category
        
        recommendations = await self.query_excel(filters)
        
        # If we got recommendations from the server, format them
        if recommendations and len(recommendations) > 0:
            return self.format_recommendations(recommendations, category)
        
        # Fallback: Generate basic recommendations based on interests
        # Note: This should rarely happen if the Excel server is working
        basic_response = "Basado en tus preferencias, te recomendaría:"
        
        if "family" in self.preferences["interests"]:
            basic_response += "\n\n• Visitar el Curaçao Sea Aquarium - excelente para familias con niños"
        
        if "natural" in self.preferences["interests"]:
            basic_response += "\n\n• Pasar tiempo en Playa Kenepa (Grote Knip) - una de las playas más hermosas"
            basic_response += "\n\n• Explorar el Parque Nacional Christoffel - el parque natural más grande de la isla"
        
        if "cultural" in self.preferences["interests"]:
            basic_response += "\n\n• Recorrer el centro histórico de Willemstad - Patrimonio de la UNESCO"
            basic_response += "\n\n• Visitar la Sinagoga Mikvé Israel-Emanuel - la más antigua del hemisferio occidental"
        
        basic_response += "\n\n¿Te gustaría más detalles sobre alguno de estos lugares o prefieres que te genere un itinerario completo?"
        
        return basic_response
    
    def format_recommendations(self, recommendations, category=None):
        """Format recommendations received from Excel server or fallback."""
        if not recommendations or len(recommendations) == 0:
            return "Lo siento, no encontré recomendaciones que se ajusten a tus preferencias. ¿Podrías darme más detalles sobre lo que buscas?"
        
        # Limit to 3 recommendations for better readability
        recommendations = recommendations[:3]
        
        # Map categories to readable names
        category_names = {
            "cultural": "cultural",
            "natural": "naturaleza y playas",
            "family": "familiar",
            "gastronomy": "gastronómica",
            "nightlife": "vida nocturna",
            None: "personalizada"
        }
        
        category_display = category_names.get(category, "personalizada")
        
        response = f"Basado en tu búsqueda {category_display}, te recomiendo estas opciones en Curaçao:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. **{rec.get('name', 'Lugar recomendado')}**: {rec.get('description', '')}\n"
            
            if "rating" in rec and "cost" in rec:
                response += f"   Rating: {rec.get('rating', '')}★ | Costo aproximado: {rec.get('cost', '')}\n"
            
            if "details" in rec:
                response += f"   *{rec.get('details', '')}*\n"
            
            # Add contact information if available
            contact_parts = []
            if "phone" in rec and rec["phone"]:
                contact_parts.append(f"📞 {rec['phone']}")
            if "website" in rec and rec["website"]:
                contact_parts.append(f"🌐 {rec['website']}")
            
            if contact_parts:
                response += f"   {' | '.join(contact_parts)}\n"
            
            # Add location if available
            if "location" in rec and rec["location"]:
                response += f"   📍 Ubicación: {rec['location']}\n"
            
            # Add other helpful info if available
            info_parts = []
            if "tips" in rec and rec["tips"]:
                info_parts.append(f"💡 Consejo: {rec['tips']}")
            if "hours" in rec and rec["hours"]:
                info_parts.append(f"🕒 Horario: {rec['hours']}")
            
            if info_parts:
                response += f"   {' | '.join(info_parts)}\n"
            
            response += "\n"
        
        # Add next steps suggestion
        if not self.itinerary_offered and self.preferences["duration"]:
            days = int(self.preferences["duration"])
            self.itinerary_offered = True
            response += f"¿Te gustaría que te prepare un itinerario completo para tus {days} días en Curaçao? También puedo darte información para hacer reservas."
        elif not self.booking_info_offered:
            response += "¿Te gustaría información sobre cómo hacer reservas o prefieres explorar otras opciones?"
        else:
            response += "¿Alguna de estas opciones te interesa particularmente? ¿O prefieres ver otras alternativas?"
        
        return response
    
    async def generate_itinerary(self):
        """Generate a day-by-day itinerary using the Orchestrator server."""
        endpoint = f"{self.base_url}:{self.server_ports['Orchestrator']}/itinerary"
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "preferences": self.preferences,
                    "context": self.session
                }
                
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("itinerary", "No pude generar un itinerario.")
                    else:
                        logger.error(f"Itinerary generation failed with status: {response.status}")
                        return self.generate_fallback_itinerary()
        except Exception as e:
            logger.error(f"Error generating itinerary: {e}")
            return self.generate_fallback_itinerary()
    
    def generate_fallback_itinerary(self):
        """Generate a basic itinerary when the Orchestrator server is unavailable."""
        if not self.preferences["duration"]:
            return "Para crear un itinerario, necesito saber cuántos días planeas estar en Curaçao. ¿Puedes decírmelo?"
        
        duration = int(self.preferences["duration"])
        
        # Mark itinerary as offered
        self.itinerary_offered = True
        
        # Basic activities based on interests
        activities = {
            "cultural": [
                {"name": "Centro histórico de Willemstad", "duration": "3-4 horas", 
                 "details": "Recorrido por las coloridas calles del Patrimonio UNESCO"},
                {"name": "Sinagoga Mikvé Israel-Emanuel", "duration": "1-2 horas",
                 "details": "La sinagoga más antigua del hemisferio occidental con suelo de arena"},
                {"name": "Museo Kura Hulanda", "duration": "2 horas",
                 "details": "Museo sobre la historia de la esclavitud y la diáspora africana"}
            ],
            "natural": [
                {"name": "Playa Kenepa (Grote Knip)", "duration": "Medio día", 
                 "details": "Una de las playas más espectaculares con aguas turquesa"},
                {"name": "Parque Nacional Christoffel", "duration": "3-4 horas",
                 "details": "El parque más grande de la isla con senderos y vistas panorámicas"},
                {"name": "Shete Boka National Park", "duration": "2-3 horas",
                 "details": "Formaciones rocosas donde las olas chocan contra los acantilados"}
            ],
            "family": [
                {"name": "Curaçao Sea Aquarium", "duration": "2-3 horas", 
                 "details": "Exhibiciones marinas interactivas perfectas para niños"},
                {"name": "Playa Jan Thiel", "duration": "Medio día",
                 "details": "Playa familiar con aguas tranquilas e instalaciones completas"},
                {"name": "Curacao Ostrich Farm", "duration": "2 horas",
                 "details": "Granja de avestruces con tours guiados y actividades para niños"}
            ],
            "gastronomy": [
                {"name": "Plasa Bieu (Mercado Viejo)", "duration": "1-2 horas", 
                 "details": "Mercado gastronómico local con platos tradicionales"},
                {"name": "Tour de licor de Curaçao", "duration": "2 horas",
                 "details": "Visita a Landhuis Chobolobo para conocer el famoso licor azul"},
                {"name": "Cena en restaurantes locales", "duration": "2 horas",
                 "details": "Probar platos típicos como Keshi Yená o Kabritu Stoba"}
            ]
        }
        
        # Select activities based on user interests
        selected_activities = []
        for interest in self.preferences["interests"]:
            if interest in activities:
                selected_activities.extend(activities[interest])
        
        # If no interests or not enough activities, add some natural ones (everyone loves beaches)
        if len(selected_activities) < duration * 2:
            for activity in activities["natural"]:
                if activity not in selected_activities:
                    selected_activities.append(activity)
        
        # Still not enough? Add cultural activities
        if len(selected_activities) < duration * 2:
            for activity in activities["cultural"]:
                if activity not in selected_activities:
                    selected_activities.append(activity)
        
        # Format the itinerary
        itinerary = f"Aquí tienes un itinerario sugerido para tus {duration} días en Curaçao, basado en tus preferencias:\n\n"
        
        # Distribute activities across days
        for day in range(1, duration + 1):
            itinerary += f"**Día {day}:**\n\n"
            
            # Morning activity
            morning_idx = (day - 1) % len(selected_activities)
            morning = selected_activities[morning_idx]
            itinerary += f"• **Mañana:** {morning['name']} ({morning['duration']})\n"
            itinerary += f"  *{morning['details']}*\n\n"
            
            # Afternoon activity
            afternoon_idx = (day - 1 + len(selected_activities)//2) % len(selected_activities)
            afternoon = selected_activities[afternoon_idx]
            itinerary += f"• **Tarde:** {afternoon['name']} ({afternoon['duration']})\n"
            itinerary += f"  *{afternoon['details']}*\n\n"
            
            # Evening suggestion
            if "family" in self.preferences["interests"]:
                itinerary += f"• **Noche:** Cena familiar en el área de {['Mambo Beach', 'Punda', 'Jan Thiel', 'Otrobanda'][day % 4]}\n\n"
            elif "gastronomy" in self.preferences["interests"]:
                restaurants = ["Kome", "Gouverneur de Rouville", "Jaanchie's Restaurant", "Blessing by Selena"]
                itinerary += f"• **Noche:** Cena en {restaurants[day % 4]}, reconocido por su excelente cocina local\n\n"
            else:
                activities = ["Paseo por la costa al atardecer", "Cena en Willemstad", "Explorar Pietermaai District", "Relax en el hotel"]
                itinerary += f"• **Noche:** {activities[day % 4]}\n\n"
        
        # Add tips
        itinerary += "**Consejos para tu viaje:**\n\n"
        itinerary += "• Alquila un coche para moverte con libertad por la isla\n"
        itinerary += "• Lleva siempre protector solar, incluso en días nublados\n"
        itinerary += "• Las playas del norte son menos concurridas pero también más salvajes\n"
        itinerary += "• El dólar americano es ampliamente aceptado, pero conviene llevar algo de florines locales\n\n"
        
        # Add prompt for booking info
        if not self.booking_info_offered:
            itinerary += "¿Te gustaría información para hacer reservas de alojamiento, restaurantes o excursiones? ¿O prefieres ajustar el itinerario?"
        else:
            itinerary += "Este itinerario es flexible y puede adaptarse según tus preferencias. ¿Quieres hacer algún ajuste o tienes preguntas sobre alguna actividad?"
        
        return itinerary
    
    async def get_booking_information(self):
        """Get booking information using the RAG server."""
        # Mark booking info as offered
        self.booking_info_offered = True
        
        try:
            booking_info = await self.query_rag("información de reservas en Curaçao", "booking")
            
            if booking_info and len(booking_info) > 100:  # Ensure we got a meaningful response
                return booking_info
            else:
                # Fallback to basic booking info
                return self.get_fallback_booking_info()
        except Exception as e:
            logger.error(f"Error getting booking information: {e}")
            return self.get_fallback_booking_info()
    
    def get_fallback_booking_info(self):
        """Provide fallback booking information when RAG server is unavailable."""
        booking_info = """
**Información para Reservas en Curaçao:**

**Alojamiento:**
• Booking.com, Airbnb y Hotels.com ofrecen buenas opciones para todos los presupuestos
• Los mejores hoteles en el área de Willemstad incluyen Renaissance Curaçao, Avila Beach Hotel y Curaçao Marriott
• Para familias, los apartamentos y resorts todo incluido son excelentes opciones

**Tours y Actividades:**
• Para excursiones, busca en Viator o GetYourGuide con anticipación
• Los tours de buceo y snorkel es mejor reservarlos con 2-3 días de anticipación
• Las excursiones populares como Klein Curaçao suelen llenarse, reserva con una semana de antelación

**Restaurantes:**
• Los restaurantes populares como Kome o Gouverneur de Rouville recomiendan reservar, especialmente para cenas
• Para restaurantes locales como Jaanchie's, una llamada el mismo día suele ser suficiente

**Transporte:**
• Las empresas de alquiler de coches como Budget, Avis y D&D Car Rental operan en la isla
• Es recomendable reservar con anticipación, especialmente en temporada alta

**Consejos generales:**
• La temporada alta (diciembre-abril) requiere reservas con mayor anticipación
• Muchos lugares aceptan reservas por teléfono, email o a través de sus páginas web
• El dólar americano es ampliamente aceptado, aunque la moneda local es el florín antillano (ANG)

¿Hay algún tipo específico de reserva con la que necesites ayuda?
"""
        return booking_info
    
    def handle_communication_improvement(self):
        """Generate a response when user asks for more human-like communication."""
        self.frustration_level = 0  # Reset frustration since we're addressing it
        
        responses = [
            "Tienes razón, me disculpo por hablar de forma poco natural. Intentaré ser más conversacional. Cuéntame qué estás buscando para tu viaje a Curaçao y te ayudaré de forma más directa y personal.",
            
            "Perdona por sonar tan formal. Vamos a empezar de nuevo. Hablemos de tu viaje a Curaçao de manera más relajada. ¿Qué te gustaría hacer allí? ¿Buscas playas, cultura, comida local...?",
            
            "Disculpa, a veces sueno demasiado robótico. Tienes toda la razón. Intentemos de nuevo: ¿qué es lo que más te emociona de visitar Curaçao? Así puedo ayudarte mejor con recomendaciones que realmente te interesen."
        ]
        
        # Append information about preferences if we have any
        response = random.choice(responses)
        
        # If we have some preferences, summarize them conversacionalmente
        if self.preferences["interests"] or self.preferences["duration"] or self.preferences["budget"]:
            pref_parts = []
            
            if self.preferences["interests"]:
                interest_names = {
                    "cultural": "la cultura",
                    "natural": "las playas y naturaleza",
                    "family": "planes familiares",
                    "gastronomy": "la gastronomía",
                    "nightlife": "la vida nocturna"
                }
                readable_interests = [interest_names.get(i, i) for i in self.preferences["interests"]]
                if len(readable_interests) == 1:
                    pref_parts.append(f"te interesa {readable_interests[0]}")
                else:
                    pref_parts.append(f"te interesan {' y '.join(readable_interests)}")
            
            if self.preferences["duration"]:
                days = int(self.preferences["duration"])
                day_text = "día" if days == 1 else "días"
                pref_parts.append(f"estarás {days} {day_text}")
            
            if self.preferences["budget"]:
                pref_parts.append(f"tienes un presupuesto de ${int(self.preferences['budget'])}")
            
            if pref_parts:
                response += f" Por lo que me has contado, {' y '.join(pref_parts)}. ¿Hay algo específico que quieras saber o prefieres un itinerario completo?"
        
        return response
    
    async def handle_frustration(self, message, intent):
        """Handle user frustration by improving responses and recovering context."""
        
        # Reset frustration level
        old_level = self.frustration_level
        self.frustration_level = 0
        
        # If the user is specifically asking for more human-like communication
        if intent == "improve_communication":
            return self.handle_communication_improvement()
        
        # If user mentioned repeating information, try to identify what we missed
        if re.search(r'\b(ya te dije|te lo dije|repetir|otra vez)\b', message.lower()):
            for keyword in ["días", "dia", "días", "presupuesto", "dinero", "gastar", "interesa", "gusta"]:
                if keyword in message.lower():
                    return f"Disculpa por hacerte repetir. He anotado la información sobre {keyword}. ¿Hay algo más que quieras añadir o prefieres que continuemos con las recomendaciones?"
        
        # General recovery based on frustration level
        if old_level >= 2:
            # High frustration, offer a fresh start
            return "Parece que no estoy siendo de mucha ayuda. Empecemos de nuevo. ¿Podrías decirme qué tipo de experiencias estás buscando en Curaçao, por cuántos días y con qué presupuesto aproximado? Intentaré ser más directo y útil."
        else:
            # Medium frustration, be more conciliatory
            return "Disculpa si no estoy entendiendo bien tus necesidades. ¿Podrías decirme directamente qué tipo de información necesitas sobre Curaçao? Puedo ayudarte con recomendaciones de lugares, itinerarios personalizados o información para reservas."
    
    async def generate_response(self, message):
        """Generate a response based on the current state and message, using server integration when possible."""
        # Extract preferences from the message
        updated_prefs = self.extract_preferences(message)
        logger.info(f"Updated preferences: {updated_prefs}")
        
        # Detect intent
        intent = self.detect_intent(message)
        logger.info(f"Detected intent: {intent}")
        
        # Check for frustration and handle it specially if needed
        if self.frustration_level >= 1:
            return await self.handle_frustration(message, intent)
        
        # Try to get an orchestrated response for most intents
        try:
            orchestrated_response = await self.get_orchestrated_response(message, intent)
            if orchestrated_response and len(orchestrated_response) > 20:  # Ensure it's a meaningful response
                return orchestrated_response
        except Exception as e:
            logger.error(f"Failed to get orchestrated response: {e}")
            # Continue with fallback response generation
        
        # Handle specific intents that require special processing
        
        # Greetings
        if intent == "greeting":
            return self.get_greeting()
        
        # Improve communication style
        if intent == "improve_communication":
            return self.handle_communication_improvement()
        
        # Topic-specific queries
        if intent == "topic_query" and "current_topic" in self.session:
            topic_info = await self.query_rag(message, self.session["current_topic"])
            if topic_info:
                return topic_info
        
        # General information request
        if intent == "information_request":
            general_info = await self.query_rag(message)
            if general_info:
                return general_info
            # Fallback to basic response if RAG fails
            return "Curaçao es una hermosa isla en el Caribe con playas de aguas cristalinas, rica historia colonial y una cultura diversa. ¿Hay algo específico sobre la isla que te gustaría conocer?"
        
        # Confirmation of preferences
        if intent == "confirm":
            # Mark all unconfirmed preferences as confirmed
            for key in self.confirmed_preferences:
                self.confirmed_preferences[key] = True
            
            # Move to next stage if all basics are provided
            if self.preferences["interests"] and self.preferences["duration"] and self.preferences["budget"]:
                self.conversation_stage = "providing_recommendations"
                return "¡Perfecto! Con esta información puedo darte recomendaciones personalizadas. ¿Prefieres que te sugiera lugares específicos o quieres un itinerario completo para tu estancia?"
            else:
                # Ask for next missing info
                return self.get_missing_info_question()
        
        # Itinerary request
        if intent == "itinerary":
            return await self.generate_itinerary()
        
        # Booking information request
        if intent == "booking_info":
            return await self.get_booking_information()
        
        # Request for alternatives
        if intent == "alternatives":
            # If we have interests, rotate through them for recommendations
            if self.preferences["interests"]:
                category = self.preferences["interests"][random.randint(0, len(self.preferences["interests"])-1)]
                return await self.generate_recommendations(category)
            else:
                return await self.generate_recommendations()
        
        # Thanks - provide closure but also offer more help
        if intent == "thanks":
            return "¡De nada! Ha sido un placer ayudarte con tu planificación para Curaçao. Si necesitas más información sobre algún aspecto específico o ayuda con reservas, no dudes en preguntarme. ¡Que tengas un viaje increíble!"
        
        # Help request
        if intent == "help":
            return "Puedo ayudarte a planificar tu viaje a Curaçao de varias formas:\n\n• Recomendarte lugares según tus intereses (playas, cultura, gastronomía...)\n• Crear un itinerario personalizado\n• Darte información para hacer reservas\n• Resolver dudas específicas sobre la isla\n\nCuéntame qué tipo de experiencia buscas y estaré encantado de ayudarte."
        
        # Default processing based on conversation stage
        if self.conversation_stage == "greeting":
            return self.get_greeting()
            
        elif self.conversation_stage == "collecting_preferences":
            return self.get_missing_info_question()
            
        elif self.conversation_stage == "providing_recommendations":
            return await self.generate_recommendations()
        
        # Fallback for unknown/unhandled intents
        return "¿Hay algo específico que te gustaría saber sobre Curaçao? Puedo recomendarte lugares para visitar, crear un itinerario o darte información para reservas."
    
    async def process_message(self, message):
        """Process user message and generate response."""
        # Add to history
        self.message_history.append({
            "role": "user",
            "content": message
        })
        
        # Generate response
        response = await self.generate_response(message)
        
        # Add response to history
        self.message_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def close_servers(self):
        """Close all server processes."""
        print(f"{Colors.BLUE}Cerrando servidores MCP...{Colors.RESET}")
        
        for name, process in self.server_processes.items():
            try:
                print(f"{Colors.DIM}Cerrando servidor {name}...{Colors.RESET}")
                process.terminate()
                time.sleep(0.5)
                
                if process.poll() is None:
                    process.kill()
                    time.sleep(0.5)
                
                logger.info(f"Server {name} terminated")
            except Exception as e:
                logger.error(f"Error terminating {name} server: {e}")
        
        print(f"{Colors.GREEN}Todos los servidores cerrados.{Colors.RESET}")
    
    def print_welcome(self):
        """Print welcome message."""
        title = "Asistente Turístico de Curaçao"
        
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
        """Run the assistant."""
        # Start servers in background
        servers_started = await self.start_servers()
        if not servers_started:
            print(f"{Colors.YELLOW}Servidores MCP no disponibles. Funcionando en modo local.{Colors.RESET}")
        
        # Welcome message
        self.print_welcome()
        
        # Register signal handler for clean shutdown
        def signal_handler(sig, frame):
            print("\n\nCerrando asistente...")
            self.close_servers()
            print(f"{Colors.GREEN}¡Gracias por usar el asistente! ¡Hasta pronto!{Colors.RESET}")
            sys.exit(0)
        
        # Register handler
        signal.signal(signal.SIGINT, signal_handler)
        
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
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n{Colors.RED}Error inesperado: {str(e)}{Colors.RESET}\n")
        finally:
            # Close servers
            self.close_servers()

def main():
    """Main function."""
    assistant = CuracaoAssistant()
    asyncio.run(assistant.run())

if __name__ == "__main__":
    main()