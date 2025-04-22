# app/conversation/manager.py
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime, timedelta
from core.nlu.processor import NLUProcessor
from utils.logger import get_logger

logger = get_logger(__name__)

class ConversationManager:
    """
    Gestiona conversaciones multi-turno para recopilar información
    gradualmente y mejorar las recomendaciones.
    """
    def __init__(self):
        """Inicializa el gestor de conversaciones"""
        self.sessions = {}
        self.session_expiry = timedelta(minutes=30)
        self.nlu_processor = NLUProcessor()
        
        # Campos requeridos para una recomendación completa
        self.required_fields = ['budget', 'duration', 'locations', 'interests']
        
        # Campos opcionales que mejoran la recomendación
        self.optional_fields = [
            'group_size', 'with_children', 'accommodation_type', 
            'transportation', 'special_occasion'
        ]
        
    def create_session(self) -> str:
        """
        Crea una nueva sesión de conversación
        
        Returns:
            str: ID de sesión
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "state": {
                "preferences": {},
                "conversation_history": [],
                "collected_fields": set(),
                "missing_fields": set(self.required_fields),
                "confirmed": False,
                "ready_for_recommendations": False
            }
        }
        logger.info(f"Sesión creada: {session_id}")
        return session_id
    
    def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario en una sesión
        
        Args:
            session_id (str): ID de sesión
            message (str): Mensaje del usuario
            
        Returns:
            Dict[str, Any]: Respuesta procesada
        """
        try:
            # Verificar si la sesión existe
            if session_id not in self.sessions:
                logger.warning(f"Sesión {session_id} no encontrada. Creando nueva sesión.")
                session_id = self.create_session()
                
            # Actualizar timestamp de última actividad
            self.sessions[session_id]["last_active"] = datetime.now()
            
            # Obtener estado actual
            state = self.sessions[session_id]["state"]
            
            # Añadir mensaje a historial
            self._add_to_history(session_id, "user", message)
            
            # Procesar con NLU
            nlu_result = self.nlu_processor.process_query(message)
            
            # Actualizar preferencias
            self._update_preferences(session_id, nlu_result["preferences"])
            
            # Verificar si estamos listos para recomendaciones
            state = self.sessions[session_id]["state"]  # Obtener estado actualizado
            if self._check_if_ready(state):
                state["ready_for_recommendations"] = True
                
                # Generar mensaje de confirmación
                response_message = self._generate_confirmation(state)
                self._add_to_history(session_id, "system", response_message)
                
                return {
                    "session_id": session_id,
                    "message": response_message,
                    "preferences": state["preferences"],
                    "ready_for_recommendations": True,
                    "needs_confirmation": not state["confirmed"]
                }
            else:
                # Generar siguiente pregunta
                next_question = self._generate_next_question(state)
                self._add_to_history(session_id, "system", next_question)
                
                return {
                    "session_id": session_id,
                    "message": next_question,
                    "preferences": state["preferences"],
                    "ready_for_recommendations": False,
                    "missing_fields": list(state["missing_fields"])
                }
                
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
            error_message = "Lo siento, ha ocurrido un error. ¿Podría reformular su consulta?"
            
            # Intentar añadir al historial
            try:
                self._add_to_history(session_id, "system", error_message)
            except:
                pass
                
            return {
                "session_id": session_id,
                "message": error_message,
                "error": str(e),
                "ready_for_recommendations": False
            }
    
    def _update_preferences(self, session_id: str, new_preferences: Dict[str, Any]) -> None:
        """
        Actualiza las preferencias en la sesión con nueva información
        
        Args:
            session_id (str): ID de sesión
            new_preferences (Dict[str, Any]): Nuevas preferencias a fusionar
        """
        state = self.sessions[session_id]["state"]
        existing = state["preferences"]
        
        # Fusionar preferencias
        for key, value in new_preferences.items():
            # Solo actualizar si hay un valor no vacío
            if value:
                existing[key] = value
                
                # Actualizar campos recolectados
                if key in state["missing_fields"]:
                    state["missing_fields"].remove(key)
                    state["collected_fields"].add(key)
    
    def _check_if_ready(self, state: Dict[str, Any]) -> bool:
        """
        Verifica si tenemos suficiente información para recomendaciones
        
        Args:
            state (Dict[str, Any]): Estado actual de la sesión
            
        Returns:
            bool: True si estamos listos para recomendaciones
        """
        # Si ya se confirmó, estamos listos
        if state.get("confirmed", False):
            return True
            
        # Verificar campos requeridos
        # Consideramos listos si tenemos al menos 2 campos recolectados
        # Esto es para no hacer el proceso demasiado largo
        collected = state["collected_fields"]
        return len(collected) >= 2
    
    def _generate_next_question(self, state: Dict[str, Any]) -> str:
        """
        Genera la siguiente pregunta basada en el estado actual
        
        Args:
            state (Dict[str, Any]): Estado de la sesión
            
        Returns:
            str: Pregunta generada
        """
        # Prioridad de campos a preguntar
        field_priority = ['budget', 'duration', 'locations', 'interests']
        
        # Buscar el primer campo faltante según prioridad
        for field in field_priority:
            if field in state["missing_fields"]:
                return self._get_question_for_field(field, state["preferences"])
        
        # Si no hay campos prioritarios faltantes, preguntar por opcionales
        for field in self.optional_fields:
            if field not in state["collected_fields"] and field not in state["preferences"]:
                return self._get_question_for_field(field, state["preferences"])
                
        # Si llegamos aquí, preguntar por confirmación
        return self._generate_confirmation(state)
    
    def _get_question_for_field(self, field: str, preferences: Dict[str, Any]) -> str:
        """
        Obtiene una pregunta específica para un campo
        
        Args:
            field (str): Campo a preguntar
            preferences (Dict[str, Any]): Preferencias actuales
            
        Returns:
            str: Pregunta generada
        """
        # Preguntas base para cada campo
        questions = {
            'budget': "¿Cuál es su presupuesto aproximado por día para este viaje?",
            'duration': "¿Por cuántos días planea visitar Curaçao?",
            'locations': "¿Qué áreas o lugares específicos de Curaçao le gustaría visitar?",
            'interests': "¿Qué tipo de actividades o experiencias le interesan más (cultura, playa, gastronomía, aventura)?",
            'group_size': "¿Cuántas personas viajarán en su grupo?",
            'with_children': "¿Viajará con niños o es un viaje solo para adultos?",
            'accommodation_type': "¿Qué tipo de alojamiento prefiere durante su estancia?",
            'transportation': "¿Cómo planea moverse por la isla?",
            'special_occasion': "¿Es este viaje para alguna ocasión especial?"
        }
        
        # Personalizar pregunta según contexto
        if field == 'interests' and preferences.get('with_children'):
            return "¿Qué actividades les gustaría hacer que sean adecuadas para niños?"
            
        if field == 'locations' and preferences.get('interests'):
            interests = ", ".join(preferences["interests"])
            return f"Considerando su interés en {interests}, ¿qué áreas de Curaçao le gustaría visitar?"
            
        if field == 'budget' and preferences.get('group_size'):
            return f"Para un grupo de {preferences['group_size']} personas, ¿cuál es su presupuesto aproximado por día?"
        
        # Pregunta predeterminada
        return questions.get(field, "¿Podría proporcionar más detalles sobre sus preferencias de viaje?")
    
    def _generate_confirmation(self, state: Dict[str, Any]) -> str:
        """
        Genera un mensaje de confirmación con las preferencias actuales
        
        Args:
            state (Dict[str, Any]): Estado actual de la sesión
            
        Returns:
            str: Mensaje de confirmación
        """
        preferences = state["preferences"]
        collected = []
        
        # Formatear cada preferencia recolectada
        if preferences.get('budget'):
            collected.append(f"presupuesto de ${preferences['budget']} por día")
            
        if preferences.get('duration'):
            days = int(preferences['duration'])
            collected.append(f"duración de {days} día{'s' if days != 1 else ''}")
            
        if preferences.get('locations'):
            locations = ", ".join(preferences['locations'])
            collected.append(f"interés en visitar {locations}")
            
        if preferences.get('interests'):
            interests = ", ".join(preferences['interests'])
            collected.append(f"interés en actividades de {interests}")
            
        if preferences.get('group_size'):
            collected.append(f"grupo de {preferences['group_size']} personas")
            
        if preferences.get('with_children'):
            collected.append("viaje familiar con niños")
        
        # Construir mensaje
        if collected:
            preferences_text = ", ".join(collected)
            return f"Entiendo que busca recomendaciones para Curaçao con {preferences_text}. ¿Es correcto? ¿Quisiera añadir algo más?"
        else:
            return "Basado en su consulta, buscaré las mejores recomendaciones para su visita a Curaçao. ¿Hay algo específico que deba considerar?"
    
    def confirm_preferences(self, session_id: str, confirmed: bool) -> Dict[str, Any]:
        """
        Confirma las preferencias para generar recomendaciones
        
        Args:
            session_id (str): ID de sesión
            confirmed (bool): Si las preferencias fueron confirmadas
            
        Returns:
            Dict[str, Any]: Estado actualizado
        """
        try:
            if session_id not in self.sessions:
                raise ValueError(f"Sesión {session_id} no encontrada")
                
            state = self.sessions[session_id]["state"]
            state["confirmed"] = confirmed
            
            # Si se confirmó, marcar como listo para recomendaciones
            if confirmed:
                state["ready_for_recommendations"] = True
                message = "Gracias por confirmar. Generaré las recomendaciones basadas en sus preferencias."
            else:
                # Si no se confirmó, reiniciar el proceso de recolección
                message = "Entendido. Por favor, indique qué aspectos de sus preferencias quisiera cambiar."
            
            # Añadir al historial
            self._add_to_history(session_id, "system", message)
            
            return {
                "session_id": session_id,
                "message": message,
                "preferences": state["preferences"],
                "ready_for_recommendations": state["ready_for_recommendations"]
            }
            
        except Exception as e:
            logger.error(f"Error confirmando preferencias: {e}")
            return {
                "session_id": session_id,
                "message": "Ha ocurrido un error al procesar su confirmación.",
                "error": str(e),
                "ready_for_recommendations": False
            }
    
    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de una sesión
        
        Args:
            session_id (str): ID de sesión
            
        Returns:
            Dict[str, Any]: Estado de la sesión
        """
        if session_id not in self.sessions:
            raise ValueError(f"Sesión {session_id} no encontrada")
            
        # Actualizar timestamp de última actividad
        self.sessions[session_id]["last_active"] = datetime.now()
        
        return self.sessions[session_id]["state"]
    
    def _add_to_history(self, session_id: str, role: str, message: str) -> None:
        """
        Añade un mensaje al historial de conversación
        
        Args:
            session_id (str): ID de sesión
            role (str): Rol del mensaje (user/system)
            message (str): Contenido del mensaje
        """
        if session_id not in self.sessions:
            raise ValueError(f"Sesión {session_id} no encontrada")
            
        state = self.sessions[session_id]["state"]
        
        state["conversation_history"].append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def clear_expired_sessions(self) -> int:
        """
        Limpia sesiones expiradas
        
        Returns:
            int: Número de sesiones limpiadas
        """
        now = datetime.now()
        expired = []
        
        for session_id, session in self.sessions.items():
            if now - session["last_active"] > self.session_expiry:
                expired.append(session_id)
                
        for session_id in expired:
            del self.sessions[session_id]
            
        if expired:
            logger.info(f"Limpiadas {len(expired)} sesiones expiradas")
            
        return len(expired)