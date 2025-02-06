"""
Implementación de cadena de conversación usando LangGraph para gestionar el flujo
de diálogo con el usuario.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import END, StateGraph
from core.rag.agents.tour_agent import TourAgent
from core.rag.agents.food_agent import FoodAgent
from core.rag.agents.activity_agent import ActivityAgent
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

class ConversationState(BaseModel):
    """Modelo para estado de conversación"""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    current_intent: Optional[str] = None
    active_agent: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationChain:
    def __init__(
        self,
        tour_agent: TourAgent,
        food_agent: FoodAgent,
        activity_agent: ActivityAgent,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tour_agent = tour_agent
        self.food_agent = food_agent
        self.activity_agent = activity_agent
        self.config = config or {}
        self.tool_executor = ToolExecutor()
        self._init_graph()

    def _init_graph(self):
        """Inicializa el grafo de estados de conversación"""
        # Definir nodos del grafo
        self.state_graph = StateGraph(nodes=[
            "start",
            "analyze_intent",
            "gather_context",
            "route_to_agent",
            "process_tour",
            "process_food",
            "process_activity",
            "generate_response",
            "update_context",
            "finalize"
        ])

        # Definir transiciones
        self._add_transitions()

        # Configurar funciones de nodo
        self._configure_node_functions()

    def _add_transitions(self):
        """Configura las transiciones entre estados"""
        self.state_graph.add_edge("start", "analyze_intent")
        self.state_graph.add_edge("analyze_intent", "gather_context")
        self.state_graph.add_edge("gather_context", "route_to_agent")
        
        # Routing basado en tipo de intención
        self.state_graph.add_conditional_edges(
            "route_to_agent",
            self._route_to_agent,
            {
                "tour": "process_tour",
                "food": "process_food",
                "activity": "process_activity"
            }
        )

        # Convergencia de procesamiento
        for node in ["process_tour", "process_food", "process_activity"]:
            self.state_graph.add_edge(node, "generate_response")

        self.state_graph.add_edge("generate_response", "update_context")
        self.state_graph.add_edge("update_context", "finalize")
        self.state_graph.add_edge("finalize", END)

    def _configure_node_functions(self):
        """Configura las funciones para cada nodo"""
        self.state_graph.add_node_function(
            "analyze_intent",
            self._analyze_intent
        )
        self.state_graph.add_node_function(
            "gather_context",
            self._gather_context
        )
        self.state_graph.add_node_function(
            "process_tour",
            self._process_tour_request
        )
        self.state_graph.add_node_function(
            "process_food",
            self._process_food_request
        )
        self.state_graph.add_node_function(
            "process_activity",
            self._process_activity_request
        )
        self.state_graph.add_node_function(
            "generate_response",
            self._generate_response
        )
        self.state_graph.add_node_function(
            "update_context",
            self._update_context
        )
        self.state_graph.add_node_function(
            "finalize",
            self._finalize_conversation
        )

    async def process_message(
        self,
        message: str,
        state: Optional[ConversationState] = None
    ) -> ConversationState:
        """
        Procesa un mensaje del usuario
        
        Args:
            message: Mensaje del usuario
            state: Estado actual de la conversación
            
        Returns:
            Nuevo estado de la conversación
        """
        if state is None:
            state = ConversationState(
                metadata={"start_time": datetime.now()}
            )

        # Añadir mensaje a la conversación
        state.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })

        # Ejecutar grafo de estados
        try:
            final_state = await self.state_graph.arun(state.dict())
            return ConversationState(**final_state)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    def _format_response(self, state: Dict[str, Any]) -> str:
        """Formatea la respuesta para el usuario"""
        try:
            intent = state.get("current_intent", "tour")
            recommendations = state.get("recommendations", [])

            if not recommendations:
                return "Lo siento, no he podido encontrar recomendaciones que coincidan con tus preferencias."

            # Formatear respuesta según el tipo de intención
            if intent == "tour":
                return self._format_tour_response(recommendations)
            elif intent == "food":
                return self._format_food_response(recommendations)
            elif intent == "activity":
                return self._format_activity_response(recommendations)
            else:
                return self._format_general_response(recommendations)

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu solicitud."

    def _format_tour_response(self, recommendations: Dict[str, Any]) -> str:
        """Formatea respuesta para recomendaciones de tour"""
        response = []
        days = recommendations.get("days", [])

        response.append("He preparado un itinerario basado en tus preferencias:")

        for day in days:
            response.append(f"\nDía {day['day']}:")
            for activity in day['activities']:
                response.append(f"- {activity['name']}: {activity['description']}")
                if activity.get('time'):
                    response.append(f"  Hora: {activity['time']}")
                if activity.get('cost'):
                    response.append(f"  Costo: ${activity['cost']}")

        return "\n".join(response)

    def _format_food_response(self, recommendations: List[Dict[str, Any]]) -> str:
        """Formatea respuesta para recomendaciones de restaurantes"""
        response = ["Aquí tienes algunas recomendaciones gastronómicas:"]

        for i, rec in enumerate(recommendations, 1):
            response.append(f"\n{i}. {rec['name']}")
            response.append(f"   Cocina: {', '.join(rec['cuisine'])}")
            response.append(f"   Ubicación: {rec['location']}")
            response.append(f"   Precio: {rec['price_range']}")
            if rec.get('specialties'):
                response.append(f"   Especialidades: {', '.join(rec['specialties'])}")

        return "\n".join(response)

    def _format_activity_response(self, recommendations: List[Dict[str, Any]]) -> str:
        """Formatea respuesta para recomendaciones de actividades"""
        response = ["Te recomiendo las siguientes actividades:"]

        for i, rec in enumerate(recommendations, 1):
            response.append(f"\n{i}. {rec['name']}")
            response.append(f"   {rec['description']}")
            response.append(f"   Duración: {rec['duration']} minutos")
            response.append(f"   Costo: ${rec['cost']}")
            if rec.get('difficulty'):
                response.append(f"   Dificultad: {rec['difficulty']}")

        return "\n".join(response)

    def _format_general_response(self, recommendations: List[Dict[str, Any]]) -> str:
        """Formatea respuesta general para recomendaciones"""
        response = ["Aquí tienes algunas recomendaciones:"]

        for i, rec in enumerate(recommendations, 1):
            response.append(f"\n{i}. {rec['name']}")
            if rec.get('description'):
                response.append(f"   {rec['description']}")

        return "\n".join(response)

    async def _update_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Actualiza el contexto de la conversación"""
        try:
            # Actualizar historial de recomendaciones
            if "recommendation_history" not in state:
                state["recommendation_history"] = []

            state["recommendation_history"].append({
                "timestamp": datetime.now(),
                "intent": state["current_intent"],
                "recommendations": state.get("recommendations", [])
            })

            # Limpiar recomendaciones actuales
            state["recommendations"] = []

            # Actualizar metadata
            state["metadata"]["context_update_time"] = datetime.now()

            return state

        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
            return state

    async def _finalize_conversation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finaliza la conversación y limpia el estado"""
        try:
            # Actualizar metadata final
            state["metadata"].update({
                "end_time": datetime.now(),
                "total_messages": len(state["messages"]),
                "status": "completed"
            })

            # Calcular métricas de la conversación
            start_time = state["metadata"].get("start_time")
            if start_time:
                duration = (datetime.now() - start_time).total_seconds()
                state["metadata"]["conversation_duration"] = duration

            return state

        except Exception as e:
            logger.error(f"Error finalizing conversation: {str(e)}")
            state["metadata"]["status"] = "error"
            state["metadata"]["error"] = str(e)
            return state

    def _extract_preferences(self, message: str) -> Dict[str, Any]:
        """Extrae preferencias del mensaje del usuario"""
        preferences = {}

        # Extraer presupuesto
        if "$" in message:
            import re
            budget_match = re.search(r'\$(\d+)', message)
            if budget_match:
                preferences["budget_per_day"] = float(budget_match.group(1))

        # Extraer duración
        duration_match = re.search(r'(\d+)\s*(?:day|días|dia)', message.lower())
        if duration_match:
            preferences["trip_duration"] = int(duration_match.group(1))

        # Extraer intereses (implementación básica)
        interests = []
        interest_keywords = ["culture", "history", "food", "beach", "nature", "adventure"]
        for keyword in interest_keywords:
            if keyword.lower() in message.lower():
                interests.append(keyword)
        if interests:
            preferences["interests"] = interests

        return preferences:
            logger.error(f"Error processing message: {str(e)}")
            # Retornar estado con error
            state.metadata["error"] = str(e)
            state.metadata["status"] = "error"
            return state

    async def _analyze_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la intención del usuario"""
        try:
            message = state["messages"][-1]["content"]

            # Implementar análisis de intención usando LLM/NLP
            # Por ahora, una implementación simple basada en palabras clave
            intent_mapping = {
                "tour": ["tour", "visit", "explore", "see", "guide"],
                "food": ["food", "eat", "restaurant", "dining", "cuisine"],
                "activity": ["activity", "do", "experience", "adventure"]
            }

            message_lower = message.lower()
            detected_intent = "tour"  # default intent
            max_matches = 0

            for intent, keywords in intent_mapping.items():
                matches = sum(1 for keyword in keywords if keyword in message_lower)
                if matches > max_matches:
                    max_matches = matches
                    detected_intent = intent

            state["current_intent"] = detected_intent
            state["metadata"]["intent_analysis_time"] = datetime.now()

            return state

        except Exception as e:
            logger.error(f"Error analyzing intent: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    async def _gather_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Recopila contexto adicional para la solicitud"""
        try:
            message = state["messages"][-1]["content"]
            current_intent = state["current_intent"]

            # Extraer información contextual básica
            context = {
                "timestamp": datetime.now(),
                "intent": current_intent,
                "detected_entities": {}  # Implementar extracción de entidades
            }

            # Actualizar preferencias basadas en el mensaje
            new_preferences = self._extract_preferences(message)
            if new_preferences:
                state["preferences"].update(new_preferences)

            state["context"].update(context)
            return state

        except Exception as e:
            logger.error(f"Error gathering context: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    def _route_to_agent(self, state: Dict[str, Any]) -> str:
        """Determina el agente apropiado basado en la intención"""
        intent = state.get("current_intent", "tour")
        state["active_agent"] = intent
        return intent

    async def _process_tour_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes relacionadas con tours"""
        try:
            preferences = state.get("preferences", {})
            recommendations = await self.tour_agent.plan_tour(preferences)
            
            state["recommendations"] = recommendations.dict()
            state["metadata"]["tour_processing_time"] = datetime.now()
            
            return state

        except Exception as e:
            logger.error(f"Error processing tour request: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    async def _process_food_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes relacionadas con comida"""
        try:
            preferences = state.get("preferences", {})
            recommendations = await self.food_agent.get_restaurants(preferences)
            
            state["recommendations"] = [rec.dict() for rec in recommendations]
            state["metadata"]["food_processing_time"] = datetime.now()
            
            return state

        except Exception as e:
            logger.error(f"Error processing food request: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    async def _process_activity_request(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa solicitudes relacionadas con actividades"""
        try:
            preferences = state.get("preferences", {})
            recommendations = await self.activity_agent.get_activities(preferences)
            
            state["recommendations"] = [rec.dict() for rec in recommendations]
            state["metadata"]["activity_processing_time"] = datetime.now()
            
            return state

        except Exception as e:
            logger.error(f"Error processing activity request: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state

    async def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta basada en recomendaciones y contexto"""
        try:
            # Construir respuesta basada en intención y recomendaciones
            response = self._format_response(state)
            
            state["messages"].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
            
            return state

        except Exception as e:
            logger.error(f"Error processing generate response: {str(e)}")
            state["metadata"]["error"] = str(e)
            return state