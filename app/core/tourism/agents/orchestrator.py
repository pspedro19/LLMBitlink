from langgraph.graph import StateGraph, END
from typing import Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI

# Import types first
from .types import ChatState, EnhancedConversationMemory
from .memory import PreferenceManager

# Then import agents
from .base import BaseAgent
from .agent_context import ContextAnalysisAgent
from .agent_intent import IntentClassificationAgent
from .agent_sentiment import SentimentAnalysisAgent
from .agent_preferences import PreferenceAgent
from .agent_nlp import NLPAgent
from .agent_rag import RAGAgent
from .agent_conversation import ConversationAgent
from .agent_response import ResponseAgent

# Import external dependencies
from app.core.rag.retriever import RAGRetriever
from app.core.rag.document_processor import DocumentProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)

class TourismOrchestrator(BaseAgent):
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: RAGRetriever,
        document_processor: DocumentProcessor,
        config_path: str
    ):
        super().__init__()
        self.llm = llm
        self.retriever = retriever 
        self.document_processor = document_processor
        self.config_path = config_path
        self.conversation_memory = {}
        self.preference_manager = PreferenceManager()
        
        # Initialize metrics
        self.metrics = {
            "total_interactions": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "intent_distribution": {
                "INITIAL_INQUIRY": 0, "SPECIFIC_QUESTION": 0, "BOOKING_INTENT": 0,
                "OBJECTION": 0, "COMPLAINT": 0, "GRATITUDE": 0, "PREFERENCE_PROVIDING": 0
            },
            "conversion_rate": 0.0,
            "language_distribution": {},
            "performance_metrics": {
                "avg_response_time": 0, "rag_hits": 0, "rag_misses": 0
            }
        }

        # Initialize agents
        self.context_agent = ContextAnalysisAgent()
        self.intent_agent = IntentClassificationAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.preference_agent = PreferenceAgent(self.preference_manager)
        self.nlp_agent = NLPAgent()
        self.rag_agent = RAGAgent(self.retriever)
        self.conversation_agent = ConversationAgent()
        self.response_agent = ResponseAgent(config_path)
        
        self._setup_graph()

    async def process_result(self, result: Dict[str, Any], state: ChatState) -> Dict[str, Any]:
        """Process graph result and update state"""
        if not result:
            logger.error("Graph processing returned None result")
            return state

        # Update state from result
        for key in ["sentiment", "current_intent", "stage", "preferences", "response", "awaiting_preference"]:
            if key in result:
                state[key] = result[key]

        # Update metrics
        sentiment = result.get("sentiment")
        if sentiment and sentiment in self.metrics["sentiment_distribution"]:
            self.metrics["sentiment_distribution"][sentiment] += 1

        intent = result.get("current_intent", {}).get("primary")
        if intent and intent in self.metrics["intent_distribution"]:
            self.metrics["intent_distribution"][intent] += 1

        # Log state updates
        self._debug_log(f"Updated state - sentiment: {sentiment}, intent: {intent}")
        return state

    def _decide_next_step(self, state: ChatState) -> str:
        """Determine next step in conversation flow"""
        state = self._initialize_state(state)
        if not state["memory"].preferences:
            return "NEEDS_PREFERENCES"
        elif not state.get("current_intent") or state["current_intent"].get("primary") == "UNKNOWN":
            return "NEEDS_CLARIFICATION"
        return "CONTINUE"

    def _setup_graph(self):
        """Setup the processing graph with all agents"""
        workflow = StateGraph(ChatState)

        # Add nodes
        workflow.add_node("context_analyzer", self.context_agent.process)
        workflow.add_node("intent_classifier", self.intent_agent.process)
        workflow.add_node("sentiment_analyzer", self.sentiment_agent.process)
        workflow.add_node("preference_manager", self.preference_agent.process)
        workflow.add_node("nlp_processor", self.nlp_agent.process)
        workflow.add_node("rag_retriever", self.rag_agent.process)
        workflow.add_node("conversation_manager", self.conversation_agent.process)
        workflow.add_node("response_generator", self.response_agent.process)

        # Set entry point and edges
        workflow.set_entry_point("context_analyzer")
        workflow.add_conditional_edges(
            "context_analyzer",
            self._decide_next_step,
            {
                "NEEDS_CLARIFICATION": "nlp_processor",
                "NEEDS_PREFERENCES": "nlp_processor",
                "CONTINUE": "nlp_processor"
            }
        )

        # Add main flow edges
        workflow.add_edge("nlp_processor", "intent_classifier")
        workflow.add_edge("intent_classifier", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "preference_manager")
        workflow.add_edge("preference_manager", "rag_retriever")
        workflow.add_edge("rag_retriever", "conversation_manager")
        workflow.add_edge("conversation_manager", "response_generator")
        workflow.add_edge("response_generator", END)

        self.graph = workflow.compile()

    async def invoke(self, user_input: str, session_id: str = "default", language: str = "en") -> Dict[str, Any]:
        """Process user input and generate response"""
        try:
            self.metrics["total_interactions"] += 1

            # Initialize or get session memory
            if session_id not in self.conversation_memory:
                self.conversation_memory[session_id] = EnhancedConversationMemory(language=language)
            memory = self.conversation_memory[session_id]

            # Auto-detect language with enhanced keywords
            lower_input = user_input.lower()
            spanish_keywords = [
                "hola", "qué", "cómo", "cuál", "dónde", "niños", "familia",
                "romántico", "luna de miel", "reserva", "fecha", "gracias",
                "encanta", "gusta", "quiero", "por favor", "gracias", "ayuda",
                "actividades", "recomiendan", "playa", "hotel", "restaurante"
            ]
            detected_lang = "es" if any(word in lower_input for word in spanish_keywords) else language

            # Create initial state
            initial_state = {
                "user_input": user_input,
                "memory": memory,
                "stage": "initial",
                "metrics": self.metrics,
                "preferences": memory.preferences,
                "preferences_complete": False,
                "language": detected_lang,
                "pending_actions": [],
                "current_intent": None,
                "sentiment": None,
                "recommendations": [],
                "rag_context": None,
                "neuro_enhanced": None,
                "response": None,
                "awaiting_preference": memory.awaiting_preference,
                "start_time": datetime.now()
            }

            # Process state
            initial_state = self._initialize_state(initial_state)
            try:
                result = await self.graph.ainvoke(initial_state)
                result = await self.process_result(result, initial_state)
            except Exception as e:
                logger.error(f"Graph processing error: {str(e)}")
                result = initial_state
                result["response"] = "Lo siento, hubo un error procesando tu solicitud." if detected_lang == "es" else "Sorry, there was an error processing your request."

            # Update memory
            memory.preferences.update(result.get("preferences", {}))
            memory.last_intent = result.get("current_intent", {}).get("primary")
            memory.conversation_stage = result.get("stage", "initial")
            
            if result.get("sentiment"):
                memory.sentiment_history.append(result["sentiment"])
            if "awaiting_preference" in result:
                memory.awaiting_preference = result["awaiting_preference"]

            # Create debug info
            debug_info = {
                "intent": memory.last_intent,
                "sentiment": result.get("sentiment"),
                "stage": result.get("stage", "initial"),
                "preferences": memory.preferences.copy(),
                "language": detected_lang,
                "pending_actions": result.get("pending_actions", []),
                "awaiting_preference": result.get("awaiting_preference")
            }

            self._debug_log(f"Final state - sentiment: {result.get('sentiment')}, language: {detected_lang}")
            return {"response": result.get("response", "I understand."), "debug_info": debug_info}

        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            error_msg = "Lo siento, hubo un error en el sistema." if language == "es" else "Sorry, there was a system error."
            return {
                "response": error_msg,
                "debug_info": {
                    "error": str(e),
                    "intent": memory.last_intent if session_id in self.conversation_memory else None,
                    "preferences": memory.preferences.copy() if session_id in self.conversation_memory else {},
                    "stage": "error",
                    "sentiment": None
                }
            }