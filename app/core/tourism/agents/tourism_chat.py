from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional, List
from pydantic import BaseModel, Field
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
import json
from datetime import datetime
import re


# Relative imports
from app.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedConversationMemory(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_history: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_recommendations: List[str] = Field(default_factory=list)
    interaction_count: int = 0
    last_intent: Optional[str] = None
    conversation_stage: str = "initial"
    language: str = "en"  # Default to English
    context_graph: Dict[str, Any] = Field(default_factory=dict)
    pending_actions: List[str] = Field(default_factory=list)
    conversation_goals: List[str] = Field(default_factory=list)

class ChatState(TypedDict):
    user_input: str
    memory: EnhancedConversationMemory
    current_intent: Optional[Dict[str, Any]]  # Enhanced intent structure
    sentiment: Optional[str]
    recommendations: List[str]
    rag_context: Optional[str]
    neuro_enhanced: Optional[str]
    response: Optional[str]
    metrics: Dict[str, Any]
    stage: str
    preferences: Dict[str, Any]
    awaiting_preference: Optional[str]
    preferences_complete: bool
    language: str

class PreferenceManager:
    def __init__(self):
        self.preference_states = {}
        self.preference_history = []

    async def detect_preference_shift(self, state: ChatState) -> ChatState:
        current_prefs = state["preferences"]
        historical_prefs = state["memory"].preferences
        
        changed_prefs = {
            k: v for k, v in current_prefs.items()
            if historical_prefs.get(k) != v
        }
        
        if changed_prefs:
            state = await self._trigger_agent_adjustment(state, changed_prefs)
            
        return state

    async def _trigger_agent_adjustment(self, state: ChatState, changed_prefs: Dict[str, Any]) -> ChatState:
        # Implement preference-based adjustments
        state["memory"].pending_actions.append(f"ADJUST_TO_PREFERENCES:{json.dumps(changed_prefs)}")
        return state

class TourismChat:
    def __init__(self, llm, vector_store: FAISS, config_path: str = "config/", language: str = "en"):
        self.llm = llm
        self.vector_store = vector_store
        self.language = language
        self.load_templates(config_path)
        self.preference_manager = PreferenceManager()
        self._setup_graph()
        self.metrics = {
            "total_interactions": 0,
            "sentiment_distribution": {},
            "intent_distribution": {},
            "conversion_rate": 0.0,
            "language_distribution": {},
            "performance_metrics": {
                "avg_response_time": 0,
                "rag_hits": 0,
                "rag_misses": 0
            }
        }

    def load_templates(self, config_path: str):
        """Load templates with language support"""
        templates = {}
        for lang in ["en", "es"]:
            with open(f"{config_path}/conversation_{lang}.yaml", "r", encoding="utf-8") as f:
                templates[lang] = yaml.safe_load(f)
            with open(f"{config_path}/neuro_{lang}.yaml", "r", encoding="utf-8") as f:
                templates[f"neuro_{lang}"] = yaml.safe_load(f)
            with open(f"{config_path}/preferences_{lang}.yaml", "r", encoding="utf-8") as f:
                templates[f"preferences_{lang}"] = yaml.safe_load(f)
        
        self.templates = templates

    async def analyze_context(self, state: ChatState) -> ChatState:
        """Enhanced context analysis"""
        text = state["user_input"].lower()
        memory = state["memory"]
        
        # Update context graph
        entities = await self._extract_entities(text)
        for entity in entities:
            memory.context_graph.setdefault(entity, {
                "mentioned": 0,
                "related_entities": {},
                "preferences": {}
            })
            memory.context_graph[entity]["mentioned"] += 1
            
            # Update entity relationships
            for other_entity in entities:
                if entity != other_entity:
                    memory.context_graph[entity]["related_entities"][other_entity] = \
                        memory.context_graph[entity]["related_entities"].get(other_entity, 0) + 1
        
        return state

    async def enhanced_classify_intent(self, state: ChatState) -> ChatState:
        """ML-powered intent classification"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze this tourism-related query. Return JSON with intent, urgency, and topics."),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        result = await chain.ainvoke({"input": state["user_input"]})
        
        state["current_intent"] = {
            "primary": result.get("intent", "UNKNOWN"),
            "secondary": result.get("topics", []),
            "urgency": result.get("urgency", "NORMAL")
        }
        
        self.metrics["intent_distribution"][state["current_intent"]["primary"]] = \
            self.metrics["intent_distribution"].get(state["current_intent"]["primary"], 0) + 1
            
        return state

    def _setup_graph(self):
        """Enhanced graph setup with dynamic routing"""
        workflow = StateGraph(ChatState)
        
        # Core nodes
        workflow.add_node("context_analyzer", self.analyze_context)
        workflow.add_node("intent_classifier", self.enhanced_classify_intent)
        workflow.add_node("sentiment_analyzer", self.analyze_sentiment)
        workflow.add_node("preference_manager", self.preference_manager.detect_preference_shift)
        workflow.add_node("nlp_processor", self.process_nlp)
        workflow.add_node("rag_retriever", self.retrieve_context)
        workflow.add_node("conversation_manager", self.manage_conversation)
        workflow.add_node("response_generator", self.generate_response)

        # Set entry point
        workflow.set_entry_point("context_analyzer")
        
        # Dynamic routing
        workflow.add_conditional_edges(
            "context_analyzer",
            self._decide_next_step,
            {
                "NEEDS_CLARIFICATION": "intent_classifier",
                "NEEDS_PREFERENCES": "preference_manager",
                "CONTINUE": "intent_classifier"
            }
        )

        # Main flow
        workflow.add_edge("intent_classifier", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "preference_manager")
        workflow.add_edge("preference_manager", "nlp_processor")
        workflow.add_edge("nlp_processor", "rag_retriever")
        workflow.add_edge("rag_retriever", "conversation_manager")
        workflow.add_edge("conversation_manager", "response_generator")
        workflow.add_edge("response_generator", END)

        self.graph = workflow.compile()

    def _decide_next_step(self, state: ChatState) -> str:
        """Dynamic routing decision"""
        if not state["memory"].preferences:
            return "NEEDS_PREFERENCES"
        elif state["current_intent"] is None or state["current_intent"].get("primary") == "UNKNOWN":
            return "NEEDS_CLARIFICATION"
        return "CONTINUE"

    async def analyze_sentiment(self, state: ChatState) -> ChatState:
        """Sentiment analysis with language support"""
        text = state["user_input"].lower()
        
        # English sentiment analysis
        if state["language"] == "en":
            if any(word in text for word in ["excellent", "great", "love", "thanks"]):
                sentiment = "POSITIVE"
            elif any(word in text for word in ["bad", "terrible", "complaint", "problem"]):
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
        # Spanish sentiment analysis
        else:
            if any(word in text for word in ["excelente", "genial", "encanta", "gracias"]):
                sentiment = "POSITIVE"
            elif any(word in text for word in ["malo", "terrible", "queja", "problema"]):
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            
        self.metrics["sentiment_distribution"][sentiment] = \
            self.metrics["sentiment_distribution"].get(sentiment, 0) + 1
            
        return {**state, "sentiment": sentiment}

    async def process_nlp(self, state: ChatState) -> ChatState:
        """NLP processing with language support"""
        # Implementation remains the same as in your original code
        # ... [previous implementation]
        pass

    async def retrieve_context(self, state: ChatState) -> ChatState:
        """RAG with language support"""
        # Implementation remains the same as in your original code
        # ... [previous implementation]
        pass

    async def invoke(self, user_input: str, session_id: str = "default", language: str = "en") -> Dict[str, Any]:
        """Enhanced invocation"""
        try:
            initial_state = {
                "user_input": user_input,
                "memory": EnhancedConversationMemory(language=language),
                "stage": "initial",
                "metrics": self.metrics,
                "preferences": {},
                "preferences_complete": False,
                "language": language,
                "pending_actions": []
            }
            
            result = await self.graph.ainvoke(initial_state)
            
            # Update metrics
            self.metrics["language_distribution"][language] = \
                self.metrics["language_distribution"].get(language, 0) + 1
            
            debug_info = {
                "intent": result.get("current_intent"),
                "sentiment": result.get("sentiment"),
                "stage": result.get("stage", "initial"),
                "preferences": result.get("preferences", {}),
                "language": language,
                "pending_actions": result.get("pending_actions", [])
            }
            
            return {
                "response": result["response"],
                "debug_info": debug_info
            }
            
        except Exception as e:
            error_msg = "Sorry, there was a system error." if language == "en" else \
                       "Lo siento, hubo un error en el sistema."
            return {
                "response": error_msg,
                "error": str(e)
            }

    async def _extract_entities(self, text: str) -> List[str]:
        """Entity extraction for context graph"""
        # Implement entity extraction logic
        # This is a placeholder implementation
        entities = []
        common_entities = ["beach", "restaurant", "hotel", "activity", "tour"]
        for entity in common_entities:
            if entity in text.lower():
                entities.append(entity)
        return entities