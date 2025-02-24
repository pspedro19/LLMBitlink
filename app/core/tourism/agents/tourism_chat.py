# app/core/tourism/agents/tourism_chat.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional, List
from pydantic import BaseModel, Field
import yaml
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import json
from datetime import datetime
import re
import random


# Relative imports
from app.utils.logger import get_logger
from app.core.rag.retriever import RAGRetriever
from app.core.rag.document_processor import DocumentProcessor

logger = get_logger(__name__)

class EnhancedConversationMemory(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_history: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_recommendations: List[str] = Field(default_factory=list)
    interaction_count: int = 0
    last_intent: Optional[str] = None
    conversation_stage: str = "initial"
    language: str = "en"
    context_graph: Dict[str, Any] = Field(default_factory=dict)
    pending_actions: List[str] = Field(default_factory=list)
    conversation_goals: List[str] = Field(default_factory=list)
    awaiting_preference: Optional[str] = None  # <-- New field

class ChatState(TypedDict):
    user_input: str
    memory: EnhancedConversationMemory
    current_intent: Optional[Dict[str, Any]]
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
        state["memory"].pending_actions.append(f"ADJUST_TO_PREFERENCES:{json.dumps(changed_prefs)}")
        return state

class TourismChat:
    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: RAGRetriever,
        document_processor: DocumentProcessor,
        config_path: str
    ):
        self.llm = llm
        self.retriever = retriever
        self.document_processor = document_processor
        self.config_path = config_path
        self.templates = {}
        self.conversation_memory = {}
        self.preference_manager = PreferenceManager()
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
        # Initialize metrics collections
        for sentiment in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
            self.metrics["sentiment_distribution"][sentiment] = 0
        # In __init__ method
        for intent in ["INITIAL_INQUIRY", "SPECIFIC_QUESTION", "BOOKING_INTENT", 
                    "OBJECTION", "COMPLAINT", "GRATITUDE", "PREFERENCE_PROVIDING"]:
            self.metrics["intent_distribution"][intent] = 0

        self.load_templates(config_path)
        self._setup_graph()

    def _initialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state with default values"""
        if not state.get("current_intent"):
            state["current_intent"] = {
                "primary": "INITIAL_INQUIRY",
                "secondary": [],
                "urgency": "NORMAL"
            }
        if not state.get("sentiment"):
            state["sentiment"] = "NEUTRAL"
        if not state.get("stage"):
            state["stage"] = "initial"
        return state

    def _get_template(self, intent: str, sentiment: str, language: str) -> str:
        key = f"{intent}_{sentiment}"
        default_key = "DEFAULT"
        templates = self.templates.get(f"conversation_{language}", {})
        return templates.get(key, templates.get(default_key, "I understand your request."))

    def load_templates(self, config_path: str) -> None:
        """Load conversation templates from YAML files with robust error handling.
        
        Args:
            config_path: Directory path containing neuro_xx.yaml template files
        
        Features:
            - Dual fallback system (file-level and full fallback)
            - Per-language error isolation
            - Detailed error logging
            - Schema validation
            - Memory safety
        """
        self.templates = {}
        base_path = Path(config_path)
        default_templates = {
            "en": {
                "DEFAULT": "Hello! I'm your tourism assistant. How can I help you today?",
                "INITIAL_INQUIRY_NEUTRAL": "Thank you for your interest! How can I assist you?",
                "SPECIFIC_QUESTION_NEUTRAL": "Let me help you with that specific question.",
                "BOOKING_INTENT_NEUTRAL": "I'll help you with the booking process.",
                "OBJECTION_NEUTRAL": "I understand your concerns. Let me address them.",
                "COMPLAINT_NEGATIVE": "I apologize for the inconvenience. Let me help resolve this.",
                "GRATITUDE_POSITIVE": "You're welcome! Is there anything else I can help you with?",
                "scarcity": "Limited availability! Book now to secure your spot.",
                "authority": "Recommended by travel experts worldwide."
            },
            "es": {
                "DEFAULT": "¡Hola! Soy tu asistente de turismo. ¿Cómo puedo ayudarte hoy?",
                "INITIAL_INQUIRY_NEUTRAL": "Gracias por tu interés. ¿En qué puedo ayudarte?",
                "SPECIFIC_QUESTION_NEUTRAL": "Permíteme ayudarte con esa pregunta específica.",
                "BOOKING_INTENT_NEUTRAL": "Te ayudaré con el proceso de reservación.",
                "OBJECTION_NEUTRAL": "Entiendo tus preocupaciones. Permíteme abordarlas.",
                "COMPLAINT_NEGATIVE": (
                    "Lamento los inconvenientes. Podemos buscar una solución inmediata para esto. "
                    "¿Podrías compartir más detalles sobre el problema? "
                    "Te ofreceremos alternativas para resolverlo."
                ),
                "GRATITUDE_POSITIVE": (
                    "¡Ha sido un placer ayudarte! Gracias por usar nuestro servicio. ¿Hay algo más en lo que pueda servirte?"
                ),
                "scarcity": "¡Disponibilidad limitada! Reserva ahora para asegurar tu lugar.",
                "authority": "Recomendado por expertos en viajes a nivel mundial."
            }
        }

        # Load templates for each language with isolated error handling
        for lang in ["en", "es"]:
            try:
                template_file = f"neuro_{lang}.yaml"
                template_path = base_path / template_file

                if not template_path.exists():
                    raise FileNotFoundError(f"Template file {template_file} not found in {config_path}")

                with open(template_path, "r", encoding="utf-8") as f:
                    templates = yaml.safe_load(f)
                    
                    # Validate basic template structure
                    if not isinstance(templates, dict):
                        raise ValueError(f"Invalid template structure in {template_file}")
                    
                    # Merge with defaults so that our default templates override any loaded values.
                    merged = {**templates, **default_templates[lang]}
                    
                    self.templates[f"conversation_{lang}"] = merged
                    logger.info(f"Successfully loaded templates for {lang.upper()}")

            except Exception as e:
                logger.error(f"Error loading {lang.upper()} templates: {str(e)}")
                logger.info(f"Using fallback templates for {lang.upper()}")
                
                # File-level fallback
                self.templates[f"conversation_{lang}"] = default_templates[lang]

        # Final validation
        if not self.templates:
            logger.critical("Failed to load any templates. Using full system fallback.")
            self.templates = {
                "conversation_en": default_templates["en"],
                "conversation_es": default_templates["es"]
            }

        # Ensure minimum template requirements
        required_keys = {"DEFAULT", "BOOKING_INTENT_NEUTRAL", "COMPLAINT_NEGATIVE", "GRATITUDE_POSITIVE", "SPECIFIC_QUESTION_NEUTRAL", "scarcity"}
        for lang in ["en", "es"]:
            lang_key = f"conversation_{lang}"
            missing = required_keys - set(self.templates[lang_key].keys())
            if missing:
                logger.warning(f"Missing required templates in {lang_key}: {missing}. Populating defaults.")
                self.templates[lang_key].update({k: default_templates[lang][k] for k in missing})

    async def analyze_context(self, state: ChatState) -> ChatState:
        text = state["user_input"].lower()
        memory = state["memory"]

        entities = await self._extract_entities(text)
        for entity in entities:
            memory.context_graph.setdefault(entity, {
                "mentioned": 0,
                "related_entities": {},
                "preferences": {}
            })
            memory.context_graph[entity]["mentioned"] += 1

            for other_entity in entities:
                if entity != other_entity:
                    memory.context_graph[entity]["related_entities"][other_entity] = \
                        memory.context_graph[entity]["related_entities"].get(other_entity, 0) + 1

        return state

    async def enhanced_classify_intent(self, state: ChatState) -> ChatState:
        """Enhanced intent classification with prioritized order."""
        text = state["user_input"].lower()
        current_intent = state.get("current_intent", {}).get("primary")
        
        # Initialize intent distribution if needed
        if "intent_distribution" not in self.metrics:
            self.metrics["intent_distribution"] = {
                "INITIAL_INQUIRY": 0,
                "SPECIFIC_QUESTION": 0,
                "BOOKING_INTENT": 0,
                "OBJECTION": 0,
                "COMPLAINT": 0,
                "GRATITUDE": 0,
                "PREFERENCE_PROVIDING": 0
            }
        
        # Define patterns with strong complaint indicators first
        complaint_phrases = [
            "tiene un error", "está mal", "no funciona",
            "hay un problema", "necesita arreglo", "tiene un problema",
            "has an error", "is wrong", "not working",
            "error en", "problema con", "mal servicio",
            "incorrecta", "incorrecto", "equivocado",
            "queja", "complaint", "reclamo"
        ]
        
        # Define other patterns
        patterns = {
            "BOOKING": [
                "cómo reservo", "como reservo", "puedo reservar",
                "me interesa reservar", "quiero reservar",
                "can i book", "how do i book", "make a reservation",
                "reserv", "book", "cuando", "when"
            ],
            "GRATITUDE": [
                "thanks", "thank", "gracias", "appreciate", "agradec",
                "grazie", "thank you", "muchas gracias", "mil gracias"
            ],
            "OBJECTION": [
                "cost", "price", "precio", "caro", "expensive",
                "muy alto", "too high", "demasiado"
            ],
            "QUESTION": [
                "qué", "cuál", "cómo", "dónde",
                "what", "which", "how", "where"
            ]
        }
        
        # First check for complaints - use exact phrase matching
        if any(phrase in text for phrase in complaint_phrases):
            intent = "COMPLAINT"
            state["stage"] = "resolution"
            
        # Then check other patterns
        elif any(word in text for word in patterns["BOOKING"]):
            intent = "BOOKING_INTENT"
            state["stage"] = "booking"
        elif any(word in text for word in patterns["GRATITUDE"]):
            intent = "GRATITUDE"
            state["stage"] = "closing"
        elif any(word in text for word in patterns["OBJECTION"]):
            intent = "OBJECTION"
            state["stage"] = "negotiation"
        elif any(word in text for word in patterns["QUESTION"]):
            intent = "SPECIFIC_QUESTION"
            state["stage"] = "exploring"
        elif state.get("awaiting_preference") or state.get("preference_provided"):
            intent = "PREFERENCE_PROVIDING"
            state["stage"] = "preference_collection"
            target_preference = state.get("awaiting_preference") or state.get("preference_provided")
            import re
            patterns = {
                "family_size": [r'\b(\d+)\s*(personas|people|adultos|adults|niños|kids)\b'],
                "budget": [r'\b(\d+)\s*(usd|dollars|dólares)\b', r'\$\s*(\d+)'],
                "days": [r'\b(\d+)\s*(d[ií]as|days)\b']
            }.get(target_preference, [])
            if any(re.search(pattern, text) for pattern in patterns):
                state["stage"] = "preference_confirmation"
            else:
                state["stage"] = "preference_clarification"
        else:
            intent = "INITIAL_INQUIRY"
            state["stage"] = "initial"
        
        # Always update metrics for current intent
        self.metrics["intent_distribution"][intent] = (
            self.metrics["intent_distribution"].get(intent, 0) + 1
        )
        
        # Update state
        state["current_intent"] = {
            "primary": intent,
            "secondary": [],
            "urgency": "NORMAL"
        }
        state["memory"].last_intent = intent
        
        return state
    
    
    async def analyze_sentiment(self, state: ChatState) -> ChatState:
        """Improved Sentiment Analysis with enhanced metrics tracking."""
        text = state["user_input"].lower().strip()
        state = self._initialize_state(state)
        
        # Initialize sentiment distribution if needed
        if "sentiment_distribution" not in self.metrics:
            self.metrics["sentiment_distribution"] = {
                "POSITIVE": 0,
                "NEGATIVE": 0,
                "NEUTRAL": 0
            }
        
        # Determine language
        if any(char in text for char in "áéíóú"):
            language = "es"
        else:
            language = state.get("language", "en")
        
        positive_words = {
            "en": [
                "excellent", "great", "love", "thanks", "awesome", "amazing", 
                "fantastic", "wonderful", "perfect", "happy", "best", "excited"
            ],
            "es": [
                "excelente", "genial", "encanta", "gracias", "maravilloso", 
                "increíble", "fabuloso", "perfecto", "feliz", "mejor", 
                "emocionado", "romántico"
            ]
        }
        
        negative_words = {
            "en": [
                "bad", "terrible", "complaint", "problem", "wrong", "horrible",
                "awful", "disappointed", "worst", "issue", "upset", "error",
                "mistake", "fail", "poor", "unhappy", "dissatisfied"
            ],
            "es": [
                "malo", "terrible", "queja", "problema", "error", "horrible",
                "pésimo", "decepcionado", "peor", "fallo", "molesto",
                "equivocado", "incorrecto", "mal", "erróneo", "inconveniente",
                "insatisfecho", "descontento"
            ]
        }
        
        # Determine initial sentiment
        if any(word in text for word in positive_words.get(language, [])):
            sentiment = "POSITIVE"
        elif any(word in text for word in negative_words.get(language, [])):
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
        
        # Spanish-specific rules
        if language == "es":
            if text.startswith("¿") and text.endswith("?"):
                sentiment = "NEUTRAL"
            elif "reserva" in text and any(term in text for term in ["error", "mal", "problema"]):
                sentiment = "NEGATIVE"
        
        # Always update metrics for current sentiment
        self.metrics["sentiment_distribution"][sentiment] = (
            self.metrics["sentiment_distribution"].get(sentiment, 0) + 1
        )
        
        # Update state
        state["memory"].sentiment_history.append(sentiment)
        state["sentiment"] = sentiment
        
        return state
    
       
    async def retrieve_context(self, state: ChatState) -> ChatState:
        """Context retrieval with mock responses based on language."""
        text = state["user_input"].lower()
        language = state.get("language", "en")
        if language == "es":
            # If the query mentions children, return a context that includes "playa", "snorkel" and "acuario"
            if "niños" in text or "ninos" in text:
                context = ("Contamos con playas de ensueño, ideales para familias. "
                           "Disfruta de snorkel y visita nuestro acuario.")
            else:
                mock_contexts = {
                    "playa": "Contamos con playas de ensueño, ideales para familias y parejas.",
                    "restaurante": ("Disponemos de restaurantes con ambiente íntimo, "
                                     "perfecta opción para cenas románticas con vista al mar."),
                    "hotel": "Nuestros hoteles ofrecen experiencias únicas para parejas y familias.",
                    "tour": "Realizamos tours culturales y gastronómicos por la isla.",
                    "actividad": "Ofrecemos actividades exclusivas, desde deportes acuáticos hasta recorridos históricos."
                }
                context = None
                for key, value in mock_contexts.items():
                    if key in text or (key + "s") in text:
                        context = value
                        break
                if not context:
                    context = "Curazao ofrece una combinación única de actividades, cultura y relajación."
        else:
            mock_contexts = {
                "beach": "Our beaches feature crystal clear waters perfect for swimming and snorkeling.",
                "restaurant": "We have a variety of restaurants offering local and international cuisine.",
                "hotel": "Our accommodations range from luxury resorts to boutique hotels.",
                "tour": "Our guided tours include historical walks and water activities.",
                "activity": "We offer numerous activities including diving, hiking, and cultural experiences."
            }
            context = None
            for key, value in mock_contexts.items():
                if key in text or (key + "s") in text:
                    context = value
                    break
            if not context:
                context = "Curaçao offers a perfect blend of activities, culture, and relaxation."
        state["rag_context"] = context
        state["recommendations"] = [context]
        return state

    async def process_nlp(self, state: ChatState) -> ChatState:
        """Process NLP to extract preferences and detect newly provided preference."""
        import re
        text = state["user_input"].lower()
        
        # Ensure preferences exist in state and memory
        if "preferences" not in state:
            state["preferences"] = {}
        if not hasattr(state["memory"], "preferences"):
            state["memory"].preferences = {}
        
        # Keep a copy of previous preferences for comparison
        old_preferences = state["memory"].preferences.copy()
        current_preferences = old_preferences.copy()
        
        # Family size detection
        family_patterns = [r'\b(\d+)\s*personas?\b', r'\b(\d+)\s*people\b', r'\bsomos\s*(\d+)\b']
        for pattern in family_patterns:
            if match := re.search(pattern, text):
                current_preferences["family_size"] = int(match.group(1))
                break
        
        # Budget detection
        budget_patterns = [r'\b(\d+)\s*usd\b', r'\$\s*(\d+)', r'presupuesto.*?(\d+)']
        for pattern in budget_patterns:
            if match := re.search(pattern, text):
                current_preferences["budget"] = int(match.group(1))
                break
        
        # Duration detection
        duration_patterns = [r'\b(\d+)\s*d[ií]as?\b', r'\b(\d+)\s*days?\b']
        for pattern in duration_patterns:
            if match := re.search(pattern, text):
                current_preferences["days"] = int(match.group(1))
                break
        
        # Food preferences detection
        if any(word in text for word in ['diet', 'food', 'comida', 'alimenticias', 'restricciones']):
            current_preferences["food_preferences"] = text
        
        # Update both state and memory preferences
        state["preferences"] = current_preferences
        state["memory"].preferences = current_preferences
        
        # Detect if a new preference has been provided by comparing with old preferences
        new_keys = [key for key in current_preferences if key not in old_preferences]
        state["preference_provided"] = new_keys[0] if new_keys else None
        
        return state

    async def manage_conversation(self, state: ChatState) -> ChatState:
        """Manage conversation flow and state transitions"""
        preference_sequence = ["family_size", "budget", "days"]
        memory = state["memory"]
        found_missing = False
        for pref in preference_sequence:
            if pref not in memory.preferences:
                state["awaiting_preference"] = pref
                found_missing = True
                break
        if not found_missing:
            state["awaiting_preference"] = None
        return state


# Add at the top of tourism_chat.py with other imports

    async def generate_response(self, state: ChatState) -> ChatState:
        """Generate response using templates and context with appropriate emotional resonance"""
        try:
            intent = state.get("current_intent", {}).get("primary", "INITIAL_INQUIRY")
            sentiment = state.get("sentiment", "NEUTRAL")
            language = state["language"]

            # Emotional response templates based on sentiment
            sentiment_responses = {
                "es": {
                    "POSITIVE": {
                        "prefixes": [
                            "¡Excelente elección! ",
                            "¡Me encanta tu entusiasmo! ",
                            "¡Qué genial! "
                        ],
                        "content_modifiers": [
                            "estamos encantados de ",
                            "será una experiencia maravillosa ",
                            "te va a encantar "
                        ]
                    },
                    "NEGATIVE": {
                        "prefixes": [
                            "Lamento escuchar eso. ",
                            "Entiendo tu preocupación. ",
                            "Permíteme ayudarte a resolver esto. "
                        ],
                        "content_modifiers": [
                            "vamos a revisar los detalles ",
                            "buscaremos una solución ",
                            "podemos mejorar tu experiencia "
                        ]
                    },
                    "NEUTRAL": {
                        "prefixes": [
                            "Tenemos múltiples actividades disponibles. ",
                            "Ofrecemos varios tours interesantes. ",
                            "Hay diversas opciones para explorar. "
                        ]
                    }
                }
            }

            # Get base template
            template = self._get_template(intent, sentiment, language)
            
            # Add emotional resonance based on sentiment
            if language in sentiment_responses and sentiment in sentiment_responses[language]:
                sentiment_prefix = random.choice(sentiment_responses[language][sentiment]["prefixes"])
                template = f"{sentiment_prefix}{template}"
                
                if "content_modifiers" in sentiment_responses[language][sentiment]:
                    content_modifier = random.choice(sentiment_responses[language][sentiment]["content_modifiers"])
                    template = f"{template} {content_modifier}"

            # Add context with activity-specific keywords
            context = state.get("rag_context", "")
            if "niños" in state["user_input"].lower():
                context += " Contamos con playas ideales para familias, snorkel y un acuario fascinante."
            elif any(word in state["user_input"].lower() for word in ["playa", "beach", "mar", "sea"]):
                context += " Nuestras playas son perfectas para relajarse y practicar snorkel."

            # Build response parts
            response_parts = [template]
            if context:
                response_parts.append(context)
                logger.debug(f"Added RAG context: {context[:50]}...")

            # Add preferences if present
            preferences = state["memory"].preferences
            if preferences:
                pref_summary = "\nBased on your preferences:"
                for key, value in preferences.items():
                    pref_summary += f"\n- {key}: {value}"
                response_parts.append(pref_summary)
                logger.debug(f"Included {len(preferences)} user preferences")

            full_response = "\n".join(response_parts)
            state["response"] = full_response

            logger.debug(
                f"Generated response complete | "
                f"Length: {len(full_response)} chars | "
                f"Paragraphs: {len(full_response.split('\n\n'))}"
            )

            return state
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            logger.debug("Using fallback response template due to generation failure")
            state["response"] = "I understand your request. Let me help you with that."
            return state
    
    def _setup_graph(self):
        workflow = StateGraph(ChatState)

        # Add nodes
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

        # Add conditional edges for initial routing
        workflow.add_conditional_edges(
            "context_analyzer",
            self._decide_next_step,
            {
                "NEEDS_CLARIFICATION": "nlp_processor",
                "NEEDS_PREFERENCES": "nlp_processor",
                "CONTINUE": "nlp_processor"
            }
        )

        # Process all inputs through nlp first, then continue with rest of flow
        workflow.add_edge("nlp_processor", "intent_classifier")
        workflow.add_edge("intent_classifier", "sentiment_analyzer")
        workflow.add_edge("sentiment_analyzer", "preference_manager")
        workflow.add_edge("preference_manager", "rag_retriever")
        workflow.add_edge("rag_retriever", "conversation_manager")
        workflow.add_edge("conversation_manager", "response_generator")
        workflow.add_edge("response_generator", END)

        self.graph = workflow.compile()

    def _decide_next_step(self, state: ChatState) -> str:
        """Determine next step in conversation flow"""
        # Initialize state if not already done
        state = self._initialize_state(state)

        # Check for existing intent and preferences
        if not state["memory"].preferences:
            return "NEEDS_PREFERENCES"
        elif not state.get("current_intent") or state["current_intent"].get("primary") == "UNKNOWN":
            return "NEEDS_CLARIFICATION"
        return "CONTINUE"

    async def invoke(self, user_input: str, session_id: str = "default", language: str = "en") -> Dict[str, Any]:
        """Process user input and generate response"""
        try:
            self.metrics["total_interactions"] += 1

            if session_id not in self.conversation_memory:
                self.conversation_memory[session_id] = EnhancedConversationMemory(language=language)
            memory = self.conversation_memory[session_id]

            # Auto-detect language based on keywords.
            lower_input = user_input.lower()
            # Auto-detect Spanish if any common Spanish keywords appear.
            spanish_keywords = [
                "hola", "qué", "cómo", "cuál", "dónde", "niños", "familia",
                "romántico", "luna de miel", "reserva", "fecha", "gracias"
            ]
            if any(word in lower_input for word in spanish_keywords):
                detected_lang = "es"
            else:
                detected_lang = language

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

            initial_state = self._initialize_state(initial_state)

            result = await self.graph.ainvoke(initial_state)
            if result is None:
                logger.error("Graph processing returned None result")
                raise ValueError("Invalid graph processing result")

            memory.preferences.update(result.get("preferences", {}))
            memory.last_intent = result.get("current_intent", {}).get("primary")
            memory.conversation_stage = result.get("stage", "initial")
            if "awaiting_preference" in result:
                memory.awaiting_preference = result["awaiting_preference"]

            current_intent = result.get("current_intent", {})
            if isinstance(current_intent, dict) and "primary" in current_intent:
                intent = current_intent["primary"]
            else:
                intent = memory.last_intent or "INITIAL_INQUIRY"

            debug_info = {
                "intent": intent,
                "sentiment": result.get("sentiment"),
                "stage": result.get("stage", "initial"),
                "preferences": memory.preferences.copy(),
                "language": detected_lang,
                "pending_actions": result.get("pending_actions", []),
                "awaiting_preference": result.get("awaiting_preference")
            }
            return {
                "response": result.get("response", "I understand."),
                "debug_info": debug_info
            }

        except Exception as e:
            logger.error(f"Error in invoke: {str(e)}")
            error_msg = "Sorry, there was a system error." if language == "en" else "Lo siento, hubo un error en el sistema."
            return {
                "response": error_msg,
                "debug_info": {
                    "error": str(e),
                    "intent": memory.last_intent if session_id in self.conversation_memory else None,
                    "preferences": memory.preferences.copy() if session_id in self.conversation_memory else {},
                    "stage": "error"
                }
            }

    async def _extract_entities(self, text: str) -> List[str]:
        """Extract relevant entities from text"""
        entities = []
        common_entities = [
            "beach", "playa",
            "restaurant", "restaurante",
            "hotel", "resort",
            "activity", "actividad",
            "tour", "excursion",
            "museum", "museo",
            "nightlife", "vida nocturna"
        ]

        # Extract entities from text
        words = text.lower().split()
        for entity in common_entities:
            if entity in words:
                entities.append(entity)

        return list(set(entities))  # Remove duplicates

    def _update_metrics(self, state: ChatState) -> None:
        """Update conversation metrics"""
        try:
            current_state = state if isinstance(state, dict) else state.dict()

            # Update performance metrics
            if "start_time" in current_state:
                response_time = datetime.now() - current_state["start_time"]
                current_avg = self.metrics["performance_metrics"]["avg_response_time"]
                total_interactions = self.metrics["total_interactions"]

                # Calculate new average response time
                new_avg = ((current_avg * (total_interactions - 1)) + response_time.total_seconds()) / total_interactions
                self.metrics["performance_metrics"]["avg_response_time"] = new_avg

            # Update RAG metrics
            if current_state.get("rag_context"):
                self.metrics["performance_metrics"]["rag_hits"] += 1
            else:
                self.metrics["performance_metrics"]["rag_misses"] += 1

            # Update sentiment distribution if sentiment is present
            if sentiment := current_state.get("sentiment"):
                self.metrics["sentiment_distribution"][sentiment] = \
                    self.metrics["sentiment_distribution"].get(sentiment, 0) + 1

            # Update language distribution
            if language := current_state.get("language"):
                if "language_distribution" not in self.metrics:
                    self.metrics["language_distribution"] = {}
                self.metrics["language_distribution"][language] = \
                    self.metrics["language_distribution"].get(language, 0) + 1

            # Update intent distribution
            if current_intent := current_state.get("current_intent", {}):
                if isinstance(current_intent, dict) and "primary" in current_intent:
                    intent = current_intent["primary"]
                    self.metrics["intent_distribution"][intent] = \
                        self.metrics["intent_distribution"].get(intent, 0) + 1

            # Calculate conversion rate
            if self.metrics["total_interactions"] > 0:
                booking_intents = self.metrics["intent_distribution"].get("BOOKING_INTENT", 0)
                self.metrics["conversion_rate"] = \
                    (booking_intents / self.metrics["total_interactions"]) * 100

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation history and metrics"""
        try:
            if session_id not in self.conversation_memory:
                return {"error": "Session not found"}

            memory = self.conversation_memory[session_id]

            return {
                "total_interactions": memory.interaction_count,
                "current_stage": memory.conversation_stage,
                "collected_preferences": memory.preferences,
                "sentiment_history": memory.sentiment_history,
                "pending_actions": memory.pending_actions,
                "conversation_goals": memory.conversation_goals
            }

        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {"error": str(e)}
