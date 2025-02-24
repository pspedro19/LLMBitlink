# app/core/tourism/agents/agent_response.py
import yaml
import random
from pathlib import Path
from .base import BaseAgent, ChatState
from app.utils.logger import get_logger

logger = get_logger(__name__)

class ResponseAgent(BaseAgent):
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.templates = {}
        self.load_templates(config_path)

    def load_templates(self, config_path: str) -> None:
        """Load conversation templates from YAML files"""
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
                "COMPLAINT_NEGATIVE": "Lamento los inconvenientes. Podemos buscar una solución inmediata.",
                "GRATITUDE_POSITIVE": "¡Ha sido un placer ayudarte!",
                "scarcity": "¡Disponibilidad limitada! Reserva ahora para asegurar tu lugar.",
                "authority": "Recomendado por expertos en viajes a nivel mundial."
            }
        }

        # Load language-specific templates
        for lang in ["en", "es"]:
            try:
                template_file = f"neuro_{lang}.yaml"
                template_path = base_path / template_file

                if not template_path.exists():
                    raise FileNotFoundError(f"Template file {template_file} not found")

                with open(template_path, "r", encoding="utf-8") as f:
                    templates = yaml.safe_load(f)
                    if not isinstance(templates, dict):
                        raise ValueError(f"Invalid template structure in {template_file}")
                    
                    # Merge with defaults
                    merged = {**templates, **default_templates[lang]}
                    self.templates[f"conversation_{lang}"] = merged

            except Exception as e:
                logger.error(f"Error loading {lang.upper()} templates: {str(e)}")
                logger.info(f"Using fallback templates for {lang.upper()}")
                self.templates[f"conversation_{lang}"] = default_templates[lang]

    def _get_template(self, intent: str, sentiment: str, language: str) -> str:
        """Get appropriate template based on intent and sentiment"""
        key = f"{intent}_{sentiment}"
        default_key = "DEFAULT"
        templates = self.templates.get(f"conversation_{language}", {})
        return templates.get(key, templates.get(default_key, "I understand your request."))

    async def process(self, state: ChatState) -> ChatState:
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