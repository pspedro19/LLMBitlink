from typing import Dict, Any
from .base import BaseAgent, ChatState

class IntentClassificationAgent(BaseAgent):
    """Agent responsible for classifying user intent"""
    
    async def process(self, state: ChatState) -> ChatState:
        """Classify user intent from input"""
        text = state["user_input"].lower()
        
        # Define patterns
        complaint_phrases = [
            "tiene un error", "está mal", "no funciona",
            "hay un problema", "necesita arreglo", "tiene un problema",
            "has an error", "is wrong", "not working",
            "error en", "problema con", "mal servicio"
        ]
        
        patterns = {
            "BOOKING": [
                "cómo reservo", "como reservo", "puedo reservar",
                "me interesa reservar", "quiero reservar", "hacer una reserva", 
                "quiero hacer una reserva",  # Added for the failing test case
                "can i book", "how do i book", "make a reservation"
            ],
            "GRATITUDE": [
                "thanks", "thank", "gracias", "appreciate", "agradec",
                "grazie", "thank you", "muchas gracias"
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
        
        # First check for complaints
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
        else:
            intent = "INITIAL_INQUIRY"
            state["stage"] = "initial"
        
        # Update metrics
        if "intent_distribution" in state["metrics"]:
            state["metrics"]["intent_distribution"][intent] = (
                state["metrics"]["intent_distribution"].get(intent, 0) + 1
            )
        
        # Update state
        state["current_intent"] = {
            "primary": intent,
            "secondary": [],
            "urgency": "NORMAL"
        }
        state["memory"].last_intent = intent
        
        return state