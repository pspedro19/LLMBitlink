# app/core/tourism/agents/agent_rag.py
from .base import BaseAgent, ChatState
from app.core.rag.retriever import RAGRetriever

class RAGAgent(BaseAgent):
    def __init__(self, retriever: RAGRetriever):
        super().__init__()
        self.retriever = retriever

    async def process(self, state: ChatState) -> ChatState:
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

        # Update metrics
        if context:
            state["metrics"]["performance_metrics"]["rag_hits"] += 1
        else:
            state["metrics"]["performance_metrics"]["rag_misses"] += 1

        return state