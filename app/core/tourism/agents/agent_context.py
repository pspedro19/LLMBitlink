# app/core/tourism/agents/agent_context.py
from typing import List
from .base import BaseAgent, ChatState

class ContextAnalysisAgent(BaseAgent):
    async def process(self, state: ChatState) -> ChatState:
        """Analyze conversation context"""
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

        words = text.lower().split()
        for entity in common_entities:
            if entity in words:
                entities.append(entity)

        return list(set(entities))