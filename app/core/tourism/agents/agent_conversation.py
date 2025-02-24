# app/core/tourism/agents/agent_conversation.py
from .base import BaseAgent, ChatState

class ConversationAgent(BaseAgent):
    async def process(self, state: ChatState) -> ChatState:
        """Manage conversation flow and state transitions"""
        preference_sequence = ["family_size", "budget", "days"]
        memory = state["memory"]
        found_missing = False

        # Check for missing preferences in sequence
        for pref in preference_sequence:
            if pref not in memory.preferences:
                state["awaiting_preference"] = pref
                found_missing = True
                break

        if not found_missing:
            state["awaiting_preference"] = None
            state["preferences_complete"] = True

        # Update conversation stage based on completion
        if state["preferences_complete"]:
            if state["stage"] == "preference_collection":
                state["stage"] = "exploring"

        # Track conversation metrics
        memory.interaction_count += 1
        if "language" in state:
            if "language_distribution" not in state["metrics"]:
                state["metrics"]["language_distribution"] = {}
            state["metrics"]["language_distribution"][state["language"]] = \
                state["metrics"]["language_distribution"].get(state["language"], 0) + 1

        # Update conversion metrics if relevant
        if state["current_intent"].get("primary") == "BOOKING_INTENT":
            total = state["metrics"]["total_interactions"]
            bookings = state["metrics"]["intent_distribution"].get("BOOKING_INTENT", 0)
            if total > 0:
                state["metrics"]["conversion_rate"] = (bookings / total) * 100

        return state