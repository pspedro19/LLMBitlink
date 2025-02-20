from .base import BaseAgent, ChatState
from .memory import PreferenceManager

class PreferenceAgent(BaseAgent):
    def __init__(self, preference_manager: PreferenceManager):
        super().__init__()
        self.preference_manager = preference_manager

    async def process(self, state: ChatState) -> ChatState:
        """Manage user preferences and detect preference shifts"""
        # First detect any preference shifts
        state = await self.preference_manager.detect_preference_shift(state)
        
        # Process preferences in sequence
        preference_sequence = ["family_size", "budget", "days"]
        memory = state["memory"]

        # Check if there's an awaiting preference response
        if state.get("awaiting_preference"):
            current_pref = state["awaiting_preference"]
            if current_pref in state["preferences"]:
                # Preference was provided, move to next
                idx = preference_sequence.index(current_pref)
                if idx + 1 < len(preference_sequence):
                    state["awaiting_preference"] = preference_sequence[idx + 1]
                else:
                    state["awaiting_preference"] = None
                    state["preferences_complete"] = True

        # Validate preferences
        preferences = state["preferences"]
        if "family_size" in preferences:
            try:
                family_size = int(preferences["family_size"])
                if family_size <= 0:
                    del preferences["family_size"]
            except (ValueError, TypeError):
                del preferences["family_size"]

        if "budget" in preferences:
            try:
                budget = float(preferences["budget"])
                if budget <= 0:
                    del preferences["budget"]
            except (ValueError, TypeError):
                del preferences["budget"]

        if "days" in preferences:
            try:
                days = int(preferences["days"])
                if days <= 0:
                    del preferences["days"]
            except (ValueError, TypeError):
                del preferences["days"]

        # Update memory with validated preferences
        memory.preferences.update(preferences)
        
        # Check if all required preferences are collected
        has_all_preferences = all(pref in memory.preferences for pref in preference_sequence)
        state["preferences_complete"] = has_all_preferences

        # If all preferences are collected, update conversation stage
        if has_all_preferences and state["stage"] == "preference_collection":
            state["stage"] = "exploring"

        return state

    def _normalize_preference_value(self, value: str) -> str:
        """Normalize preference values for consistent storage"""
        if isinstance(value, (int, float)):
            return str(value)
        return value.lower().strip()