from typing import Dict, Any
import json
from .types import EnhancedConversationMemory

class PreferenceManager:
    """Manager for handling user preferences"""
    def __init__(self):
        self.preference_states = {}
        self.preference_history = []

    async def detect_preference_shift(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect changes in user preferences"""
        current_prefs = state["preferences"]
        historical_prefs = state["memory"].preferences

        changed_prefs = {
            k: v for k, v in current_prefs.items()
            if historical_prefs.get(k) != v
        }

        if changed_prefs:
            state = await self._trigger_agent_adjustment(state, changed_prefs)

        return state

    async def _trigger_agent_adjustment(self, state: Dict[str, Any], changed_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger adjustments based on preference changes"""
        state["memory"].pending_actions.append(f"ADJUST_TO_PREFERENCES:{json.dumps(changed_prefs)}")
        return state