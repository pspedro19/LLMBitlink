from app.utils.logger import get_logger
from .types import ChatState

class BaseAgent:
    """Base class for all tourism agents"""
    
    def __init__(self):
        self.logger = get_logger(__name__)

    def _initialize_state(self, state: ChatState) -> ChatState:
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

    async def process(self, state: ChatState) -> ChatState:
        """Process method to be implemented by each agent"""
        raise NotImplementedError()

    def _debug_log(self, message: str, state: ChatState = None) -> None:
        """Helper method for debug logging"""
        if state:
            self.logger.debug(f"{message} | State: {state.get('stage', 'unknown')}")
        else:
            self.logger.debug(message)