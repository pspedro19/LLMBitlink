# app/core/tourism/agents/base.py
from app.utils.logger import get_logger
from .types import ChatState
from typing import Dict, Any, Optional

class BaseAgent:
    """Base class for all tourism agents"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            name (str): Agent name
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        self.name = name
        self.config = config or {}
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
            
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data.
        Should be implemented by subclasses.
        
        Args:
            data (Dict[str, Any]): Input data
            
        Returns:
            Dict[str, Any]: Processed result
        """
        raise NotImplementedError("Subclasses must implement process method")