from typing import TypedDict, Dict, Any, Optional, List
from pydantic import BaseModel, Field

class EnhancedConversationMemory(BaseModel):
    """Enhanced memory model for conversation state tracking"""
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
    awaiting_preference: Optional[str] = None

class ChatState(TypedDict):
    """Type definition for chat state"""
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