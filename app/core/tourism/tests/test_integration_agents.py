import pytest
from typing import Dict, Any
from app.core.tourism.agents.agent_context import ContextAnalysisAgent
from app.core.tourism.agents.agent_intent import IntentClassificationAgent
from app.core.tourism.agents.agent_sentiment import SentimentAnalysisAgent
from app.core.tourism.agents.agent_nlp import NLPAgent
from app.core.tourism.agents.memory import EnhancedConversationMemory

@pytest.fixture
def base_state():
    """Create a base state for testing"""
    return {
        "user_input": "",
        "memory": EnhancedConversationMemory(),
        "current_intent": None,
        "sentiment": None,
        "recommendations": [],
        "rag_context": None,
        "response": None,
        "metrics": {
            "total_interactions": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
            "intent_distribution": {}
        },
        "stage": "initial",
        "preferences": {},
        "language": "es",
        "awaiting_preference": None,
        "preferences_complete": False
    }

@pytest.mark.asyncio
async def test_context_agent(base_state):
    """Test context analysis agent"""
    agent = ContextAnalysisAgent()
    
    # Test entity extraction
    base_state["user_input"] = "Quiero visitar la playa y un restaurante"
    result = await agent.process(base_state)
    
    assert "playa" in result["memory"].context_graph
    assert "restaurante" in result["memory"].context_graph
    assert result["memory"].context_graph["playa"]["mentioned"] == 1

@pytest.mark.asyncio
async def test_intent_agent(base_state):
    """Test intent classification agent"""
    agent = IntentClassificationAgent()
    
    test_cases = [
        ("Quiero hacer una reserva", "BOOKING_INTENT", "booking"),
        ("Gracias por la ayuda", "GRATITUDE", "closing"),
        ("El precio es muy alto", "OBJECTION", "negotiation"),
        ("¿Qué actividades tienen?", "SPECIFIC_QUESTION", "exploring")
    ]
    
    for input_text, expected_intent, expected_stage in test_cases:
        base_state["user_input"] = input_text
        result = await agent.process(base_state)
        
        assert result["current_intent"]["primary"] == expected_intent
        assert result["stage"] == expected_stage

@pytest.mark.asyncio
async def test_sentiment_agent(base_state):
    """Test sentiment analysis agent"""
    agent = SentimentAnalysisAgent()
    
    test_cases = [
        ("¡Me encanta la idea!", "POSITIVE"),
        ("Esto es terrible", "NEGATIVE"),
        ("¿Qué actividades tienen?", "NEUTRAL"),
        ("El servicio es excelente", "POSITIVE"),
        ("Hay un problema", "NEGATIVE")
    ]
    
    for input_text, expected_sentiment in test_cases:
        base_state["user_input"] = input_text
        result = await agent.process(base_state)
        assert result["sentiment"] == expected_sentiment

@pytest.mark.asyncio
async def test_nlp_agent(base_state):
    """Test NLP processing agent"""
    agent = NLPAgent()
    
    # Test preference extraction
    test_cases = [
        (
            "Somos 4 personas con presupuesto de 5000 USD por 7 días",
            {
                "family_size": 4,
                "budget": 5000,
                "days": 7
            }
        ),
        (
            "Queremos estar 3 días, somos 2 personas",
            {
                "family_size": 2,
                "days": 3
            }
        )
    ]
    
    for input_text, expected_prefs in test_cases:
        base_state["user_input"] = input_text
        result = await agent.process(base_state)
        
        for key, value in expected_prefs.items():
            assert result["preferences"].get(key) == value

@pytest.mark.asyncio
async def test_preference_validation(base_state):
    """Test preference validation in NLP agent"""
    agent = NLPAgent()
    
    # Test invalid preferences
    test_cases = [
        ("Somos -1 personas", "family_size"),  # Negative number
        ("Presupuesto de 0 USD", "budget"),    # Zero value
        ("Queremos estar 0 días", "days")      # Zero value
    ]
    
    for input_text, pref_key in test_cases:
        base_state["user_input"] = input_text
        result = await agent.process(base_state)
        assert pref_key not in result["preferences"]