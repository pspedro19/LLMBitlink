import pytest
import pytest_asyncio
from typing import Dict, Any

# Test scenarios from original test file
CONVERSATION_SCENARIOS = [
    {
        "name": "familia_primera_vez",
        "description": "Familia planeando su primera visita",
        "conversation": [
            {
                "user": "Hola, queremos visitar Curaçao en familia",
                "expected": {
                    "intent": "INITIAL_INQUIRY",
                    "sentiment": "NEUTRAL",
                    "should_ask_preference": True,
                    "preference": "family_size"
                }
            },
            {
                "user": "Somos 2 adultos y 2 niños",
                "expected": {
                    "intent": "PREFERENCE_PROVIDING",
                    "sentiment": "NEUTRAL",
                    "should_ask_preference": True,
                    "preference": "budget"
                }
            },
            {
                "user": "Tenemos un presupuesto de 6000 USD",
                "expected": {
                    "intent": "PREFERENCE_PROVIDING",
                    "sentiment": "NEUTRAL",
                    "should_ask_preference": True,
                    "preference": "days"
                }
            }
        ]
    },
    {
        "name": "pareja_luna_miel",
        "description": "Pareja en luna de miel",
        "conversation": [
            {
                "user": "Busco algo romántico para luna de miel",
                "expected": {
                    "intent": "INITIAL_INQUIRY",
                    "sentiment": "POSITIVE",
                    "should_ask_preference": True
                }
            },
            {
                "user": "¿Qué restaurantes románticos recomiendan?",
                "expected": {
                    "intent": "SPECIFIC_QUESTION",
                    "sentiment": "NEUTRAL",
                    "should_include": ["restaurante", "cena", "vista"]
                }
            }
        ]
    }
]

@pytest.mark.asyncio
async def test_conversation_scenarios(orchestrator_agent):
    """Test complete conversation scenarios"""
    for scenario in CONVERSATION_SCENARIOS:
        print(f"\nEscenario: {scenario['name']}")
        print(f"Descripción: {scenario['description']}")
        
        session_id = f"test_{scenario['name']}"
        
        for step in scenario["conversation"]:
            response = await orchestrator_agent.invoke(step["user"], session_id)
            
            # Basic verifications
            assert response["response"]
            assert "debug_info" in response
            
            # Verify intent and sentiment
            if "intent" in step["expected"]:
                assert response["debug_info"]["intent"] == step["expected"]["intent"]
            if "sentiment" in step["expected"]:
                assert response["debug_info"]["sentiment"] == step["expected"]["sentiment"]
            
            # Verify preferences
            if step["expected"].get("should_ask_preference"):
                assert response["debug_info"]["awaiting_preference"] is not None
                if "preference" in step["expected"]:
                    assert response["debug_info"]["awaiting_preference"] == step["expected"]["preference"]
            
            # Verify specific content
            if step["expected"].get("should_include"):
                response_lower = response["response"].lower()
                for term in step["expected"]["should_include"]:
                    assert term.lower() in response_lower
            
            print(f"\nUsuario: {step['user']}")
            print(f"Asistente: {response['response']}")
            print(f"Debug: {response['debug_info']}")

@pytest.mark.asyncio
async def test_preference_handling(orchestrator_agent):
    """Test preference collection and management"""
    session_id = "test_preferences"
    
    # Test family size preference
    response = await orchestrator_agent.invoke("Somos 4 personas", session_id)
    assert "family_size" in response["debug_info"]["preferences"]
    assert response["debug_info"]["preferences"]["family_size"] == 4
    
    # Test budget preference
    response = await orchestrator_agent.invoke("Nuestro presupuesto es 5000 USD", session_id)
    assert "budget" in response["debug_info"]["preferences"]
    assert response["debug_info"]["preferences"]["budget"] == 5000
    
    # Test days preference
    response = await orchestrator_agent.invoke("Queremos quedarnos 7 días", session_id)
    assert "days" in response["debug_info"]["preferences"]
    assert response["debug_info"]["preferences"]["days"] == 7

@pytest.mark.asyncio
async def test_language_detection(orchestrator_agent):
    """Test automatic language detection"""
    # Test Spanish detection
    response = await orchestrator_agent.invoke(
        "Hola, quiero información sobre playas", 
        "test_lang_es"
    )
    assert response["debug_info"]["language"] == "es"
    
    # Test English fallback
    response = await orchestrator_agent.invoke(
        "Hi, I want information about beaches", 
        "test_lang_en"
    )
    assert response["debug_info"]["language"] == "en"

@pytest.mark.asyncio
async def test_sentiment_analysis(orchestrator_agent):
    """Test sentiment analysis across different inputs"""
    test_cases = [
        ("¡Me encanta la idea!", "POSITIVE"),
        ("Esto es terrible", "NEGATIVE"),
        ("¿Qué actividades tienen?", "NEUTRAL"),
        ("El servicio es excelente", "POSITIVE"),
        ("Hay un problema con mi reserva", "NEGATIVE")
    ]
    
    for input_text, expected_sentiment in test_cases:
        response = await orchestrator_agent.invoke(input_text, "test_sentiment")
        assert response["debug_info"]["sentiment"] == expected_sentiment

@pytest.mark.asyncio
async def test_error_handling(orchestrator_agent):
    """Test error handling capabilities"""
    error_cases = [
        "",  # Empty input
        "?"*1000,  # Very long input
        "κόσμε",  # Special characters
        None  # None input
    ]
    
    for error_input in error_cases:
        response = await orchestrator_agent.invoke(error_input, "test_errors")
        assert "error" in response or "response" in response