import pytest
import asyncio
from typing import Dict, Any, List

# Escenarios completos de conversación
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
            },
            {
                "user": "Planeamos quedarnos 8 días",
                "expected": {
                    "intent": "PREFERENCE_PROVIDING",
                    "sentiment": "NEUTRAL",
                    "should_recommend": True
                }
            },
            {
                "user": "¿Qué actividades son buenas para niños?",
                "expected": {
                    "intent": "SPECIFIC_QUESTION",
                    "sentiment": "NEUTRAL",
                    "should_include": ["playa", "snorkel", "acuario"]
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
                "user": "Somos solo mi esposa y yo",
                "expected": {
                    "intent": "PREFERENCE_PROVIDING",
                    "sentiment": "NEUTRAL",
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
    },
    {
        "name": "queja_y_resolucion",
        "description": "Cliente con queja sobre reserva",
        "conversation": [
            {
                "user": "Mi reserva del tour de snorkel tiene un error",
                "expected": {
                    "intent": "COMPLAINT",
                    "sentiment": "NEGATIVE",
                    "should_apologize": True
                }
            },
            {
                "user": "La fecha está mal",
                "expected": {
                    "intent": "COMPLAINT",
                    "sentiment": "NEGATIVE",
                    "should_offer_solution": True
                }
            },
            {
                "user": "Gracias por arreglarlo",
                "expected": {
                    "intent": "GRATITUDE",
                    "sentiment": "POSITIVE",
                    "should_include": ["placer", "servicio"]
                }
            }
        ]
    }
]

@pytest.mark.asyncio
async def test_conversation_scenarios(chat_agent):
    """Test de escenarios completos de conversación"""
    for scenario in CONVERSATION_SCENARIOS:
        print(f"\nEscenario: {scenario['name']}")
        print(f"Descripción: {scenario['description']}")
        
        session_id = f"test_{scenario['name']}"
        
        for step in scenario["conversation"]:
            response = await chat_agent.invoke(step["user"], session_id)
            
            # Verificaciones básicas
            assert response["response"]
            assert "debug_info" in response
            
            # Verificar intent y sentiment esperados
            if "intent" in step["expected"]:
                assert response["debug_info"]["intent"] == step["expected"]["intent"]
            if "sentiment" in step["expected"]:
                assert response["debug_info"]["sentiment"] == step["expected"]["sentiment"]
            
            # Verificar preferencias
            if step["expected"].get("should_ask_preference"):
                assert "preference" in response["debug_info"]
                if "preference" in step["expected"]:
                    assert response["debug_info"]["preference"] == step["expected"]["preference"]
            
            # Verificar contenido específico
            if step["expected"].get("should_include"):
                response_lower = response["response"].lower()
                for term in step["expected"]["should_include"]:
                    assert term.lower() in response_lower
            
            # Verificar tono apropiado
            if step["expected"].get("should_apologize"):
                assert any(word in response["response"].lower() 
                         for word in ["disculp", "lament", "sentimos"])
            
            if step["expected"].get("should_offer_solution"):
                assert any(word in response["response"].lower() 
                         for word in ["podem", "solución", "alternativ"])
            
            print(f"\nUsuario: {step['user']}")
            print(f"Asistente: {response['response']}")
            print(f"Debug: {response['debug_info']}")

if __name__ == "__main__":
    pytest.main(["-v", "test_tourism_conversations.py"])