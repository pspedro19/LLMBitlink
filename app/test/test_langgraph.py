"""
Pruebas para los componentes basados en LangGraph del sistema de recomendaciones.
"""
import pytest
from typing import Dict, Any, List
from datetime import datetime
from langchain.graphs import Neo4jGraph
from langgraph.prebuilt import ToolExecutor
from core.rag.chains.conversation_chain import ConversationChain
from core.rag.chains.recommendation_chain import RecommendationChain
from core.rag.agents import TourAgent, FoodAgent, ActivityAgent

# Fixtures
@pytest.fixture
def neo4j_graph():
    """Fixture para conexión a Neo4j"""
    return Neo4jGraph(
        url="neo4j://localhost:7687",
        username="neo4j",
        password="testpassword"
    )

@pytest.fixture
def agents(neo4j_graph):
    """Fixture para instanciar todos los agentes"""
    return {
        "tour_agent": TourAgent(neo4j_graph),
        "food_agent": FoodAgent(neo4j_graph),
        "activity_agent": ActivityAgent(neo4j_graph)
    }

@pytest.fixture
def conversation_chain(agents):
    """Fixture para ConversationChain"""
    return ConversationChain(
        tour_agent=agents["tour_agent"],
        food_agent=agents["food_agent"],
        activity_agent=agents["activity_agent"]
    )

@pytest.fixture
def recommendation_chain(agents):
    """Fixture para RecommendationChain"""
    return RecommendationChain(
        tour_agent=agents["tour_agent"],
        food_agent=agents["food_agent"],
        activity_agent=agents["activity_agent"]
    )

@pytest.fixture
def sample_conversation_state():
    """Fixture para estado de conversación inicial"""
    return {
        "messages": [],
        "context": {},
        "preferences": {
            "interests": ["history", "culture"],
            "locations": ["Willemstad"],
            "budget_per_day": 150.0,
            "trip_duration": 3
        },
        "metadata": {
            "start_time": datetime.now(),
            "status": "active"
        }
    }

# Test casos para el flujo de conversación
async def test_conversation_flow(conversation_chain, sample_conversation_state):
    """Prueba flujo completo de conversación"""
    # Mensaje inicial del usuario
    message = (
        "I want to explore the historic sites in Willemstad and try local food. "
        "My budget is $150 per day for 3 days."
    )
    
    # Procesar mensaje
    result = await conversation_chain.process_message(
        message=message,
        state=sample_conversation_state
    )
    
    # Verificar estructura de la respuesta
    assert "messages" in result
    assert len(result["messages"]) >= 2  # Mensaje usuario + respuesta
    assert "context" in result
    assert "preferences" in result
    
    # Verificar que se actualizaron las preferencias
    assert "interests" in result["preferences"]
    assert "budget_per_day" in result["preferences"]
    assert "trip_duration" in result["preferences"]
    
    # Verificar metadata
    assert "metadata" in result
    assert "processing_time" in result["metadata"]
    assert result["metadata"]["status"] == "completed"

async def test_conversation_intent_analysis(conversation_chain):
    """Prueba análisis de intenciones en la conversación"""
    messages = [
        "I want to try local restaurants in Willemstad",
        "What activities are available for adventure seekers?",
        "Can you recommend a cultural tour of the historic sites?"
    ]
    
    for message in messages:
        result = await conversation_chain.process_message(message)
        
        # Verificar detección de intenciones
        assert "current_intent" in result
        assert result["current_intent"] in ["food", "activity", "tour"]
        
        # Verificar activación del agente correcto
        assert "active_agent" in result
        assert result["active_agent"] == result["current_intent"]

async def test_conversation_context_maintenance(
    conversation_chain,
    sample_conversation_state
):
    """Prueba mantenimiento del contexto en la conversación"""
    # Primera interacción
    result1 = await conversation_chain.process_message(
        "I'm interested in historical sites in Willemstad",
        state=sample_conversation_state
    )
    
    # Segunda interacción que hace referencia al contexto anterior
    result2 = await conversation_chain.process_message(
        "Are there any good restaurants nearby?",
        state=result1
    )
    
    # Verificar que el contexto se mantiene
    assert "Willemstad" in result2["context"].get("current_location", "")
    assert "historical sites" in str(result2["context"].get("interests", []))

# Test casos para la cadena de recomendaciones
async def test_recommendation_chain_execution(
    recommendation_chain,
    sample_conversation_state
):
    """Prueba ejecución completa de la cadena de recomendaciones"""
    query = (
        "Looking for historic sites and local restaurants in Willemstad. "
        "Budget is $150 per day for 3 days."
    )
    
    result = await recommendation_chain.run(
        query=query,
        preferences=sample_conversation_state["preferences"]
    )
    
    # Verificar estructura del resultado
    assert isinstance(result.recommendations, list)
    assert len(result.recommendations) > 0
    assert result.conversation_flow
    assert result.detected_intents
    assert result.detected_locations
    assert result.metrics
    
    # Verificar calidad de las recomendaciones
    for rec in result.recommendations:
        assert "name" in rec
        assert "type" in rec
        assert "description" in rec
        assert "location" in rec
        assert "cost" in rec
        assert rec["cost"] <= sample_conversation_state["preferences"]["budget_per_day"]

async def test_recommendation_ranking(recommendation_chain):
    """Prueba ranking y filtrado de recomendaciones"""
    query = "Show me the best rated attractions in Willemstad"
    preferences = {
        "interests": ["sightseeing"],
        "locations": ["Willemstad"],
        "min_rating": 4.0
    }
    
    result = await recommendation_chain.run(query, preferences)
    
    # Verificar orden por rating
    ratings = [rec.get("rating", 0) for rec in result.recommendations]
    assert ratings == sorted(ratings, reverse=True)
    
    # Verificar filtro de rating mínimo
    assert all(rec.get("rating", 0) >= 4.0 for rec in result.recommendations)

async def test_recommendation_diversity(recommendation_chain):
    """Prueba diversidad en las recomendaciones"""
    query = "I want to experience different activities in Willemstad"
    preferences = {
        "interests": ["culture", "food", "adventure"],
        "locations": ["Willemstad"],
        "budget_per_day": 200
    }
    
    result = await recommendation_chain.run(query, preferences)
    
    # Verificar diversidad de tipos
    types = set(rec.get("type") for rec in result.recommendations)
    assert len(types) >= 3  # Al menos 3 tipos diferentes
    
    # Verificar diversidad de ubicaciones
    locations = set(rec.get("location") for rec in result.recommendations)
    assert len(locations) >= 2  # Al menos 2 ubicaciones diferentes

# Test casos para el procesamiento paralelo
async def test_parallel_recommendation_processing(
    recommendation_chain,
    sample_conversation_state
):
    """Prueba procesamiento paralelo de recomendaciones"""
    import asyncio
    
    async def get_recommendations(query: str):
        return await recommendation_chain.run(
            query=query,
            preferences=sample_conversation_state["preferences"]
        )
    
    # Ejecutar múltiples consultas en paralelo
    queries = [
        "Historical sites in Willemstad",
        "Local restaurants near downtown",
        "Cultural activities and tours"
    ]
    
    tasks = [get_recommendations(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    # Verificar resultados
    assert len(results) == len(queries)
    assert all(isinstance(result.recommendations, list) for result in results)
    assert all(len(result.recommendations) > 0 for result in results)

# Test casos para el manejo de errores
async def test_conversation_error_handling(conversation_chain):
    """Prueba manejo de errores en la conversación"""
    # Probar con mensaje vacío
    result = await conversation_chain.process_message("")
    assert result["metadata"]["status"] == "error"
    
    # Probar con preferencias inválidas
    result = await conversation_chain.process_message(
        "I want recommendations",
        state={"preferences": {"budget_per_day": -100}}
    )
    assert "error" in result["metadata"]

async def test_recommendation_error_handling(recommendation_chain):
    """Prueba manejo de errores en recomendaciones"""
    # Probar con query vacío
    with pytest.raises(ValueError):
        await recommendation_chain.run("", {})
    
    # Probar con preferencias inválidas
    result = await recommendation_chain.run(
        "Show me attractions",
        {"budget_per_day": -100}
    )
    assert len(result.recommendations) == 0
    assert result.metrics.get("error_rate", 0) > 0

# Test casos para métricas y logging
def test_conversation_metrics(conversation_chain, sample_conversation_state):
    """Prueba recopilación de métricas de conversación"""
    async def check_metrics():
        result = await conversation_chain.process_message(
            "Tell me about historical sites",
            state=sample_conversation_state
        )
        
        assert "metadata" in result
        assert "processing_time" in result["metadata"]
        assert "intent_confidence" in result["metadata"]
        assert "recommendation_quality" in result["metadata"]
    
    asyncio.run(check_metrics())

def test_recommendation_metrics(recommendation_chain):
    """Prueba recopilación de métricas de recomendaciones"""
    async def check_metrics():
        result = await recommendation_chain.run(
            "Show me the best attractions",
            {"interests": ["sightseeing"]}
        )
        
        assert result.metrics
        assert "preference_match_score" in result.metrics
        assert "diversity_score" in result.metrics
        assert "coverage_score" in result.metrics
    
    asyncio.run(check_metrics())

if __name__ == "__main__":
    pytest.main([__file__])