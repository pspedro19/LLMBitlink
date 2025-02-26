import pytest
from types import SimpleNamespace
from app.core.tourism.agents.agent_rag import RAGAgent

# Como el agente RAG utiliza lógica interna basada en el input y el idioma,
# podemos probar distintos casos.

@pytest.mark.asyncio
async def test_rag_agent_english_default():
    # Creamos un estado dummy para el caso en inglés
    state = {
        "user_input": "I want to visit the beach",
        "language": "en",
        "metrics": {"performance_metrics": {"rag_hits": 0, "rag_misses": 0}},
        "memory": SimpleNamespace(preferences={}),
    }
    # Para este test no usamos el retriever, pero lo pasamos como dummy
    dummy_retriever = object()
    agent = RAGAgent(dummy_retriever)
    
    updated_state = await agent.process(state)
    
    # En la rama en inglés, se espera que se detecte "beach" en el input
    # y se retorne el contexto asociado. Según la lógica:
    # "Our beaches feature crystal clear waters perfect for swimming and snorkeling."
    # Si no se detecta, se usaría el fallback.
    context = updated_state.get("rag_context", "")
    assert context != ""
    assert "beach" not in context.lower() or "swimming" in context.lower()
    # Verificar que las métricas se actualicen (rag_hits aumente)
    assert updated_state["metrics"]["performance_metrics"]["rag_hits"] >= 1

@pytest.mark.asyncio
async def test_rag_agent_spanish_ninos():
    # Estado para el caso en español con mención de "niños"
    state = {
        "user_input": "Quiero visitar lugares para niños y familia",
        "language": "es",
        "metrics": {"performance_metrics": {"rag_hits": 0, "rag_misses": 0}},
        "memory": SimpleNamespace(preferences={}),
    }
    dummy_retriever = object()
    agent = RAGAgent(dummy_retriever)
    
    updated_state = await agent.process(state)
    
    # Para "niños" se espera el contexto fijo
    expected_context = ("Contamos con playas de ensueño, ideales para familias. "
                        "Disfruta de snorkel y visita nuestro acuario.")
    assert updated_state.get("rag_context") == expected_context
    # Verificar actualización de métricas
    assert updated_state["metrics"]["performance_metrics"]["rag_hits"] >= 1
