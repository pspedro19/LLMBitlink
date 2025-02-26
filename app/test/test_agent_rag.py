import pytest
from app.core.tourism.agents.agent_rag import RAGAgent

# Fixture para inicializar el agente
@pytest.fixture
def rag_agent():
    return RAGAgent()

# Prueba de consulta en inglés sobre Curazao
@pytest.mark.asyncio
async def test_process_query_english(rag_agent):
    query = "What are the top tourist attractions in Curacao?"
    
    # Procesar la consulta real con el agente RAG
    result = await rag_agent.process_query(query)
    
    # Verificar que la respuesta contiene la consulta y la respuesta generada
    assert result["query"] == query
    assert "response" in result and len(result["response"]) > 0  # La respuesta debe ser no vacía
    assert len(result["documents"]) == 0  # No recuperar documentos en esta prueba

# Prueba de consulta en español sobre Curazao
@pytest.mark.asyncio
async def test_process_query_spanish(rag_agent):
    query = "¿Cuáles son las principales atracciones turísticas de Curazao?"
    
    # Procesar la consulta real con el agente RAG
    result = await rag_agent.process_query(query)
    
    # Verificar que la respuesta contiene la consulta y la respuesta generada
    assert result["query"] == query
    assert "response" in result and len(result["response"]) > 0  # La respuesta debe ser no vacía
    assert len(result["documents"]) == 0  # No recuperar documentos en esta prueba

# Prueba de consulta sobre playas en Curazao
@pytest.mark.asyncio
async def test_process_query_beaches(rag_agent):
    query = "What are the best beaches in Curacao?"
    
    # Procesar la consulta real con el agente RAG
    result = await rag_agent.process_query(query)
    
    # Verificar que la respuesta contiene la consulta y la respuesta generada
    assert result["query"] == query
    assert "response" in result and len(result["response"]) > 0  # La respuesta debe ser no vacía
    assert len(result["documents"]) == 0  # No recuperar documentos en esta prueba

# Prueba de consulta sobre clima en Curazao
@pytest.mark.asyncio
async def test_process_query_weather(rag_agent):
    query = "What is the weather like in Curacao?"
    
    # Procesar la consulta real con el agente RAG
    result = await rag_agent.process_query(query)
    
    # Verificar que la respuesta contiene la consulta y la respuesta generada
    assert result["query"] == query
    assert "response" in result and len(result["response"]) > 0  # La respuesta debe ser no vacía
    assert len(result["documents"]) == 0  # No recuperar documentos en esta prueba

# Prueba de consulta en español sobre clima en Curazao
@pytest.mark.asyncio
async def test_process_query_clima(rag_agent):
    query = "¿Cómo es el clima en Curazao?"
    
    # Procesar la consulta real con el agente RAG
    result = await rag_agent.process_query(query)
    
    # Verificar que la respuesta contiene la consulta y la respuesta generada
    assert result["query"] == query
    assert "response" in result and len(result["response"]) > 0  # La respuesta debe ser no vacía
    assert len(result["documents"]) == 0  # No recuperar documentos en esta prueba
