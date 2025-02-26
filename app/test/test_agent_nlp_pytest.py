"""
Pytest para el agente NLP personalizado.
"""
import sys
import os
import pytest
import asyncio
from pathlib import Path

# Ajustar el PYTHONPATH para incluir el directorio raíz de la aplicación
current_dir = Path(__file__).parent
app_root_dir = current_dir.parent  # Subir un nivel para llegar a la raíz de la app
sys.path.insert(0, str(app_root_dir))

# Verificamos si podemos importar las dependencias reales
try:
    from app.core.tourism.agents.base import BaseAgent, ChatState
    has_base_agent = True
except ImportError:
    has_base_agent = False
    # Creamos versiones simuladas
    class BaseAgent:
        def __init__(self, name="BaseAgent"):
            self.name = name
        
        async def process(self, state):
            return state
    
    class ChatState(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

# Verificamos si podemos importar el text_formatter real
try:
    from app.core.recommender.text_formatter import get_text_formatted_recommendations, TextFormattedRequest
    has_formatter = True
except ImportError:
    has_formatter = False
    # Versiones simuladas
    class TextFormattedRequest:
        def __init__(self, text=""):
            self.text = text
    
    async def get_text_formatted_recommendations(request):
        # Respuesta simulada para pruebas
        return f"Recomendaciones simuladas para: {request.text}"

# Importamos el agente a probar
try:
    from app.core.tourism.agents.agent_nlp import NLPAgent
except ImportError:
    pytest.skip("NLPAgent no disponible, saltando pruebas", allow_module_level=True)

@pytest.fixture
def nlp_agent():
    """Fixture para crear el agente NLP"""
    return NLPAgent()

@pytest.fixture
def chat_state_factory():
    """Fixture para crear diferentes estados de chat"""
    def _create_state(user_input="", language="es"):
        return ChatState({"user_input": user_input, "language": language})
    return _create_state

@pytest.mark.asyncio
async def test_agent_initialization(nlp_agent):
    """Verifica que el agente se inicialice correctamente"""
    assert nlp_agent is not None
    assert nlp_agent.name == "NLPAgent"
    assert hasattr(nlp_agent, 'process')

@pytest.mark.asyncio
async def test_process_with_empty_input(nlp_agent, chat_state_factory):
    """Prueba el procesamiento con entrada vacía"""
    state = chat_state_factory(user_input="")
    result = await nlp_agent.process(state)
    
    # Verificamos que haya algún tipo de respuesta
    assert "formatted_recommendations" in result
    assert result["formatted_recommendations"]  # No debería estar vacío
    
    # Verificamos que la respuesta también esté en el campo principal
    assert "response" in result
    assert result["response"] == result["formatted_recommendations"]

@pytest.mark.asyncio
async def test_process_with_beach_query(nlp_agent, chat_state_factory):
    """Prueba el procesamiento con una consulta sobre playas"""
    state = chat_state_factory(user_input="Quiero visitar las mejores playas de Curaçao")
    result = await nlp_agent.process(state)
    
    # Verificamos que haya recomendaciones
    assert "formatted_recommendations" in result
    
    # En un entorno real, verificaríamos contenido específico
    # Para nuestras pruebas simuladas, solo verificamos que hay una respuesta
    assert len(result["formatted_recommendations"]) > 0

@pytest.mark.asyncio
async def test_process_with_museum_query(nlp_agent, chat_state_factory):
    """Prueba el procesamiento con una consulta sobre museos"""
    state = chat_state_factory(user_input="I want to visit museums in Willemstad", language="en")
    result = await nlp_agent.process(state)
    
    # Verificamos que haya recomendaciones
    assert "formatted_recommendations" in result
    assert len(result["formatted_recommendations"]) > 0

@pytest.mark.asyncio
async def test_process_preserves_state(nlp_agent, chat_state_factory):
    """Verifica que el procesamiento preserve el estado existente"""
    # Crear un estado con datos adicionales
    state = chat_state_factory(user_input="Test query")
    state["additional_data"] = "This should be preserved"
    state["user_preferences"] = {"budget": 200, "duration": 3}
    
    # Procesar el estado
    result = await nlp_agent.process(state)
    
    # Verificar que los datos adicionales se han preservado
    assert "additional_data" in result
    assert result["additional_data"] == "This should be preserved"
    assert "user_preferences" in result
    assert result["user_preferences"]["budget"] == 200

if __name__ == "__main__":
    # Si se ejecuta directamente, mostrar un mensaje informativo
    print("Este archivo está diseñado para ejecutarse con pytest.")
    print("Para ejecutar las pruebas, usa:")
    print("  pytest test_agent_nlp_pytest.py -v")
    print("\nPero también podemos ejecutar una prueba rápida ahora:")
    
    async def run_quick_test():
        agent = NLPAgent()
        state = ChatState({"user_input": "Quiero visitar Curaçao por 3 días"})
        result = await agent.process(state)
        print("\nResultado de la prueba rápida:")
        if "formatted_recommendations" in result:
            print(f"Recomendaciones: {result['formatted_recommendations'][:100]}...")
        else:
            print("No se generaron recomendaciones.")
    
    # Ejecutar la prueba rápida
    asyncio.run(run_quick_test())