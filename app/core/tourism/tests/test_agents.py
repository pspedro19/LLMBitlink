import pytest
import asyncio
from typing import Dict, Any, List
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from app.core.tourism.agents.tourism_chat import TourismChat, EnhancedConversationMemory


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Simulación de conversaciones típicas
CONVERSATION_FLOWS = [
    # Flujo 1: Consulta inicial → Presupuesto → Actividades → Reserva
    {
        "name": "booking_flow",
        "messages": [
            {
                "input": "Hola, me gustaría visitar Curaçao",
                "expected_intent": "INITIAL_INQUIRY",
                "expected_sentiment": "NEUTRAL"
            },
            {
                "input": "Mi presupuesto es de 5000 USD para dos personas",
                "expected_intent": "INITIAL_INQUIRY",
                "expected_sentiment": "NEUTRAL"
            },
            {
                "input": "¿Qué actividades recomiendan para una semana?",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_sentiment": "NEUTRAL"
            },
            {
                "input": "¡Excelente! Me gustaría reservar el tour de snorkel",
                "expected_intent": "BOOKING_INTENT",
                "expected_sentiment": "POSITIVE"
            }
        ]
    },
    # Flujo 2: Queja → Resolución → Satisfacción
    {
        "name": "complaint_resolution",
        "messages": [
            {
                "input": "Tengo un problema con mi reserva actual",
                "expected_intent": "COMPLAINT",
                "expected_sentiment": "NEGATIVE"
            },
            {
                "input": "No puedo acceder a los detalles del tour",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_sentiment": "NEGATIVE"
            },
            {
                "input": "Gracias por solucionarlo tan rápido",
                "expected_intent": "INITIAL_INQUIRY",
                "expected_sentiment": "POSITIVE"
            }
        ]
    },
    # Flujo 3: Preguntas específicas → Detalles → Consideración
    {
        "name": "detailed_inquiry",
        "messages": [
            {
                "input": "¿Cuáles son las mejores playas?",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_sentiment": "NEUTRAL"
            },
            {
                "input": "¿Hay restaurantes cerca de Playa Kenepa?",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_sentiment": "NEUTRAL"
            },
            {
                "input": "¿Qué opciones de transporte hay desde el aeropuerto?",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_sentiment": "NEUTRAL"
            }
        ]
    }
]

class TestTourismChat:
    @pytest.fixture
    def chat_agent(self):
        """Configuración del agente para testing"""
        llm = ChatOpenAI(
            openai_api_key="fake-key-for-testing",
            model="gpt-4"
        )
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.load_local("curacao_docs", embeddings)
        
        return TourismChat(
            llm=llm,
            vector_store=vector_store,
            config_path="config/"
        )

    @pytest.mark.asyncio
    async def test_conversation_flows(self, chat_agent):
        """Test de flujos completos de conversación"""
        for flow in CONVERSATION_FLOWS:
            print(f"\nTesteando flujo: {flow['name']}")
            for msg in flow["messages"]:
                response = await chat_agent.invoke(
                    msg["input"],
                    session_id=f"test_{flow['name']}"
                )
                
                # Verificaciones básicas
                assert "response" in response
                assert "debug_info" in response
                assert len(response["response"]) > 0
                
                # Verificar intent y sentiment
                debug_info = response["debug_info"]
                assert debug_info["intent"] == msg["expected_intent"]
                assert debug_info["sentiment"] == msg["expected_sentiment"]
                
                print(f"Usuario: {msg['input']}")
                print(f"Asistente: {response['response']}")
                print(f"Debug: {debug_info}")

    @pytest.mark.asyncio
    async def test_preference_collection(self, chat_agent):
        """Test de recolección de preferencias"""
        conversation = [
            ("Quiero visitar Curaçao", None),
            ("Somos 4 personas", "family_size"),
            ("Tenemos 5000 USD de presupuesto", "budget"),
            ("Nos quedaremos 7 días", "days"),
            ("Sin restricciones alimenticias", "food_preferences")
        ]
        
        session_id = "test_preferences"
        for input_text, expected_pref in conversation:
            response = await chat_agent.invoke(input_text, session_id)
            
            if expected_pref:
                assert expected_pref in response["debug_info"].get("preferences", {})

    @pytest.mark.asyncio
    async def test_error_handling(self, chat_agent):
        """Test de manejo de errores"""
        error_cases = [
            "",  # Input vacío
            "?"*1000,  # Input muy largo
            "κόσμε",  # Caracteres especiales
            None  # Input nulo
        ]
        
        for error_input in error_cases:
            response = await chat_agent.invoke(error_input, "test_errors")
            assert "error" in response or "response" in response

    @pytest.mark.asyncio
    async def test_intent_transitions(self, chat_agent):
        """Test de transiciones entre intenciones"""
        conversation = [
            {
                "input": "¿Qué playas tienen?",
                "expected_intent": "SPECIFIC_QUESTION",
                "expected_stage": "exploring"
            },
            {
                "input": "¡Me encanta! ¿Cómo reservo?",
                "expected_intent": "BOOKING_INTENT",
                "expected_stage": "booking"
            },
            {
                "input": "El precio es muy alto",
                "expected_intent": "OBJECTION",
                "expected_stage": "negotiation"
            }
        ]
        
        session_id = "test_transitions"
        for msg in conversation:
            response = await chat_agent.invoke(msg["input"], session_id)
            debug_info = response["debug_info"]
            
            assert debug_info["intent"] == msg["expected_intent"]
            assert debug_info["stage"] == msg["expected_stage"]

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, chat_agent):
        """Test de tracking de métricas"""
        # Ejecutar varias interacciones
        interactions = [
            "Hola, quiero información",
            "Me interesa reservar",
            "El servicio es excelente",
            "¿Tienen tours de snorkel?"
        ]
        
        for input_text in interactions:
            await chat_agent.invoke(input_text, "test_metrics")
        
        metrics = chat_agent.metrics
        
        # Verificar conteo de interacciones
        assert metrics["total_interactions"] == len(interactions)
        
        # Verificar distribución de sentimientos
        assert sum(metrics["sentiment_distribution"].values()) == len(interactions)
        
        # Verificar distribución de intenciones
        assert sum(metrics["intent_distribution"].values()) == len(interactions)
        
        # Verificar tasa de conversión
        assert "conversion_rate" in metrics
        assert isinstance(metrics["conversion_rate"], float)

if __name__ == "__main__":
    pytest.main(["-v", "test_agents.py"])