import pytest
import pytest_asyncio
import os
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from langchain_openai import ChatOpenAI

with patch('prometheus_client.exposition.start_http_server'):
    pass 

# Add project root to PYTHONPATH
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)
sys.path.insert(0, project_root)

# Import from core tourism types and agents
from app.core.tourism.agents.types import EnhancedConversationMemory, ChatState
from app.core.tourism.agents import TourismOrchestrator
from app.core.rag.retriever import RAGRetriever
from app.core.rag.document_processor import DocumentProcessor

class MockRAGRetriever:
    async def retrieve_context(self, query: str) -> str:
        """Mock context retrieval with predefined responses"""
        responses = {
            "playa": "Tenemos hermosas playas con aguas cristalinas.",
            "hotel": "Ofrecemos hoteles de lujo y boutique.",
            "restaurante": "Disponemos de excelentes restaurantes con vista al mar.",
            "actividad": "Múltiples actividades disponibles para toda la familia.",
            "niños": "Actividades perfectas para familias: playa, snorkel y acuario."
        }
        
        for key, response in responses.items():
            if key in query.lower():
                return response
        return "Curaçao ofrece una experiencia única para todos los visitantes."

@pytest_asyncio.fixture
async def orchestrator_agent():
    """Fixture that initializes the orchestrator with mocked dependencies"""
    # Mock RAG Retriever
    mock_rag = MockRAGRetriever()
    
    # Mock Document Processor
    mock_doc_processor = Mock(
        spec=DocumentProcessor,
        process=Mock(return_value=[])
    )
    
    # Initialize LLM with test config
    llm = ChatOpenAI(
        openai_api_key="fake-key-for-testing",
        model="gpt-4"
    )

    # Create orchestrator instance
    agent = TourismOrchestrator(
        llm=llm,
        retriever=mock_rag,
        document_processor=mock_doc_processor,
        config_path="app/core/tourism/config/"
    )

    # Initialize conversation memory
    agent.conversation_memory = {}

    return agent

@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture that provides the path to test data directory."""
    return Path(__file__).parent / "test_data"