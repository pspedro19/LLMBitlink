# app/core/tourism/tests/test_tourism_chat_cli.py

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Get the absolute path to the config directory
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = str(BASE_DIR / "config")

from app.core.tourism.agents.orchestrator import TourismOrchestrator

class MockConfig:
    """Mock config for testing"""
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = "text-embedding-3-small"
        self.collection_name = "curacao_tourism"

class MockDocumentProcessor:
    """Mock document processor for testing"""
    def __init__(self, *args, **kwargs):
        pass

    async def process_text(self, text: str):
        return [text]  # Return text as single chunk

    async def process_documents(self, documents: list):
        return documents  # Return documents as is

class MockRAGRetriever:
    """Mock RAG retriever for testing without database dependency"""
    def __init__(self, *args, **kwargs):
        pass

    async def get_relevant_chunks(self, query: str, *args, **kwargs):
        return []  # Return empty list for now

    async def add_texts(self, texts: list, metadata: list = None):
        pass  # Do nothing for now

# Load environment variables
load_dotenv()

async def initialize_chat():
    # Initialize OpenAI client
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview"
    )

    # Create mock config
    config = MockConfig()

    # Initialize components with mocks
    retriever = MockRAGRetriever(config=config)
    document_processor = MockDocumentProcessor(config=config)
    
    # Initialize orchestrator with the correct config path
    orchestrator = TourismOrchestrator(
        llm=llm,
        retriever=retriever,
        document_processor=document_processor,
        config_path=CONFIG_PATH
    )
    
    return orchestrator

async def chat_loop():
    print("\nInitializing Curaçao Tourism Chat...")
    try:
        orchestrator = await initialize_chat()
        session_id = "test_session"
        
        print("\nWelcome to Curaçao Tourism Chat!")
        print("Type 'quit' to exit")
        print("-" * 50)

        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break

            # Process through orchestrator
            response = await orchestrator.invoke(
                user_input=user_input,
                session_id=session_id
            )

            # Print response
            print("\nAssistant:", response["response"])
            
            # Print debug info
            if "debug_info" in response:
                print("\nDebug Info:")
                print(f"Intent: {response['debug_info'].get('intent')}")
                print(f"Sentiment: {response['debug_info'].get('sentiment')}")
                print(f"Stage: {response['debug_info'].get('stage')}")
                if response['debug_info'].get('preferences'):
                    print("Preferences:", response['debug_info']['preferences'])

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your configuration and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(chat_loop())