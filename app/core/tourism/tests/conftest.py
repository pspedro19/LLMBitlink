import pytest
import os
import sys

# Get the root directory (LLMBitlink folder)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the project root to PYTHONPATH
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture that provides the path to test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def db():
    """Fixture for database connection."""
    return Database()

@pytest.fixture(scope="session")
def faiss_manager():
    """Fixture for FAISS manager."""
    return FAISSManager()

@pytest.fixture(scope="session")
def embedding_manager():
    """Fixture for embedding manager."""
    return EmbeddingManager()