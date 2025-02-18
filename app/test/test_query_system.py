# test/test_query_system.py
import asyncio
import logging
from pathlib import Path
import pytest
import pytest_asyncio
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.rag.config import Config
from core.rag.faiss_manager import PersistentFAISSManager
from core.rag.sync_manager import SyncManager
from core.query_system import QuerySystem  # Ajusta la ruta según tu estructura de proyecto

@pytest_asyncio.fixture(scope="function")
async def setup_sync_env():
    """
    Fixture que prepara el entorno de sincronización:
      - Crea (o limpia) la carpeta de índices.
      - Inicializa la configuración, el FAISSManager y el SyncManager.
    """
    config = Config()
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)
    config.INDICES_DIR = indices_dir

    # Si existe un índice previo, se elimina.
    faiss_index_path = indices_dir / "faiss_index.bin"
    if faiss_index_path.exists():
        faiss_index_path.unlink()
        logger.info(f"Archivo existente {faiss_index_path} eliminado para crear uno nuevo.")

    faiss_manager = PersistentFAISSManager(config)
    sync_manager = SyncManager(config, faiss_manager)
    yield sync_manager

@pytest_asyncio.fixture(scope="function")
async def dummy_document(setup_sync_env):
    """
    Fixture que asegura que existe un documento dummy
    (necesario si la tabla chunks requiere un doc_id válido).
    """
    dummy_doc_id = "00000000-0000-0000-0000-000000000010"
    pool = await setup_sync_env._get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (doc_id, title)
            VALUES ($1, $2)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            dummy_doc_id, "Dummy Document"
        )
    return dummy_doc_id

@pytest_asyncio.fixture(scope="function")
async def query_system_fixture(setup_sync_env):
    """
    Inicializa el QuerySystem y fuerza la indexación
    de todos los chunks reales que mencionan Curazao.
    """
    qs = QuerySystem(config=setup_sync_env.config)
    pool = await qs.sync_manager._get_pool()

    async with pool.acquire() as conn:
        # Marca para indexar todos los chunks que mencionan 'Curacao' o 'Curaçao'
        await conn.execute("""
            UPDATE chunks
            SET needs_indexing = TRUE
        """)

    # Sincroniza para que FAISS indexe la data real
    await qs.synchronize()
    return qs

@pytest.mark.asyncio
async def test_query_system(query_system_fixture):
    """
    Realiza una consulta en lenguaje natural sobre Curazao
    y verifica la estructura de la respuesta.
    """
    qs = query_system_fixture
    question = "tourist information about curazao?"
    result = await qs.query(question, k=1)

    # Muestra la entrada y la salida
    print("Entrada (Query):", result["question"])
    print("Salida (Respuesta):", result["answer"])
    print("Contexto recuperado:", result["context"])
    print("Detalles de búsqueda - Distancias:", result["distances"], "Indices:", result["indices"])

    # Validaciones mínimas
    assert "question" in result, "La respuesta debe incluir la pregunta original."
    assert "context" in result, "La respuesta debe incluir el contexto recuperado."
    assert "answer" in result, "La respuesta debe incluir la respuesta generada."
    assert "distances" in result, "La respuesta debe incluir las distancias de la búsqueda en FAISS."
    assert "indices" in result, "La respuesta debe incluir los índices de los chunks encontrados."
    assert result["context"], "El contexto recuperado no debe estar vacío."
    assert isinstance(result["answer"], dict), "La respuesta generada debe ser un diccionario."
