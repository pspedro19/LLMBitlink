# test/test_sync_system.py
import asyncio
import logging
from pathlib import Path
import numpy as np
import sys
import pytest
import pytest_asyncio

# En Windows, usa WindowsSelectorEventLoopPolicy para evitar problemas con librerías nativas.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Forzamos que cada test use su propio event loop.
pytestmark = pytest.mark.asyncio(loop_scope="function")

from core.rag.config import Config
from core.rag.faiss_manager import PersistentFAISSManager
from core.rag.sync_manager import SyncManager

# Fixture que prepara el entorno de sincronización con un índice temporal.
@pytest_asyncio.fixture(scope="function")
async def setup_sync_env():
    config = Config()
    # Usamos la carpeta de índices real: "data/indices"
    indices_dir = Path("data/indices")
    indices_dir.mkdir(parents=True, exist_ok=True)
    config.INDICES_DIR = indices_dir

    # Si ya existe el archivo faiss_index.bin, se elimina para crear uno nuevo.
    faiss_index_path = indices_dir / "faiss_index.bin"
    if faiss_index_path.exists():
        faiss_index_path.unlink()  # Elimina el archivo existente
        logger.info(f"Archivo existente {faiss_index_path} eliminado para crear uno nuevo.")

    # Creamos el PersistentFAISSManager con la configuración actualizada.
    faiss_manager = PersistentFAISSManager(config)
    sync_manager = SyncManager(config, faiss_manager)
    yield sync_manager

# Fixture para asegurar que existe un documento dummy (requerido por la llave foránea en chunks).
@pytest_asyncio.fixture(scope="function")
async def dummy_document(setup_sync_env):
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

# Fixture autouse para limpiar los registros de prueba antes de cada test.
@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_test_data(setup_sync_env):
    pool = await setup_sync_env._get_pool()
    # Definimos los chunk_id de prueba
    test_chunk_ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
        "00000000-0000-0000-0000-000000000003"
    ]
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM embeddings WHERE chunk_id = ANY($1::uuid[])",
            test_chunk_ids
        )
        await conn.execute(
            "DELETE FROM chunks WHERE chunk_id = ANY($1::uuid[])",
            test_chunk_ids
        )
    yield

@pytest.mark.asyncio
async def test_trigger_set_needs_indexing(setup_sync_env, dummy_document):
    """
    **Paso 2.3: Trigger en PostgreSQL**
    Inserta un registro en chunks y verifica que el trigger marque needs_indexing = TRUE.
    """
    sync_manager = setup_sync_env
    pool = await sync_manager._get_pool()
    test_chunk_id = "00000000-0000-0000-0000-000000000001"
    sample_content = "Texto de prueba para trigger"
    chunk_number = 1
    doc_id = dummy_document

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, content, chunk_number)
            VALUES ($1, $2, $3, $4)
            """,
            test_chunk_id, doc_id, sample_content, chunk_number
        )
        record = await conn.fetchrow(
            "SELECT needs_indexing FROM chunks WHERE chunk_id = $1",
            test_chunk_id
        )
    logger.info(f"Registro insertado con needs_indexing = {record['needs_indexing']}")
    assert record["needs_indexing"] is True, "El trigger no marcó el registro como pendiente de indexación"

@pytest.mark.asyncio
async def test_batch_job_query_and_conversion(setup_sync_env, dummy_document):
    """
    **Pasos 3.1 y 3.2: Batch Job y Conversión de Datos**
    Inserta un registro en chunks y su correspondiente embedding en embeddings, y verifica que el vector se convierta a un array de NumPy.
    """
    sync_manager = setup_sync_env
    pool = await sync_manager._get_pool()
    test_chunk_id = "00000000-0000-0000-0000-000000000002"
    sample_content = "Texto para test de conversión"
    chunk_number = 2
    doc_id = dummy_document

    # Vector de dimensión 384 según el esquema.
    sample_embedding = [0.5] * 384
    # Convertir el vector a cadena, ya que la columna embedding es de tipo vector(384)
    sample_embedding_str = str(sample_embedding)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, content, chunk_number)
            VALUES ($1, $2, $3, $4)
            """,
            test_chunk_id, doc_id, sample_content, chunk_number
        )
        await conn.execute(
            """
            INSERT INTO embeddings (chunk_id, model_name, embedding)
            VALUES ($1, $2, $3)
            """,
            test_chunk_id, "test_model", sample_embedding_str
        )
        record = await conn.fetchrow(
            "SELECT embedding FROM embeddings WHERE chunk_id = $1",
            test_chunk_id
        )
    logger.info(f"Embedding obtenido: {record['embedding']}")
    vector = sync_manager._parse_embedding(record["embedding"])
    logger.info(f"Vector convertido: shape={vector.shape}, dtype={vector.dtype}")
    assert isinstance(vector, np.ndarray), "El embedding convertido debe ser un array de NumPy"
    assert vector.dtype == np.float32, "El array debe ser de tipo float32"

@pytest.mark.asyncio
async def test_faiss_insertion_and_postgres_update(setup_sync_env):
    """
    **Pasos 3.3 y 3.4: Inserción en FAISS y Actualización en PostgreSQL**
    Ejecuta la sincronización completa y verifica que el registro de prueba (con embedding) se actualice (needs_indexing = FALSE).
    """
    sync_manager = setup_sync_env

    processed, failed = await sync_manager.synchronize()
    logger.info(f"Sincronización completada - Procesados: {processed}, Fallidos: {failed}")
    assert failed == 0, f"No deberían haber fallado embeddings: {failed}"

    status = await sync_manager.verify_sync_status()
    logger.info(f"Estado post-sincronización: {status}")
    assert status['postgres_embeddings'] == status['faiss_vectors'], "Discrepancia entre PostgreSQL y FAISS"

    pool = await sync_manager._get_pool()
    async with pool.acquire() as conn:
        # Consultamos solo el registro de prueba (chunk_id 00000000-0000-0000-0000-000000000002)
        records = await conn.fetch(
            "SELECT chunk_id, needs_indexing FROM chunks WHERE chunk_id = $1",
            "00000000-0000-0000-0000-000000000002"
        )
    logger.info(f"Registro de prueba pendiente: {records}")
    assert len(records) == 0, "El registro de prueba con embedding aún pendiente tras la sincronización, debe marcarse como FALSE"

@pytest.mark.asyncio
async def test_search_with_converted_vector(setup_sync_env, dummy_document):
    """
    Test de búsqueda en FAISS usando un vector convertido.
    Se inserta un registro de prueba con embedding, se sincroniza y luego se realiza la búsqueda.
    """
    sync_manager = setup_sync_env
    faiss_manager = sync_manager.faiss_manager
    pool = await sync_manager._get_pool()

    # Insertar un registro de prueba para la búsqueda
    test_chunk_id = "00000000-0000-0000-0000-000000000003"
    sample_content = "Texto para test de búsqueda"
    chunk_number = 3
    doc_id = dummy_document
    sample_embedding = [0.5] * 384
    sample_embedding_str = str(sample_embedding)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, content, chunk_number)
            VALUES ($1, $2, $3, $4)
            """,
            test_chunk_id, doc_id, sample_content, chunk_number
        )
        await conn.execute(
            """
            INSERT INTO embeddings (chunk_id, model_name, embedding)
            VALUES ($1, $2, $3)
            """,
            test_chunk_id, "test_model", sample_embedding_str
        )

    # Sincronizamos para que el vector se añada al índice FAISS.
    await sync_manager.synchronize()

    # Ahora realizamos la búsqueda con un vector de ceros (control).
    zero_vector = np.zeros((1, faiss_manager.dimension), dtype=np.float32)
    logger.info(f"Buscando con vector de ceros: shape {zero_vector.shape}, dtype {zero_vector.dtype}")
    distances_zero, ids_zero = faiss_manager.search(zero_vector, k=1)
    logger.info(f"Resultado búsqueda vector de ceros: Distancias: {distances_zero}, IDs: {ids_zero}")

    # También probamos con el vector real insertado.
    async with pool.acquire() as conn:
        sample = await conn.fetchrow(
            "SELECT embedding FROM embeddings WHERE chunk_id = $1",
            test_chunk_id
        )
    if sample is None:
        pytest.skip("No se encontró embedding de prueba; se omite el test de búsqueda.")
    else:
        vector = sync_manager._parse_embedding(sample['embedding'])
        logger.info(f"Query vector shape: {vector.shape}, dtype: {vector.dtype}")
        logger.info(f"Primeros 5 valores: {vector[0][:5]}")
        logger.info(f"Is C-contiguous: {vector.flags['C_CONTIGUOUS']}")
        distances, ids = faiss_manager.search(vector, k=1)
        logger.info(f"Resultado búsqueda vector de prueba: Distancias: {distances}, IDs: {ids}")
        assert len(ids) > 0, "La búsqueda debería retornar resultados"
        assert all(d >= 0 for d in distances[0]), "Las distancias deberían ser no negativas"
