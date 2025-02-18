# app\test\test_search.py
import os
import sys
import logging
import pytest

# Agregar la raíz del proyecto al sys.path para poder importar "core"
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from app.core.rag.config import Config
from app.core.rag.retriever import RAGRetriever
from app.core.rag.document_processor import DocumentInput

logging.basicConfig(level=logging.INFO)

@pytest.mark.timeout(60)  # Opcional: limita el tiempo máximo de ejecución del test
def test_search():
    """
    Test para verificar la funcionalidad de búsqueda en el sistema RAG utilizando
    los datos de Curacao ya alimentados en la base de datos. Se asume que la
    base de datos contiene documentos en inglés relacionados con Curacao.
    
    Mejoras implementadas:
      - Se prueban dos variantes de query: "Curacao" y "Curaçao" para cubrir diferencias
        de codificación.
      - Si no se encuentran resultados, se ingiere un documento de prueba con contenido
        conocido sobre Curacao, y se repite la búsqueda.
      - Se agregan logs detallados para depuración.
      - Se valida que cada resultado contenga las claves esperadas y que el valor de
        "similarity" sea razonable.
    """
    # Configuración de prueba: ajusta las credenciales a tu entorno real.
    config = Config(DB_PASSWORD="mypassword", DB_NAME="mydatabase", DB_USER="myuser")
    rag = RAGRetriever(config)

    # Variantes de query para cubrir diferencias en la escritura del nombre
    queries = ["Curacao", "Curaçao"]
    results = None
    for query in queries:
        results = rag.search(query, top_k=5, model_name="miniLM")
        logging.info(f"Query '{query}' retornó {len(results)} resultados.")
        if results and len(results) > 0:
            break

    # Si no se encuentran resultados para ninguna variante, se ingiere un documento de prueba.
    if not results or len(results) == 0:
        logging.warning("No se encontraron resultados. Ingeriendo un documento de prueba sobre Curacao.")
        test_content = (
            "Curacao is a vibrant tourist destination in the Caribbean. Known for its "
            "colorful colonial architecture, beautiful beaches, and rich cultural heritage, "
            "the city of Willemstad offers a unique blend of modern attractions and historical charm."
        )
        test_metadata = {"origen": "TestData", "language": "en"}
        doc_input = DocumentInput(
            title="Curacao Test Document",
            content=test_content,
            metadata=test_metadata,
            model_name="miniLM"
        )
        doc_id = rag.process_document(doc_input)
        logging.info(f"Documento de prueba ingresado con doc_id={doc_id}")
        # Repetir la búsqueda usando la misma lista de queries
        for query in queries:
            results = rag.search(query, top_k=5, model_name="miniLM")
            logging.info(f"After ingestion, query '{query}' returned {len(results)} results.")
            if results and len(results) > 0:
                break

    # Verificar que se hayan encontrado resultados
    assert results is not None and len(results) > 0, (
        "No se encontraron resultados para la búsqueda de datos de Curacao después de ingerir el documento de prueba. "
        "Verifica que la base de datos esté alimentada correctamente y que el query sea adecuado."
    )

    # Validar que cada resultado contenga las claves esperadas
    for result in results:
        for key in ["title", "content", "similarity"]:
            assert key in result, f"El resultado no contiene la clave '{key}'."
        try:
            similarity = float(result["similarity"])
        except ValueError:
            pytest.fail(f"El valor de 'similarity' no se puede convertir a float: {result['similarity']}")
        # Opcional: se verifica que la similitud tenga un valor razonable (por ejemplo, >= 0.3)
        assert similarity >= 0.3, f"La similitud de '{result['title']}' es demasiado baja: {similarity}"
        logging.info(f"Resultado - Título: {result['title']}, Similaridad: {result['similarity']}")
