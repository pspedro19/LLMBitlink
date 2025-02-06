import logging
import pytest
from core.rag.config import Config
from core.rag.retriever import RAGRetriever

logging.basicConfig(level=logging.INFO)

def test_search():
    """
    Test para verificar la funcionalidad de búsqueda en el sistema RAG.
    Se asume que previamente se haya ingresado al menos un documento de prueba.
    """
    # Configuración de prueba (ajusta los parámetros según tu entorno de test)
    config = Config(DB_PASSWORD="mi_contraseña", DB_NAME="rag_db")
    rag = RAGRetriever(config)

    # Consulta de prueba (debe coincidir con los metadatos del documento ingresado)
    query = "¿Qué dice Test User sobre el tema prueba?"
    results = rag.search(query, top_k=5, model_name="miniLM", filters={"autor": "Test User"})

    # Verificar que se haya devuelto una lista de resultados
    assert isinstance(results, list)
    
    # Si existen resultados, se verifica que cada uno contenga la información esperada
    if results:
        for r in results:
            assert "title" in r
            assert "content" in r
            assert "similarity" in r
            logging.info(f"Resultado - Título: {r['title']}, Similaridad: {r['similarity']}")
    else:
        logging.warning("No se encontraron resultados para la búsqueda de prueba.")
