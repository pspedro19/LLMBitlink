import logging
import pytest
from core.rag.config import Config
from core.rag.retriever import RAGRetriever
from core.rag.document_processor import DocumentInput

logging.basicConfig(level=logging.INFO)

def test_process_document():
    """
    Test para verificar la ingesta de un documento mediante el pipeline RAG.
    """
    # Configuración de prueba (ajusta los parámetros según tu entorno de test)
    config = Config(DB_PASSWORD="mypassword", DB_NAME="mydatabase", DB_USER="myuser")
    rag = RAGRetriever(config)

    # Ejemplo de contenido y metadatos para la prueba
    content = (
        "Este es un documento de prueba para verificar la ingesta en el sistema RAG. "
        "El contenido debe dividirse en chunks y procesarse correctamente."
    )
    metadata = {"autor": "Test User", "tema": "prueba"}

    # Creación y validación de la entrada del documento
    doc_input = DocumentInput(
        title="Documento de Prueba",
        content=content,
        metadata=metadata,
        model_name="miniLM"
    )

    # Ejecución del pipeline de ingesta
    doc_id = rag.process_document(doc_input)
    
    # Verificar que se haya generado un doc_id (se espera que sea un string)
    assert doc_id is not None
    assert isinstance(doc_id, str)
    
    logging.info(f"Documento ingresado con doc_id={doc_id}")
