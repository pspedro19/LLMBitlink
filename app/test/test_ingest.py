import os
import glob
import logging
import pytest

from core.rag.config import Config
from core.rag.retriever import RAGRetriever
from core.rag.document_processor import DocumentInput

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae el texto de un archivo PDF utilizando PyPDF2.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 is required to extract text from PDF files. Install it via pip: pip install PyPDF2")
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def test_process_document():
    """
    Test para verificar la ingesta de documentos PDF mediante el pipeline RAG.

    Se procesan los documentos PDF en los siguientes directorios:
      - app/data/knowledge_base/curaçao_information/0_Introduccion
      - app/data/knowledge_base/curaçao_information/1_Atracciones
      - app/data/knowledge_base/curaçao_information/2_Cultura_e_Historia
      - app/data/knowledge_base/curaçao_information/3_Consejos_Practicos
    """
    # Configuración de prueba (ajusta las credenciales según tu entorno de test)
    config = Config(DB_PASSWORD="mypassword", DB_NAME="mydatabase", DB_USER="myuser")
    rag = RAGRetriever(config)

    # Definir el directorio base (ruta absoluta relativa a la raíz del repositorio)
    base_dir = os.path.abspath("app/data/knowledge_base/curaçao_information")
    logging.info(f"Buscando PDFs en: {base_dir}")
    
    # Lista explícita de subdirectorios a procesar
    subdirectories = [
        os.path.join(base_dir, "0_Introduccion"),
        os.path.join(base_dir, "1_Atracciones"),
        os.path.join(base_dir, "2_Cultura_e_Historia"),
        os.path.join(base_dir, "3_Consejos_Practicos")
    ]
    
    # Procesar cada subdirectorio
    for subdir in subdirectories:
        logging.info(f"Procesando directorio: {subdir}")
        # Buscar todos los archivos PDF en el subdirectorio (incluyendo subdirectorios anidados, si existen)
        pdf_files = glob.glob(os.path.join(subdir, "**", "*.pdf"), recursive=True)
        
        # Verificar que se encontraron archivos en este subdirectorio
        assert len(pdf_files) > 0, f"No se encontraron archivos PDF en el directorio: {subdir}"
        
        # Procesar cada archivo PDF
        for pdf_file in pdf_files:
            logging.info(f"Procesando archivo: {pdf_file}")
            try:
                text = extract_text_from_pdf(pdf_file)
            except Exception as e:
                pytest.fail(f"Error extrayendo texto de {pdf_file}: {e}")
            
            # Utilizar el nombre del archivo como título y el nombre del subdirectorio como categoría
            title = os.path.basename(pdf_file)
            metadata = {
                "file_path": pdf_file,
                "category": os.path.basename(subdir)
            }
            
            doc_input = DocumentInput(
                title=title,
                content=text,
                metadata=metadata,
                model_name="miniLM"
            )
            
            try:
                doc_id = rag.process_document(doc_input)
            except Exception as e:
                pytest.fail(f"Error procesando documento {title}: {e}")
            
            # Verificar que se haya generado un doc_id (se espera que sea un string)
            assert doc_id is not None, f"El documento {title} no fue procesado correctamente."
            assert isinstance(doc_id, str)
            logging.info(f"Documento {title} ingresado con doc_id={doc_id}")
