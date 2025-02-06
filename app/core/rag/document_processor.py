# core/rag/document_processor.py
from typing import List, Dict
import tiktoken
from pydantic import BaseModel, Field

from core.rag.config import Config

# Modelo para validar la entrada del documento
class DocumentInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    metadata: Dict = Field(default_factory=dict)
    model_name: str = Field(default="miniLM")

class DocumentProcessor:
    def __init__(self, config: Config):
        self.config = config
        # Se utiliza el tokenizer de tiktoken (ejemplo: para GPT)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def validate_document(self, doc: DocumentInput) -> DocumentInput:
        if not doc.content.strip():
            raise ValueError("El contenido del documento no puede estar vacío.")
        return doc

    def create_chunks(self, text: str) -> List[Dict]:
        """
        Divide el texto en chunks respetando el tamaño máximo y el solapamiento.
        """
        chunks = []
        start = 0
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP

        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Se intenta cortar en el último espacio para evitar partir palabras
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            chunk_text = text[start:end].strip()
            if chunk_text:
                token_ids = self.tokenizer.encode(chunk_text)
                chunk_info = {
                    'content': chunk_text,
                    'start': start,
                    'end': end,
                    'token_count': len(token_ids)
                }
                chunks.append(chunk_info)
            # Se mueve el inicio considerando el solapamiento
            start = end - overlap
        return chunks
