import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from minio import Minio
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import fitz  # PyMuPDF for reading PDFs
import logging

app = FastAPI()

# Configuración de MinIO
MINIO_URL = "minio:9000"  # Actualizado a 'minio' en lugar de 'localhost'
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"

print("Configurando MinIO...")  # <-- PRINT DE DEPURACIÓN

client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

print("MinIO configurado exitosamente...")  # <-- PRINT DE DEPURACIÓN

model = SentenceTransformer('all-MiniLM-L6-v2')

print("Modelo cargado exitosamente...")  # <-- PRINT DE DEPURACIÓN

class Chunk(BaseModel):
    document_id: int
    content: str
    embedding: List[float]

@app.post("/vectorize")
def vectorize_and_store():
    print("Iniciando vectorización...")  # <-- PRINT DE DEPURACIÓN
    try:
        objects = client.list_objects("files")
        print(f"Objetos en MinIO: {objects}")  # <-- PRINT DE DEPURACIÓN
        all_vectors = []

        for obj in objects:
            print(f"Procesando objeto: {obj.object_name}")  # <-- PRINT DE DEPURACIÓN
            response = client.get_object("files", obj.object_name)
            with open(obj.object_name, "wb") as file_data:
                for d in response.stream(32*1024):
                    file_data.write(d)

            print(f"Archivo {obj.object_name} descargado de MinIO...")  # <-- PRINT DE DEPURACIÓN

            # Dividir en chunks y calcular embeddings
            chunks = split_into_chunks(obj.object_name)
            print(f"Chunks generados: {chunks}")  # <-- PRINT DE DEPURACIÓN

            df_chunks = pd.DataFrame(chunks)
            df_chunks['embedding'] = df_chunks['content'].apply(lambda x: model.encode(x).tolist())

            print(f"Embeddings calculados para el archivo {obj.object_name}...")  # <-- PRINT DE DEPURACIÓN
            all_vectors.append(df_chunks)

        result_df = pd.concat(all_vectors, ignore_index=True)

        # Guardar el DataFrame en un archivo CSV (opcional)
        result_df.to_csv("vectorized_data.csv", index=False)

        vectorized_data = result_df.to_dict(orient='records')
        logging.info(f"Data to be sent: {vectorized_data}")

        print(f"Vectorized data: {vectorized_data}")  # <-- PRINT DE DEPURACIÓN

        # Enviar los datos vectorizados a Django
        response = requests.post("http://django-chat-interface:8800/save_vectorization/", json=vectorized_data)
        print(f"Respuesta de Django: {response.status_code}, {response.text}")  # <-- PRINT DE DEPURACIÓN

        if response.status_code == 200:
            return {"status": "Success", "data": vectorized_data}
        else:
            raise HTTPException(status_code=500, detail=f"Error al enviar datos a Django: {response.status_code}, {response.text}")

    except requests.exceptions.RequestException as e:
        logging.error(f"RequestException: {e}")
        print(f"RequestException: {e}")  # <-- PRINT DE DEPURACIÓN
        raise HTTPException(status_code=500, detail=f"RequestException: {e}")
    except Exception as e:
        logging.error(f"General Exception: {e}")
        print(f"General Exception: {e}")  # <-- PRINT DE DEPURACIÓN
        raise HTTPException(status_code=500, detail=f"General Exception: {e}")

def split_into_chunks(file_path):
    print(f"Dividiendo el archivo {file_path} en chunks...")  # <-- PRINT DE DEPURACIÓN
    chunks = []
    document_id = 1  # Esto debería ser dinámico, pero usaremos un valor de ejemplo
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            chunks.append({"document_id": document_id, "content": text})
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        with open(file_path, "r") as file:
            text = file.read()
            # Dividir el texto en chunks (puedes ajustar esta lógica según sea necesario)
            text_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
            for idx, chunk in enumerate(text_chunks):
                chunks.append({"document_id": document_id, "content": chunk})
    os.remove(file_path)  # Limpiar después de procesar
    print(f"Archivo {file_path} procesado y eliminado...")  # <-- PRINT DE DEPURACIÓN
    return chunks