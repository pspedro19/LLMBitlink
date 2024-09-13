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
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# MinIO configuration
MINIO_URL = os.getenv("MINIO_URL", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")

client = Minio(
    MINIO_URL,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

model = SentenceTransformer('all-MiniLM-L6-v2')

class Chunk(BaseModel):
    document_id: int
    content: str
    embedding: List[float]

@app.post("/vectorize")
def vectorize_and_store():
    try:
        objects = client.list_objects("files")
        all_vectors = []

        for obj in objects:
            response = client.get_object("files", obj.object_name)
            with open(obj.object_name, "wb") as file_data:
                for d in response.stream(32*1024):
                    file_data.write(d)
            chunks = split_into_chunks(obj.object_name)
            df_chunks = pd.DataFrame(chunks)
            df_chunks['embedding'] = df_chunks['content'].apply(lambda x: model.encode(x).tolist())
            all_vectors.append(df_chunks)

        result_df = pd.concat(all_vectors, ignore_index=True)

        # Save the DataFrame to a CSV file (optional)
        result_df.to_csv("vectorized_data.csv", index=False)

        vectorized_data = result_df.to_dict(orient='records')
        logger.info(f"Data to be sent: {vectorized_data}")

        # Send vectorized data to Django
        django_url = os.getenv("DJANGO_URL", "http://django_chat_interface:8800")
        response = requests.post(f"{django_url}/save_vectorization/", json=vectorized_data)
        if response.status_code == 200:
            return {"status": "Success", "data": vectorized_data}
        else:
            raise HTTPException(status_code=500, detail=f"Error sending data to Django: {response.status_code}, {response.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException: {e}")
        raise HTTPException(status_code=500, detail=f"RequestException: {e}")
    except Exception as e:
        logger.error(f"General Exception: {e}")
        raise HTTPException(status_code=500, detail=f"General Exception: {e}")

def split_into_chunks(file_path):
    chunks = []
    document_id = 1  # This should be dynamic, but we'll use an example value
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            chunks.append({"document_id": document_id, "content": text})
    elif file_path.endswith('.txt') or file_path.endswith('.md'):
        with open(file_path, "r") as file:
            text = file.read()
            # Split the text into chunks (you can adjust this logic as needed)
            text_chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
            for idx, chunk in enumerate(text_chunks):
                chunks.append({"document_id": document_id, "content": chunk})
    os.remove(file_path)  # Clean up after processing
    return chunks

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)