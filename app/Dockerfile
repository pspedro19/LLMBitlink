FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para mejor uso de caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar modelos de spaCy
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_md
RUN python -m spacy validate

# Instalar uvicorn
RUN pip install uvicorn

# Copiar el resto de la aplicación
COPY . .

# Puerto y comando
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
