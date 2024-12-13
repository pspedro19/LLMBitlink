# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias del sistema necesarias para compilar algunos paquetes
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia el archivo requirements.txt y luego instálalo
COPY ./app/requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Verifica e instala los modelos de spaCy
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_md
RUN python -m spacy validate

# Instalar uvicorn globalmente
RUN pip install uvicorn

# Copia el código de la aplicación
COPY ./app .

# Expone el puerto 8800
EXPOSE 8800

# Variables de entorno
ENV PYTHONPATH=/app
ENV MEDIA_DIR=/app/images

# Comando para ejecutar la aplicación FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8800", "--reload"]