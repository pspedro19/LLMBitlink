FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \ 
    && rm -rf /var/lib/apt/lists/*
    
# Crear directorios necesarios
RUN mkdir -p /app/chat-Interface /app/images /app/data/database /app/logs /app/real_estate_logs

# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar dependencias principales primero
RUN pip install --no-cache-dir \
    "langchain==0.3.15" \
    "langchain-core==0.3.31" \
    "langchain-openai==0.3.1" \
    "openai>=1.58.1" \
    "pandasql" \
    "pandas" \
    "openpyxl>=3.1.2" \
    "xlrd>=2.0.1" \
    "SQLAlchemy<2.0.0"

# Copiar requirements y dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos Excel específicamente
COPY data/database/*.xlsx /app/data/database/

# Instalar spaCy y todos los modelos necesarios
RUN pip install --no-cache-dir spacy
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download es_core_news_md
RUN python -m spacy download en_core_web_md
RUN python -m spacy validate

# Verificar instalación de modelos spaCy
RUN python -c "import spacy; nlp = spacy.load('es_core_news_sm'); print('spaCy models loaded successfully')"

# Verificar instalación de Excel y archivos
RUN ls -la /app/data/database/ && \
    python -c "import pandas as pd; \
    import os; \
    files = ['activities.xlsx', 'tourist_spots.xlsx', 'restaurants.xlsx', 'nightclubs.xlsx', 'tourism_packages.xlsx']; \
    [print(f'Loading {f}...') or pd.read_excel(f'/app/data/database/{f}', engine='openpyxl') for f in files]; \
    print('Excel files loaded successfully')"

# Copiar el resto de la aplicación
COPY . .

# Asegurar permisos de directorios
RUN chmod -R 755 /app/data/database /app/logs /app/real_estate_logs

# Configurar PATHS
ENV PYTHONPATH=/app
ENV MEDIA_DIR=/app/images
ENV DB_PATH=/app/chat-Interface/db.sqlite3
ENV SPACY_WARNING_IGNORE=W008

# Puerto y comando
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]