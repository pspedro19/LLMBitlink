FROM python:3.9-slim

WORKDIR /app

# 1. Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    git \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 2. Crear estructura de directorios
RUN mkdir -p /app/chat-interface /app/images /app/data/database /app/logs /app/real_estate_logs /app/models /app/data/documents /app/data/embeddings

# 2.1. Configurar pip para mayor timeout y mirror más rápido
RUN pip config set global.timeout 1000 && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 3. Gestión de dependencias Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

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

# 4. Copiar e instalar requirements primero para cacheo eficiente
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Instalación específica de spaCy con modelos (versiones explícitas)
RUN pip install --no-cache-dir spacy==3.7.4 && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download es_core_news_sm && \
    python -m spacy download es_core_news_md && \
    python -m spacy validate

# 6. Copiar archivos Excel (manteniendo verificación)
COPY data/database/*.xlsx /app/data/database/

# 7. Validación de archivos Excel (solo si existen)
RUN if ls /app/data/database/*.xlsx 1> /dev/null 2>&1; then \
    echo "Archivos Excel encontrados:"; ls /app/data/database/*.xlsx; \
    else echo "Advertencia: No se encontraron archivos Excel"; fi

# 8. Copiar aplicación
COPY . .

# 9. Configuración final
RUN chmod -R 755 /app/data/database /app/logs /app/real_estate_logs /app/data /app/models
ENV PYTHONPATH=/app \
    MEDIA_DIR=/app/images \
    DB_PATH=/app/chat-interface/db.sqlite3 \
    SPACY_WARNING_IGNORE=W008

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]