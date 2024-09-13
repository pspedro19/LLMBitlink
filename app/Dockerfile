FROM python:3.8-slim

WORKDIR /app

# Copiamos solo el archivo de requisitos primero y lo instalamos
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Ahora copiamos el resto del código
COPY . .

EXPOSE 8000
CMD ["uvicorn", "Vectorize:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
