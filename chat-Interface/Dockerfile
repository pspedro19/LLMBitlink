# Usamos una imagen oficial de Python como base
FROM python:3.8-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema para PostgreSQL y otros paquetes necesarios
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos primero, para aprovechar la caché de Docker
COPY ./requirements.txt ./

# Instalar dependencias de Python y actualizar pip
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto Django al contenedor
COPY . /app

# Verificar la instalación de Django
RUN pip list

# Recopilar archivos estáticos
RUN python manage.py collectstatic --noinput

# Exponer el puerto donde se ejecutará el servidor web
EXPOSE 8000

# Define environment variable
ENV NAME World

# Comando para ejecutar la aplicación usando Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "chatbot.wsgi:application"]
