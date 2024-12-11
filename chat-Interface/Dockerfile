# Utiliza una imagen oficial de Python como imagen base
FROM python:3.9

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Después de WORKDIR /app
RUN mkdir -p property_images
RUN ls -la property_images/

# Copia el archivo de requisitos y lo instala
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto en el contenedor
COPY . .

# Expone el puerto que Django usará
EXPOSE 8000

# Ejecuta el servidor Django cuando el contenedor inicie
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
