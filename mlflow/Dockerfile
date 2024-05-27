# Use a specific Python slim image from official sources
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /mlflow

# Install system dependencies and ensure curl is correctly installed as part of the apt-get install command
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl && \
    # Clean up APT when done to reduce image size
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies file to the container
COPY requirements.txt .

# Install Python dependencies, include specific MLflow extras for additional features
# Using extras like 'sqlalchemy' and 'google' for enhanced functionality if needed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlflow[extras,sqlalchemy,google] psycopg2-binary boto3 cryptography pymysql

# Expose the port MLflow will run on
EXPOSE 5000

# Define the command to run the MLflow server with the standard port and configurations
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql+psycopg2://airflow:airflow@postgres/mlflow_db", \
     "--default-artifact-root", "s3://minio:9000/mlflow", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
