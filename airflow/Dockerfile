FROM apache/airflow:2.9.1-python3.9

# Set user to root to install dependencies and create directories
USER root

# Create the necessary directories
RUN mkdir -p /opt/airflow/logs/scheduler && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 775 /opt/airflow/logs

# Switch to user airflow to install packages
USER airflow

# Install required Python packages including psycopg2-binary for PostgreSQL integration
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt && \
    pip install psycopg2-binary

# Set the working directory to Airflow home
WORKDIR /opt/airflow

# Copy DAGs, configuration files, and entrypoint script into the image
COPY ./dags ./dags
COPY airflow.cfg /opt/airflow/airflow.cfg
COPY entrypoint.sh /entrypoint.sh

# Set environment variables for Airflow configuration
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__FERNET_KEY=base64encodedFernetKeyHere
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW_PORT=8081

# Ensure the user has adequate permissions to perform operations
USER root
RUN chmod +x /entrypoint.sh

# Ensure the entrypoint script is executable and configure the container to start with it
USER airflow
ENTRYPOINT ["/entrypoint.sh"]
CMD ["airflow", "webserver", "--port", "8081"]
