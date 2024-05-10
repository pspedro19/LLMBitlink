# Dockerfile in ./airflow/Dockerfile
# Example using Airflow version 2.9.1 with Python 3.9
FROM apache/airflow:slim-2.9.1rc2-python3.9

# Install required Python packages including psycopg2-binary for PostgreSQL integration
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    && pip install psycopg2-binary

# Set the working directory to Airflow home
WORKDIR /opt/airflow


# Copy DAGs, configuration files, and entrypoint script into the image
COPY ./dags ./dags
COPY airflow.cfg /opt/airflow/airflow.cfg
COPY entrypoint.sh /entrypoint.sh
COPY [^settings.py]* /home/airflow/.local/lib/python3.9/site-packages/airflow/

# Set environment variables for Airflow configuration
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__FERNET_KEY=base64encodedFernetKeyHere
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True

# Ensure the user has adequate permissions to perform operations, especially useful if using a non-root user
USER root
RUN chmod +x /entrypoint.sh
USER airflow

# Define the entry point and the command to run Airflow webserver
ENTRYPOINT ["/entrypoint.sh"]
CMD ["airflow", "webserver", "--port", "8080"]
