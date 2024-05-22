FROM apache/airflow:2.9.1-python3.9

USER root

# Set ARG for user IDs
ARG AIRFLOW_UID=1000
ARG AIRFLOW_GID=1000

# Check if the group 'airflow' exists, and create if not
RUN getent group ${AIRFLOW_GID} || groupadd -g ${AIRFLOW_GID} airflow
RUN id -u airflow &>/dev/null || useradd -u ${AIRFLOW_UID} -g ${AIRFLOW_GID} -m -s /bin/bash airflow

# Create the necessary directories and set permissions
RUN mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 775 /opt/airflow/logs

# Set the working directory to Airflow home
WORKDIR /opt/airflow

# Copy DAGs, configuration files, and scripts
COPY ./dags ./dags
COPY airflow.cfg /opt/airflow/airflow.cfg
COPY entrypoint.sh /entrypoint.sh
COPY requirements.txt /requirements.txt

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__FERNET_KEY=base64encodedFernetKeyHere
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW_PORT=8081

USER airflow

# Install required Python packages as the root user
RUN pip install --no-cache-dir -r /requirements.txt && \
    pip install --no-cache-dir psycopg2-binary


USER root
# Ensure entrypoint script is executable
RUN chmod +x /entrypoint.sh

# Switch to the airflow user for running Airflow processes
USER airflow

ENTRYPOINT ["/entrypoint.sh"]
CMD ["airflow", "webserver", "--port", "8081"]
