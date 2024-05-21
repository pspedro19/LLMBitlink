FROM apache/airflow:2.9.1-python3.9

# Set user to root to install dependencies and create directories
USER root
ARG AIRFLOW_UID
ARG AIRFLOW_GID

# Check if the group 'airflow' exists, and create if not
# Added check for existing group and user to prevent errors during setup
RUN getent group ${AIRFLOW_GID} || groupadd -g ${AIRFLOW_GID} airflow
RUN id -u airflow &>/dev/null || useradd -u ${AIRFLOW_UID} -g ${AIRFLOW_GID} -m -s /bin/bash airflow

# If the user exists, ensure the UID and GID are correctly set
RUN usermod -u ${AIRFLOW_UID} airflow && groupmod -g ${AIRFLOW_GID} airflow

# Create the necessary directories and set permissions
# Ensured the directories are created with the right permissions
RUN mkdir -p /opt/airflow/logs/scheduler && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 775 /opt/airflow/logs

# Switch to user airflow to install packages
USER airflow

# Install required Python packages including psycopg2-binary for PostgreSQL integration
# Updated to include psycopg2-binary directly in requirements
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
# Ensure that the entrypoint script is executable
RUN chmod +x /entrypoint.sh

# Switch back to the airflow user before running the entrypoint script
USER airflow
ENTRYPOINT ["/entrypoint.sh"]
CMD ["airflow", "webserver", "--port", "8081"]
