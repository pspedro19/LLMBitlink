FROM apache/airflow:2.8.1

USER root

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install required Python packages globally
COPY wait-for-it.sh /wait-for-it.sh

# Ensure necessary directories exist and set permissions
RUN mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler && \
    chown -R airflow:root /opt/airflow/logs && \
    chmod -R 775 /opt/airflow/logs && \
    chmod +x /wait-for-it.sh

# Ensure that the airflow user can write to the logs directory
RUN chmod -R 777 /opt/airflow/logs

# Switch to airflow user
USER airflow

# Install required Python packages globally
RUN pip install --no-cache-dir -r /requirements.txt

# Set PATH to include user-installed packages
ENV PATH=$PATH:/home/airflow/.local/bin

# Initialize the Airflow database
RUN airflow db init

CMD ["airflow", "webserver"]
