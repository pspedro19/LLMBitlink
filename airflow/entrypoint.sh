#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# Check and create directories with error handling to prevent crashes on permissions issues
if [ ! -d "/opt/airflow/logs/scheduler" ]; then
    mkdir -p /opt/airflow/logs/scheduler
fi

# Update permissions, using conditional checks to avoid unnecessary operations
if [ "$(stat -c '%U:%G' /opt/airflow/logs)" != "airflow:root" ]; then
    chown -R airflow:root /opt/airflow/logs || {
        echo "Warning: Unable to change ownership of /opt/airflow/logs"
    }
fi

if [ "$(stat -c '%a' /opt/airflow/logs)" != "775" ]; then
    chmod -R 775 /opt/airflow/logs || {
        echo "Warning: Unable to set permissions on /opt/airflow/logs"
    }
fi

# Initialize the Airflow database and handle potential errors
airflow db init || {
    echo "Failed to initialize Airflow DB"
    exit 1
}

# Apply necessary migrations
airflow db upgrade || {
    echo "Failed to apply database migrations"
    exit 1
}

# Install required dependencies safely
pip install --user --upgrade pendulum==2.1.2 || {
    echo "Failed to install pendulum, trying without upgrading"
    pip install --user pendulum==2.1.2 || {
        echo "Failed to install pendulum entirely"
        exit 1
    }
}

# Attempt to patch the Airflow settings file to correct timezone configuration
if [ -f "/home/airflow/.local/lib/python3.9/site-packages/airflow/settings.py" ]; then
    sed -i "s/import pendulum\nTIMEZONE = 'UTC'/from pendulum import timezone\nTIMEZONE = timezone('UTC')/g" /home/airflow/.local/lib/python3.9/site-packages/airflow/settings.py || {
        echo "Failed to patch the Airflow settings file"
    }
else
    echo "Airflow settings file not found, skipping patch"
fi

# Start the Airflow webserver as the last command
exec airflow webserver
