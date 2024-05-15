#!/bin/bash

# Crear directorios y establecer permisos
mkdir -p /opt/airflow/logs/scheduler
chown -R airflow:root /opt/airflow/logs

# Inicializar la base de datos de Airflow
airflow db init

# Aplicar cualquier migración necesaria
airflow db upgrade

# Instalar pendulum y aplicar parche a settings.py si es necesario
pip install --upgrade pendulum==2.1.2
sed -i "s/TIMEZONE = pendulum.tz.timezone('UTC')/import pendulum\nTIMEZONE = pendulum.timezone('UTC')/g" /home/airflow/.local/lib/python3.9/site-packages/airflow/settings.py

# Iniciar el servidor web de Airflow
exec airflow webserver
