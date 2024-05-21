#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# Crear directorios y establecer permisos
mkdir -p /opt/airflow/logs /opt/airflow/logs/scheduler
chown -R airflow:root /opt/airflow/logs
chmod -R 775 /opt/airflow/logs

# Inicializar la base de datos de Airflow
airflow db init

# Aplicar cualquier migración necesaria
airflow db upgrade

# Instalar dependencias necesarias
pip install --upgrade pendulum==2.1.2

# Corregir el archivo de configuración si es necesario
# Aplica un parche a settings.py para ajustar la configuración de la zona horaria
sed -i "s/import pendulum\nTIMEZONE = 'UTC'/from pendulum import timezone\nTIMEZONE = timezone('UTC')/g" /home/airflow/.local/lib/python3.9/site-packages/airflow/settings.py

# Comando para iniciar el servidor web de Airflow, asegurándose de que se ejecute como el último comando
exec airflow webserver
