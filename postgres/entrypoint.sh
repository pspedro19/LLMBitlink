#!/bin/bash
set -e

# Inicializar el servicio de PostgreSQL
pg_ctl -D "$PGDATA" -o "-c listen_addresses=''" -w start

# Verificar si la base de datos existe y crearla si no existe
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE IF NOT EXISTS mlflow_db;
EOSQL

# Detener el servicio de PostgreSQL para que el CMD de Dockerfile tome el control
pg_ctl -D "$PGDATA" -m fast -w stop

exec "$@"
