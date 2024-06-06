#!/bin/bash
set -e

# Define the full path to the PostgreSQL binaries
POSTGRES_BIN="/usr/lib/postgresql/13/bin/postgres"
INITDB_BIN="/usr/lib/postgresql/13/bin/initdb"

# Ajusta permisos para el directorio de datos de PostgreSQL
chown -R postgres:postgres /var/lib/postgresql/data
chmod -R 0700 /var/lib/postgresql/data

# Verifica si el directorio de datos es válido
if [ ! -f /var/lib/postgresql/data/pgdata/PG_VERSION ]; then
    echo "No valid data found in /var/lib/postgresql/data/pgdata, initializing database."
    # Verifica si el directorio está vacío o solo contiene archivos no esenciales
    if [ -z "$(ls -A /var/lib/postgresql/data/pgdata | grep -vE 'pg_hba.conf|postgresql.conf')" ]; then
        # Inicializa la base de datos
        su - postgres -c "$INITDB_BIN -D /var/lib/postgresql/data/pgdata"
    else
        echo "Error: /var/lib/postgresql/data/pgdata is not empty and does not contain a valid PostgreSQL data directory."
        exit 1
    fi
fi

# Inicia PostgreSQL usando el usuario postgres con la ruta completa al binario
su - postgres -c "$POSTGRES_BIN -D /var/lib/postgresql/data/pgdata &"

# Espera a que PostgreSQL esté listo
until pg_isready -h localhost -U postgres; do
  echo "Waiting for PostgreSQL to start..."
  sleep 2
done
echo "PostgreSQL started successfully."

# Crea el rol airflow si no existe
su - postgres -c "psql -v ON_ERROR_STOP=1 <<-EOSQL
    CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow' NOCREATEDB NOCREATEROLE NOINHERIT;
EOSQL"

# Crea la base de datos airflow si no existe
su - postgres -c "psql -v ON_ERROR_STOP=1 <<-EOSQL
    CREATE DATABASE airflow OWNER airflow;
EOSQL"

# Crea la base de datos mlflow_db si no existe
su - postgres -c "psql -v ON_ERROR_STOP=1 <<-EOSQL
    CREATE DATABASE mlflow_db OWNER airflow;
EOSQL"

# Asegura que la extensión pgvector se haya creado
su - postgres -c "psql -v ON_ERROR_STOP=1 --username airflow --dbname airflow <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS pgvector;
EOSQL"
echo "pgvector extension ensured."

# Espera a que el proceso principal inicie
wait
