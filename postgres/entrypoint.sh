#!/bin/bash
set -e

# Define paths to PostgreSQL binaries
POSTGRES_BIN="/usr/lib/postgresql/13/bin/postgres"
INITDB_BIN="/usr/lib/postgresql/13/bin/initdb"

# Adjust permissions for the PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/data
chmod -R 0700 /var/lib/postgresql/data

# Validate the data directory
if [ ! -f /var/lib/postgresql/data/pgdata/PG_VERSION ]; then
    echo "No valid data found in /var/lib/postgresql/data/pgdata, initializing database."
    if [ -z "$(ls -A /var/lib/postgresql/data/pgdata | grep -vE 'pg_hba.conf|postgresql.conf')" ]; then
        su - postgres -c "$INITDB_BIN -D /var/lib/postgresql/data/pgdata"
    else
        echo "Error: Directory not empty or invalid."
        exit 1
    fi
fi

# Start PostgreSQL
su - postgres -c "$POSTGRES_BIN -D /var/lib/postgresql/data/pgdata &"
sleep 5  # Wait for server to start



# Wait for PostgreSQL to be ready
until pg_isready -h localhost -U postgres; do
    echo "Waiting for PostgreSQL to start..."
    sleep 2
done
echo "PostgreSQL started successfully."

# Execute SQL commands
su - postgres -c "psql -v ON_ERROR_STOP=1 --command=\"DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow') THEN CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow' NOCREATEDB NOCREATEROLE NOINHERIT; END IF; END \$\$;\""

# Create 'airflow' database if it doesn't exist
su - postgres -c "psql -v ON_ERROR_STOP=1 --command='CREATE DATABASE airflow OWNER airflow;'"

# Create 'mlflow_db' database if it doesn't exist
su - postgres -c "psql -v ON_ERROR_STOP=1 --command='CREATE DATABASE mlflow_db OWNER airflow;'"

# Ensure the 'pgvector' extension is installed
if [ ! -f /usr/share/postgresql/13/extension/pgvector.control ]; then
    echo "pgvector control file not found. Ensure pgvector is installed correctly."
    exit 1
fi
su - postgres -c "psql -v ON_ERROR_STOP=1 --username airflow --dbname airflow --command='CREATE EXTENSION IF NOT EXISTS pgvector;'"
echo "pgvector extension ensured."

# Wait for the main process to start
wait
