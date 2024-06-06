#!/bin/bash
set -e
# Define the full path to the PostgreSQL binary
POSTGRES_BIN="/usr/lib/postgresql/13/bin/postgres"

# Adjust permissions for PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/data
chmod -R 0700 /var/lib/postgresql/data

# Check if the data directory is empty
if [ -z "$(ls -A /var/lib/postgresql/data)" ]; then
    echo "No data found in /var/lib/postgresql/data, initializing database."
    # Start as the postgres user if the data directory is empty
    su - postgres -c "docker-entrypoint.sh postgres &"
else
    echo "Data found in /var/lib/postgresql/data, using existing data."
    # Start PostgreSQL using the postgres user with full path to the binary if data exists
    su - postgres -c "$POSTGRES_BIN -D /var/lib/postgresql/data &"
fi

# Wait for PostgreSQL to be ready
until pg_isready -h localhost -U "$POSTGRES_USER"; do
  echo "Waiting for PostgreSQL to start..."
  sleep 2
done
echo "PostgreSQL started successfully."

# Create multiple databases if they do not exist
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  echo "Multiple databases creation requested: $POSTGRES_MULTIPLE_DATABASES"
  for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
    echo "Creating database '$db' if it does not exist."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE $db'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\\gexec
EOSQL
  done
  echo "Databases checked/created."
fi

# Ensure the pgvector extension is created
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS pgvector;
EOSQL
echo "pgvector extension ensured."

# Wait for the main PostgreSQL process to finish
wait
