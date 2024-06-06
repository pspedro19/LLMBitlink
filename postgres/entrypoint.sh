#!/bin/bash
set -e

# Define the full path to the PostgreSQL binaries
POSTGRES_BIN="/usr/lib/postgresql/13/bin/postgres"
INITDB_BIN="/usr/lib/postgresql/13/bin/initdb"

# Adjust permissions for PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/data
chmod -R 0700 /var/lib/postgresql/data

# Check if the data directory is valid
if [ ! -f /var/lib/postgresql/data/PG_VERSION ]; then
    echo "No valid data found in /var/lib/postgresql/data, initializing database."
    # Remove all contents if the directory is not empty but not valid
    rm -rf /var/lib/postgresql/data/*
    # Initialize the database
    su - postgres -c "$INITDB_BIN -D /var/lib/postgresql/data"
fi

# Start PostgreSQL using the postgres user with full path to the binary
su - postgres -c "$POSTGRES_BIN -D /var/lib/postgresql/data &"

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
