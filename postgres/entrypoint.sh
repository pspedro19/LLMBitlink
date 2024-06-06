#!/bin/bash
set -e

# Define the full path to the PostgreSQL binaries
POSTGRES_BIN="/usr/lib/postgresql/13/bin/postgres"
INITDB_BIN="/usr/lib/postgresql/13/bin/initdb"

# Ensure environment variables are set
if [ -z "$PG_USER" ] || [ -z "$PG_PASSWORD" ] || [ -z "$PG_DATABASE" ]; then
  echo "PG_USER, PG_PASSWORD, and PG_DATABASE must be set"
  exit 1
fi

# Assign environment variables to local variables
pg_user=$PG_USER
pg_password=$PG_PASSWORD
pg_database=$PG_DATABASE
pg_multiple_databases=$PG_MULTIPLE_DATABASES

# Adjust permissions for PostgreSQL data directory
chown -R postgres:postgres /var/lib/postgresql/data
chmod -R 0700 /var/lib/postgresql/data

# Check if the data directory is valid
if [ ! -f /var/lib/postgresql/data/pgdata/PG_VERSION ]; then
    echo "No valid data found in /var/lib/postgresql/data/pgdata, initializing database."
    # Check if the directory is empty or only contains non-essential files
    if [ -z "$(ls -A /var/lib/postgresql/data/pgdata | grep -vE 'pg_hba.conf|postgresql.conf')" ]; then
        # Initialize the database
        su - postgres -c "$INITDB_BIN -D /var/lib/postgresql/data/pgdata"
    else
        echo "Error: /var/lib/postgresql/data/pgdata is not empty and does not contain a valid PostgreSQL data directory."
        exit 1
    fi
fi

# Start PostgreSQL using the postgres user with full path to the binary
su - postgres -c "$POSTGRES_BIN -D /var/lib/postgresql/data/pgdata &"

# Wait for PostgreSQL to be ready
until pg_isready -h localhost -U postgres; do
  echo "Waiting for PostgreSQL to start..."
  sleep 2
done
echo "PostgreSQL started successfully."

# Create the airflow role and database if they do not exist
su - postgres -c "psql -v ON_ERROR_STOP=1 <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$pg_user') THEN
            EXECUTE format('CREATE ROLE %I WITH LOGIN PASSWORD %L', '$pg_user', '$pg_password');
        END IF;
        IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '$pg_database') THEN
            EXECUTE format('CREATE DATABASE %I OWNER %I', '$pg_database', '$pg_user');
        END IF;
    END
    \$\$;
EOSQL"

# Create multiple databases if they do not exist
if [ -n "$pg_multiple_databases" ]; then
  echo "Multiple databases creation requested: $pg_multiple_databases"
  for db in $(echo $pg_multiple_databases | tr ',' ' '); do
    echo "Creating database '$db' if it does not exist."
    su - postgres -c "psql -v ON_ERROR_STOP=1 --username \"$pg_user\" --dbname \"$pg_database\" <<-EOSQL
        SELECT 'CREATE DATABASE $db'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\\gexec
EOSQL"
  done
  echo "Databases checked/created."
fi

# Ensure the pgvector extension is created
su - postgres -c "psql -v ON_ERROR_STOP=1 --username \"$pg_user\" --dbname \"$pg_database\" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS pgvector;
EOSQL"
echo "pgvector extension ensured."

# Wait for the main process to start
wait
