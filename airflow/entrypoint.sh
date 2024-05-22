#!/bin/bash

# Wait for PostgreSQL to be ready
/wait-for-it.sh postgres:5432 --timeout=60 --strict -- echo "Postgres is up and running"


# Start the Airflow webserver
airflow webserver
