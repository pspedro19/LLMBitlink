#!/bin/bash
set -e

# Ejecuta el script de entrada predeterminado de PostgreSQL para la inicialización
docker-entrypoint.sh postgres &

# Espera a que PostgreSQL inicie
until pg_isready -h localhost -U "$POSTGRES_USER"; do
  echo "Esperando a que PostgreSQL inicie..."
  sleep 2
done

# Ejecuta el script de creación de bases de datos
if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
  echo "Creación de múltiples bases de datos solicitada: $POSTGRES_MULTIPLE_DATABASES"
  for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
    echo "Creando base de datos '$db'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
      SELECT 'CREATE DATABASE $db'
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\\gexec
EOSQL
  done
  echo "Bases de datos creadas"
fi

# Espera a que el proceso principal de PostgreSQL termine
wait
