import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    # Obtener variables de entorno
    db_user = os.getenv('PG_USER', 'postgres')
    db_password = os.getenv('PG_PASSWORD', 'password')
    db_host = 'localhost'  # El host generalmente será localhost en la mayoría de los setups de contenedores, ajusta según sea necesario
    db_port = os.getenv('PG_PORT', '5432')
    db_name = os.getenv('PG_DATABASE', 'postgres')

    connection = psycopg2.connect(user=db_user,
                                  password=db_password,
                                  host=db_host,
                                  port=db_port,
                                  database=db_name)
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = connection.cursor()

    # Crear rol si no existe
    cursor.execute("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow') THEN CREATE ROLE airflow LOGIN PASSWORD 'your_secure_password' NOCREATEDB NOCREATEROLE NOINHERIT; END IF; END $$;")

    # Crear bases de datos
    cursor.execute("CREATE DATABASE IF NOT EXISTS airflow WITH OWNER airflow;")
    cursor.execute("CREATE DATABASE IF NOT EXISTS mlflow_db WITH OWNER airflow;")

    cursor.close()
    connection.close()

if __name__ == "__main__":
    setup_database()
