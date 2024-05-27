# Dockerfile.postgres
FROM postgres:13

COPY entrypoint.sh /entrypoint.sh

# Set environment variables for default user and password
ENV POSTGRES_USER=airflow
ENV POSTGRES_PASSWORD=airflow
RUN apt-get update && apt-get install -y postgresql-13-pgvector

# Instala los paquetes necesarios
RUN apt-get update && \
    apt-get install -y postgresql-server-dev-13 build-essential git

# Clona e instala pgvector
RUN git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install && \
    cd .. && \
    rm -rf pgvector && \
    apt-get remove --purge -y postgresql-server-dev-13 build-essential git && \
    apt-get autoremove -y && \
    apt-get clean

# Copy the database initialization and configuration scripts to the container
COPY ./init_pgvector.sh /docker-entrypoint-initdb.d/
COPY ./entrypoint.sh /docker-entrypoint-initdb.d/
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

# Make sure the scripts are executable
RUN chmod +x /entrypoint.sh
RUN chmod +x /docker-entrypoint-initdb.d/init_pgvector.sh
RUN chmod +x /var/lib/postgresql/data/pg_hba.conf


# Establece el entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Expose the default PostgreSQL port
EXPOSE 5432

# Set the default command for the container to run PostgreSQL server
CMD ["postgres"]
