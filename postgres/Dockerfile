# postgres/Dockerfile
FROM postgres:latest

# Set environment variables for default user and password
ENV POSTGRES_USER=airflow
ENV POSTGRES_PASSWORD=airflow
RUN apt-get update && apt-get install -y postgresql-13-pgvector

# Install pgvector extension
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates build-essential postgresql-server-dev-all libpq-dev \
    && wget https://github.com/pgvector/pgvector/archive/refs/tags/v0.2.1.tar.gz \
    && tar -xzf v0.2.1.tar.gz \
    && cd pgvector-0.2.1 && make && make install \
    && cd .. && rm -rf pgvector-0.2.1 v0.2.1.tar.gz \
    && apt-get remove -y wget ca-certificates build-essential postgresql-server-dev-all \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the database initialization and configuration scripts to the container
COPY init_pgvector.sh /docker-entrypoint-initdb.d/10_init_pgvector.sh
COPY create-multiple-postgresql-databases.sh /docker-entrypoint-initdb.d/20_create_multiple_dbs.sh
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

# Make sure the scripts are executable
RUN chmod +x /docker-entrypoint-initdb.d/10_init_pgvector.sh
RUN chmod +x /docker-entrypoint-initdb.d/20_create_multiple_dbs.sh
RUN chmod +x /var/lib/postgresql/data/pg_hba.conf

# Expose the default PostgreSQL port
EXPOSE 5432

# Set the default command for the container to run PostgreSQL server
CMD ["postgres"]
