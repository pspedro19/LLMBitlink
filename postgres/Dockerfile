# Dockerfile.postgres
FROM postgres:13

# Set environment variables for default user and password
ENV POSTGRES_USER=airflow
ENV POSTGRES_PASSWORD=airflow
ENV POSTGRES_DB=airflow
ENV POSTGRES_MULTIPLE_DATABASES=mlflow_db,airflow_db

# Install necessary packages
RUN apt-get update && \
    apt-get install -y postgresql-server-dev-13 build-essential git llvm-16

# Clone, compile, and install pgvector
RUN git clone https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Set the working directory to pgvector directory to avoid path issues
WORKDIR /pgvector

# Check the contents of the current directory to confirm files are present
RUN ls -la

# Ensure the target directories exist for pgvector files
RUN mkdir -p /usr/share/postgresql/13/extension/ && \
    cp vector.control /usr/share/postgresql/13/extension/ && \
    cp sql/*.sql /usr/share/postgresql/13/extension/

# Reset the working directory to root (optional if more commands follow that need the root directory)
WORKDIR /

# Clean up unnecessary packages to reduce image size
RUN apt-get remove --purge -y build-essential git && \
    apt-get autoremove -y && \
    apt-get clean

# Copy the database initialization and configuration scripts to the container
COPY ./init_pgvector.sh /docker-entrypoint-initdb.d/
COPY ./entrypoint.sh /docker-entrypoint-initdb.d/
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

# Make sure the scripts are executable
RUN chmod +x /docker-entrypoint-initdb.d/init_pgvector.sh
RUN chmod +x /docker-entrypoint-initdb.d/entrypoint.sh

# Establish the entrypoint
ENTRYPOINT ["/docker-entrypoint-initdb.d/entrypoint.sh"]

# Expose the default PostgreSQL port
EXPOSE 5432

# Set the default command for the container to run PostgreSQL server
CMD ["postgres"]
