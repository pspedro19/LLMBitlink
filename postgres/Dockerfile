FROM postgres:latest

# Set environment variables for default user and password
ENV POSTGRES_USER=airflow
ENV POSTGRES_PASSWORD=airflow

# Copy the database initialization and configuration scripts to the container
COPY create-multiple-postgresql-databases.sh /docker-entrypoint-initdb.d/
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

# Make sure the scripts are executable
RUN chmod +x /docker-entrypoint-initdb.d/create-multiple-postgresql-databases.sh
RUN chmod +x /var/lib/postgresql/data/pg_hba.conf

# Expose the default PostgreSQL port
EXPOSE 5432

# Set the default command for the container to run PostgreSQL server
CMD ["postgres"]
