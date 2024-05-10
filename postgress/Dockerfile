# Use the official PostgreSQL image from the Docker Hub
FROM postgres:latest
RUN chmod +x create-multiple-postgresql-databases.sh

# Set environment variables for default user and password
ENV POSTGRES_USER=airflow
ENV POSTGRES_PASSWORD=airflow

# Copy the database initialization script to the container
COPY create-multiple-postgresql-databases.sh /docker-entrypoint-initdb.d/

# Copy the custom pg_hba.conf file into the container
# Note: Adjust the target directory path according to the PostgreSQL version if different
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf

# Make sure the initialization script is executable
RUN chmod +x /docker-entrypoint-initdb.d/create-multiple-postgresql-databases.sh

# Expose the default PostgreSQL port
EXPOSE 5432

# Set the default command for the container to run PostgreSQL server
CMD ["postgres"]
