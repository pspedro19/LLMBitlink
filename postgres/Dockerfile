FROM postgres:13

USER root

# Copy requirements file and scripts
COPY init_pgvector.sh /docker-entrypoint-initdb.d/init_pgvector.sh
COPY entrypoint.sh /entrypoint.sh
COPY pg_hba.conf /var/lib/postgresql/data/pg_hba.conf
COPY postgresql.conf /var/lib/postgresql/data/postgresql.conf

# Ensure necessary scripts are executable
RUN chmod +x /docker-entrypoint-initdb.d/init_pgvector.sh \
    && chmod +x /entrypoint.sh

# Clone the pgvector repository and set up the extension
RUN apt-get update && apt-get install -y postgresql-server-dev-13 build-essential git \
    && git clone https://github.com/pgvector/pgvector.git /pgvector \
    && cd /pgvector && make && make install \
    && mkdir -p /usr/share/postgresql/13/extension/ \
    && cp vector.control /usr/share/postgresql/13/extension/ \
    && cd / && rm -rf /pgvector \
    && apt-get remove --purge -y build-essential git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Set permissions for configuration files
RUN chmod -R 777 /var/lib/postgresql/data/pg_hba.conf \
    && chmod -R 777 /var/lib/postgresql/data/postgresql.conf

# Set custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Expose the default PostgreSQL port
EXPOSE 5432

# Command to run the PostgreSQL server
CMD ["postgres"]
