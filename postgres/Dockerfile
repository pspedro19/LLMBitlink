FROM postgres:13

USER root

# Copy configuration files and scripts
COPY init_pgvector.sh /docker-entrypoint-initdb.d/init_pgvector.sh
COPY entrypoint.sh /entrypoint.sh
COPY pg_hba.conf /etc/postgresql/pg_hba.conf
COPY postgresql.conf /etc/postgresql/postgresql.conf

# Ensure necessary scripts are executable
RUN chmod +x /docker-entrypoint-initdb.d/init_pgvector.sh && \
    chmod +x /entrypoint.sh

# Install necessary tools and build pgvector
RUN apt-get update && \
    apt-get install -y git iputils-ping telnet build-essential postgresql-server-dev-13 && \
    rm -rf /pgvector && \
    git clone https://github.com/pgvector/pgvector.git /pgvector && \
    cd /pgvector && \
    make && \
    make install 

# Copy pgvector control file to the appropriate directory
RUN mkdir -p /usr/share/postgresql/13/extension/ && \
    cp /pgvector/vector.control /usr/share/postgresql/13/extension/

# Cleanup
RUN apt-get remove --purge -y build-essential git postgresql-server-dev-13 && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /pgvector

# Set permissions for configuration files
RUN chmod 644 /etc/postgresql/pg_hba.conf && \
    chmod 644 /etc/postgresql/postgresql.conf && \
    chmod 700 /var/lib/postgresql/data

# Set custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Expose the default PostgreSQL port
EXPOSE 5432

# Command to run the PostgreSQL server
CMD ["postgres"]
