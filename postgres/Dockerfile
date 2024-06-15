FROM postgres:13

# Instalar paquetes necesarios
RUN apt-get update && \
    apt-get install -y wget gnupg lsb-release software-properties-common && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - && \
    sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    apt-get update && \
    apt-get install -y postgresql-server-dev-13 build-essential git

# Copiar configuraciones y scripts
COPY ./init-db.sql /docker-entrypoint-initdb.d/
COPY ./pg_hba.conf /etc/postgresql/13/main/
COPY ./postgresql.conf /etc/postgresql/13/main/
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh

# Asegurar permisos adecuados
RUN chown -R postgres:postgres /etc/postgresql/13/main && \
    chmod 755 /docker-entrypoint-initdb.d/init-db.sql /usr/local/bin/entrypoint.sh




    
# Establecer el script de entrada personalizado
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["postgres"]

EXPOSE 5432
