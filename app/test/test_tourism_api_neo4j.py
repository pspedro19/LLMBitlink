import sys
import os
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from neo4j import GraphDatabase
import pytest

class TestNeo4jConnection:
    def setup_method(self):
        """Configuración inicial para cada test"""
        self.uri = "neo4j://localhost:7687"  # o la URI que corresponda
        self.user = "neo4j"
        self.password = "StrongPassword123!"
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def teardown_method(self):
        """Limpieza después de cada test"""
        if self.driver:
            self.driver.close()

    def test_connection(self):
        """Prueba la conexión básica a Neo4j"""
        with self.driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
            print("✓ Conexión básica exitosa")

    def test_create_and_query_node(self):
        """Prueba la creación y consulta de un nodo"""
        with self.driver.session() as session:
            # Crear nodo
            result = session.run("""
                CREATE (d:District {
                    name: 'Test District',
                    description: 'Test Description'
                })
                RETURN d
            """)
            assert result.single() is not None
            print("✓ Creación de nodo exitosa")

            # Consultar nodo
            result = session.run("""
                MATCH (d:District {name: 'Test District'})
                RETURN d.name, d.description
            """)
            record = result.single()
            assert record["d.name"] == "Test District"
            print("✓ Consulta de nodo exitosa")

            # Limpiar
            session.run("""
                MATCH (d:District {name: 'Test District'})
                DELETE d
            """)

    def test_create_relationship(self):
        """Prueba la creación de relaciones entre nodos"""
        with self.driver.session() as session:
            # Crear nodos y relación
            result = session.run("""
                CREATE (d:District {name: 'Punda'})
                CREATE (p:Place {name: 'Queen Emma Bridge'})
                CREATE (p)-[r:LOCATED_IN]->(d)
                RETURN type(r) as relation_type
            """)
            assert result.single()["relation_type"] == "LOCATED_IN"
            print("✓ Creación de relación exitosa")

            # Limpiar
            session.run("""
                MATCH (d:District {name: 'Punda'})
                MATCH (p:Place {name: 'Queen Emma Bridge'})
                DETACH DELETE d, p
            """)

    def test_query_with_parameters(self):
        """Prueba consultas con parámetros"""
        with self.driver.session() as session:
            # Crear datos de prueba
            session.run("""
                CREATE (p:Place {
                    name: 'Test Beach',
                    type: 'beach',
                    rating: 4.5
                })
            """)

            # Consultar con parámetros
            result = session.run("""
                MATCH (p:Place)
                WHERE p.type = $type AND p.rating > $min_rating
                RETURN p.name
            """, type="beach", min_rating=4.0)
            
            record = result.single()
            assert record["p.name"] == "Test Beach"
            print("✓ Consulta con parámetros exitosa")

            # Limpiar
            session.run("""
                MATCH (p:Place {name: 'Test Beach'})
                DELETE p
            """)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])