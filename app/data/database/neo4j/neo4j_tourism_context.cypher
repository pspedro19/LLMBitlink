// Ejemplo de consultas para Neo4j (travel_routes.json)
// Búsqueda de eventos por temporada
MATCH (e:Event)-[:OCCURS_IN]->(s:Season)
WHERE s.name = 'High Season'
RETURN e

// Eventos y ubicaciones
MATCH (e:Event)-[:TAKES_PLACE_AT]->(l:Location)
WHERE e.type = 'Festival'
RETURN e, l

// Rutas de eventos
MATCH path = (s:Location)-[:PART_OF_ROUTE]->(r:Route)
WHERE r.event = 'Carnival Parade'
RETURN path


// Ejemplo de consultas para Neo4j (weather_patterns.json)
// Consulta de condiciones por temporada
MATCH (s:Season)-[:HAS_CONDITION]->(w:Weather)
WHERE s.name = 'Dry Season'
RETURN s, w

// Actividades recomendadas según clima
MATCH (w:Weather)-[:SUITABLE_FOR]->(a:Activity)
WHERE w.type = 'Clear' AND w.temperature < 30
RETURN a

// Patrones marítimos
MATCH (l:Location)-[:HAS_CONDITION]->(m:Marine)
WHERE m.visibility > 20
RETURN l, m



