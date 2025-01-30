// Ejemplo de consultas para Neo4j (districts_info.json)
MATCH (d:District)-[:CONTAINS]->(a:Attraction)
WHERE d.name = 'Punda'
RETURN d, a

// Rutas entre distritos
MATCH path = (d1:District)-[:CONNECTED_TO]->(d2:District)
WHERE d1.name = 'Punda' AND d2.name = 'Otrobanda'
RETURN path

// Búsqueda de atracciones por tipo
MATCH (d:District)-[:HAS]->(a:Attraction)
WHERE a.type = 'Historical' AND d.unesco_status = true
RETURN d.name, COLLECT(a.name)



// Ejemplo de consultas para Neo4j (beaches_data.json)
MATCH (b:Beach)-[:HAS_ACTIVITY]->(a:Activity)
WHERE b.type = 'Natural bay' AND a.type = 'snorkeling'
RETURN b, a

// Rutas y accesos
MATCH (b:Beach)-[:ACCESS_FROM]->(l:Location)
WHERE b.name = 'Playa Knip'
RETURN b, l

// Facilidades y servicios
MATCH (b:Beach)-[:OFFERS]->(f:Facility)
WHERE b.name = 'Mambo Beach'
RETURN b, COLLECT(f.type)



// Ejemplo de consultas para Neo4j (travel_routes.json)
MATCH (r:Route)-[:INCLUDES]->(p:POI)
WHERE r.type = 'Walking tour'
RETURN r, p

// Conexiones entre distritos
MATCH path = shortestPath((a:Location)-[:CONNECTED_TO*]->(b:Location))
WHERE a.name = 'Airport' AND b.name = 'Willemstad'
RETURN path

// Puntos de interés en una ruta
MATCH (r:Route)-[:HAS_STOP]->(s:Stop)
WHERE r.name = 'West Coast Beach Route'
RETURN r, COLLECT(s)