// Ejemplo de consultas para Neo4j (historical_events.json)

MATCH (l:Location)-[:HAS_HISTORICAL_CONTEXT]->(h:HistoricalEvent)
WHERE h.period = "dutch_colonial"
RETURN l, h

// Ejemplo de consultas para Neo4j (cultural_traditions.json)
MATCH (e:Event)-[:HAS_TRADITION]->(t:Tradition)
WHERE t.type = 'carnival'
RETURN e, t

MATCH (l:Location)-[:HOSTS_FESTIVAL]->(f:Festival)
WHERE f.season = 'Carnival'
RETURN l, f

// Ejemplo de consultas para Neo4j (local_customs.json)
MATCH (c:Custom)-[:APPLIES_TO]->(l:Location)
WHERE l.name = 'Willemstad' AND c.type = 'dining'
RETURN c, l

MATCH (e:Etiquette)-[:REQUIRED_AT]->(v:Venue)
WHERE v.type = 'religious_site'
RETURN e, v