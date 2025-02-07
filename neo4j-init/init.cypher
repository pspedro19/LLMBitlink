CREATE INDEX IF NOT EXISTS FOR (n:Document) ON (n.id);
CREATE INDEX IF NOT EXISTS FOR (n:Chunk) ON (n.text);
CALL apoc.schema.assert({}, {});