import psycopg2
from psycopg2.extras import execute_values
from transformers import BertTokenizer, BertModel
import numpy as np

class EnhancedVectorMemory:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.setup_database()

    def setup_database(self):
        # Ensure the pgvector extension and table are setup properly
        with self.conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_vectors (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    vector float4[]
                );
                CREATE INDEX IF NOT EXISTS idx_vector ON conversation_vectors USING ivfflat (vector);
            """)
            self.conn.commit()

    def add_to_memory(self, question, answer):
        tokens = self.tokenizer(question, return_tensors="pt")
        vector = self.model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()[0]
        vector_list = vector.tolist()
        with self.conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversation_vectors (question, answer, vector) VALUES (%s, %s, %s)",
                (question, answer, vector_list)
            )
            self.conn.commit()

    def get_closest_memory(self, query):
        tokens = self.tokenizer(query, return_tensors="pt")
        query_vector = self.model(**tokens).last_hidden_state.mean(dim=1).detach().numpy()[0]
        query_list = query_vector.tolist()
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT answer FROM conversation_vectors ORDER BY vector <-> %s LIMIT 1",
                (query_list,)
            )
            result = cursor.fetchone()
        return result[0] if result else None

    def __del__(self):
        self.conn.close()
