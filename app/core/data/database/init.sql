-- data/database/init.sql

-- Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla para documentos
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    original_content TEXT,
    file_path TEXT,
    mime_type VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla para chunks
CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    page_number INTEGER,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(doc_id, chunk_number)
);

-- Tabla para embeddings
CREATE TABLE embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    embedding vector(384),       -- Ajusta la dimensión según el modelo que utilices
    faiss_index_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);

-- Índice vectorial utilizando ivfflat
CREATE INDEX idx_embeddings_vector
    ON embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
