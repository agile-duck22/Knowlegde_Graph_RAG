-- Migration 001: Initial Schema for PostgreSQL and pgvector
-- Creates vector extension, documents, chunks, and entity_links tables with HNSW index.

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Table: documents
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(256) NOT NULL UNIQUE,
    source_uri VARCHAR(1024) NOT NULL UNIQUE,
    title VARCHAR(512) NOT NULL,
    doc_type VARCHAR(64) NOT NULL DEFAULT 'technical_doc',
    content_hash CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);

-- 2. Table: chunks
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY, -- Canonical chunk UUID (UUID5 matching Neo4j provenance)
    document_id VARCHAR(256) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536), -- Configurable dimension (default 1536 for OpenAI text-embedding-3-small)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- HNSW Cosine Distance Vector Index (using vector_cosine_ops)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 3. Table: entity_links
CREATE TABLE IF NOT EXISTS entity_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    canonical_entity_id VARCHAR(256) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    mention_text VARCHAR(256) NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entity_links_chunk_id ON entity_links(chunk_id);
CREATE INDEX IF NOT EXISTS idx_entity_links_canonical_id ON entity_links(canonical_entity_id);
