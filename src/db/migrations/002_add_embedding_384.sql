-- Migration 002: Add embedding_384 column and index for local sentence-transformers models
-- Preserves existing chunks.embedding VECTOR(1536) and all table data.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_384 VECTOR(384);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_384_hnsw
ON chunks USING hnsw (embedding_384 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
