CREATE OR REPLACE TABLE genai_assistant.unstructured_data.document_chunks (
    chunk_id STRING PRIMARY KEY,
    document_id STRING REFERENCES genai_assistant.unstructured_data.raw_documents(document_id),
    chunk_index INT,
    chunk_text STRING,
    embedding VECTOR(FLOAT, 768),  -- Reserved for later
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);