CREATE OR REPLACE TABLE genai_assistant.unstructured_data.raw_documents (
    document_id STRING PRIMARY KEY,
    file_name STRING,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_stage STRING,           -- Optional: S3/Streamlit etc.
    status STRING DEFAULT 'uploaded', -- uploaded | processing | processed | failed
    category STRING,               -- Populated later via CLASSIFY_TEXT
    metadata VARIANT               -- Optional for extra tags
);