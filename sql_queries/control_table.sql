CREATE OR REPLACE TABLE genai_assistant.control_plane.pipeline_status (
    pipeline_id STRING PRIMARY KEY,
    document_id STRING,
    stage STRING,  -- extract, chunk, classify, embed, index
    status STRING, -- success, failed, pending
    log_message STRING,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);