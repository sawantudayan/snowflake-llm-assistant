import json
import logging
import os

import snowflake.connector
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
logger = logging.getLogger(__name__)


def get_snowflake_connection():
    """
    Returns a Snowflake connection using environment variables.
    """
    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )


def fetch_chunks_from_snowflake():
    """
    Fetches document chunks from Snowflake to be used for FAISS index creation.
    """
    conn = get_snowflake_connection()
    cursor = conn.cursor()

    # Query to retrieve chunk details
    query = """
    SELECT chunk_id, document_id, chunk_text
    FROM genai_assistant.unstructured_data.document_chunks
    """
    cursor.execute(query)

    chunks = []
    for row in cursor.fetchall():
        chunks.append({
            "chunk_id": row[0],
            "document_id": row[1],
            "chunk_text": row[2]
        })

    cursor.close()
    conn.close()

    return chunks


def insert_embeddings_to_snowflake(chunk_id, document_id, chunk_text, embedding, chunk_index=None):
    """
    Insert or update document chunk and its embedding into Snowflake.
    Embedding is automatically serialized to JSON and stored as a VARIANT.

    Parameters:
    - chunk_id: str
    - document_id: str
    - chunk_text: str
    - embedding: List[float] or np.ndarray
    - chunk_index: int or None
    """
    try:
        # Serialize embedding to JSON string
        embedding_json = json.dumps(embedding)

        # Establish connection
        conn = get_snowflake_connection()
        cursor = conn.cursor()

        # MERGE query to UPSERT (insert or update) embedding
        merge_query = """
        MERGE INTO GENAI_ASSISTANT.UNSTRUCTURED_DATA.DOCUMENT_CHUNKS AS target
        USING (
            SELECT 
                %s AS chunk_id, 
                %s AS document_id, 
                %s AS chunk_text, 
                PARSE_JSON(%s) AS embedding, 
                %s AS chunk_index
        ) AS source
        ON target.chunk_id = source.chunk_id
        WHEN MATCHED THEN
            UPDATE SET 
                target.embedding = source.embedding,
                target.chunk_index = COALESCE(target.chunk_index, source.chunk_index)
        WHEN NOT MATCHED THEN
            INSERT (chunk_id, document_id, chunk_text, embedding, chunk_index)
            VALUES (source.chunk_id, source.document_id, source.chunk_text, source.embedding, source.chunk_index);
        """

        # Execute with params
        cursor.execute(merge_query, (
            chunk_id,
            document_id,
            chunk_text,
            embedding_json,
            chunk_index
        ))

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✅ Inserted/updated chunk {chunk_id} with embedding into Snowflake.")

    except Exception as e:
        print(f"❌ Error inserting/updating chunk {chunk_id} into Snowflake: {e}")
