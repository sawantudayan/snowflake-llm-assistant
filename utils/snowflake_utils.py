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


def insert_embeddings_to_snowflake(chunk_id, document_id, chunk_text, embedding):
    """
    Insert embedding as proper VARIANT using parse_json() wrapper.
    Embedding is passed as a JSON string to be interpreted as VARIANT in Snowflake.
    """
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO GENAI_ASSISTANT.UNSTRUCTURED_DATA.DOCUMENT_CHUNKS 
        (CHUNK_ID, DOCUMENT_ID, CHUNK_TEXT, EMBEDDING)
        SELECT %s, %s, %s, PARSE_JSON(%s)
        """

        embedding_json = json.dumps(embedding)
        logger.info(f"Inserting chunk: {chunk_id}, embedding_dim: {len(embedding)}")

        cursor.execute(insert_query, (chunk_id, document_id, chunk_text, embedding_json))

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error inserting chunk {chunk_id} into Snowflake: {e}")
        raise
