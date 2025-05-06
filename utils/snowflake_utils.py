import os

import snowflake.connector
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


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
