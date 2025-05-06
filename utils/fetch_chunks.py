from utils.snowflake_connector import get_snowflake_connection

def fetch_chunks_from_snowflake():
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            c.chunk_id,
            c.chunk_text,
            d.file_name,
            d.category,
            d.upload_timestamp
        FROM 
            genai_assistant.unstructured_data.document_chunks c
        JOIN 
            genai_assistant.unstructured_data.raw_documents d
        ON 
            c.document_id = d.document_id
    """)
    
    results = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    
    # Convert to list of dicts for easier processing
    return [dict(zip(column_names, row)) for row in results]
