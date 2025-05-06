from utils.snowflake_connector import get_snowflake_connection

def test_connection():
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_TIMESTAMP;")
        result = cursor.fetchone()
        print(f"✅ Snowflake connected successfully. Server time: {result[0]}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Snowflake connection failed: {e}")

if __name__ == "__main__":
    test_connection()