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
        database="genai_assistant",
        schema="unstructured_data",
        role=os.getenv("SNOWFLAKE_ROLE")
    )
