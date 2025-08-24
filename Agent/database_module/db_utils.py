from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.base import Connection
from dotenv import load_dotenv
import pandas as pd
import os
import sys
from database_manager import DatabaseManager


def get_schemas():
    db_name = "employee"
    manager = DatabaseManager(db_name)
    engine = manager.engine

    schemas = {db_name: {}} 
    result = manager.execute_query("SHOW TABLES")
    tables = [row[0] for row in result.fetchall()]
    
    for table_name in tables:
        query = f"DESC {db_name}.{table_name}"
        result = pd.read_sql_query(query, engine)

        schemas[db_name][table_name] = result.to_markdown(index=False)

    return schemas


    