from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine.base import Connection
from dotenv import load_dotenv
import pandas as pd
import os

class DatabaseManager:
    def __init__(self, db_name="employee"):
        self.env_path = r"C:\Users\aniru\Downloads\agentic_framework_mcp\text2sql_agent_mcp\.env"
        load_dotenv(self.env_path)
        self.password = os.getenv("MYSQL_ROOT_PASSWORD")
        self.db_name = db_name
        self.engine = create_engine(
            f"mysql+pymysql://root:{self.password}@localhost:56582/{self.db_name}"
        )
        self.connection = None
    
    def establish_connection(self):
        if not self.connection:
            try:
                self.connection = self.engine.connect()
                
                # Second query on same connection
                result = self.connection.execute(text("USE employee"))
                result = self.connection.execute(text("SHOW TABLES"))
                databases = [row[0] for row in result.fetchall()]
                print("✅ Connected to MySQL Database:", databases)
                            
            except SQLAlchemyError as e:
                
                print(f"❌ Connection failed: {e}")
                self.connection = "Connection Failed"

        return self.connection
    
    def execute_query(self, query):

        if self.connection is None:
            conn = self.establish_connection()
            return conn.execute(text(query))
        
        elif isinstance(self.connection, Connection):
            return self.connection.execute(text(query))
    
        else:
            return self.connection
    
    def close(self):
        if isinstance(self.connection, Connection):
            self.connection.close()
            self.connection = None

# Usage examples:
if __name__ == "__main__":

    db = DatabaseManager()
    try:
        result = db.execute_query("SELECT VERSION()")
        print(f"✅ Version: {result.fetchone()[0]}")
        
        result = db.execute_query("SHOW TABLES")
        databases = [row[0] for row in result.fetchall()]
        print("Databases:", databases)
        df = pd.read_sql("SELECT * FROM employee", db.engine)
        print(f"Dataframe: {df}")
    finally:
        db.close()