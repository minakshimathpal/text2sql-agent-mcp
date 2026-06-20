import os
import asyncio
from sqlalchemy import create_engine, MetaData, Table, inspect
from sqlalchemy.ext.asyncio import create_async_engine

async def inspect_db():
    db_url = os.environ.get("DB_CONNECTION_URL")
    if not db_url:
        print("DB_CONNECTION_URL not set")
        return

    print(f"Inspecting DB: {db_url}")
    
    # Use sync engine for inspection as it's easier
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    if sync_url.startswith("postgresql://"):
        pass
    else:
        # fallback
        sync_url = db_url
        
    try:
        engine = create_engine(sync_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables: {tables}")
        
        for table in tables:
            columns = inspector.get_columns(table)
            col_names = [c['name'] for c in columns]
            print(f"Table '{table}': {col_names}")
            
    except Exception as e:
        print(f"Error during inspection: {e}")

if __name__ == "__main__":
    asyncio.run(inspect_db())