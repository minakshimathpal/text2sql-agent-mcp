import asyncio
import os
import sys
# Ensure Agent package path is importable
sys.path.insert(0, r"e:/Capstone_Project/text2sql-agent-mcp/Agent")
import backups.agentic_workflow4 as aw

async def run_one():
    # Initialize DB (use the module's configured DB_CONNECTION_URL when available)
    db_uri = os.environ.get('DB_CONNECTION_URL') or getattr(aw, 'DB_CONNECTION_URL', None)
    await aw.initialize_database(db_uri)
    # Run a single question
    # Prefix with 're-execute' to force regeneration and bypass any cached error in memory
    res = await aw.agentic_query_process('re-execute Who are the top 10 highest-paid employees?')
    print('\n--- AGENT OUTPUT START ---')
    print(res)
    print('--- AGENT OUTPUT END ---')

if __name__ == '__main__':
    asyncio.run(run_one())
