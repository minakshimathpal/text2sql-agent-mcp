import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent import ClientSession
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.graph_stores import SimpleGraphStore
from toolbox.raw_db_tools import RawDatabaseToolSpec
from config import Config
import os
import json
import uuid
from toolbox.prompt import  prompt_template_str1, prompt_template_str2, prompt_template_str3

def main():# Ensure JSON files exist
    for path in [Config.Path.chat_store_private, Config.Path.chat_store_public]:
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump({}, f)

    # Initialize dependencies
    chat_store = SimpleChatStore()
    db_spec = RawDatabaseToolSpec(Config.Path.Database_path)  # Adjust with your database connection details
    graph_store = SimpleGraphStore()

    # Create ClientSession
    session = ClientSession(
        session_id= str(uuid.uuid4()) ,  
        db_spec=db_spec
    
    )
    print("session created")
    print("graph store created")
    print("db spec created")
    print("chat store created")

    # Process a query
    queries = ["What badge did most users earn?",
            "What are different types of badges available?",# which is the less common type
            "list all the unique badges available"
            "Which top 10 users have earned the most"
            "Which top 10 users who have earned the most unique badges?",
            "What are the rarest badges that fewer than 10 users have earned?",
            "Which users have earned specific badges,and what are their reputation scores",
            "What comments were made on posts that are linked to other posts (via postlinks) by users with a reputation greater than 1000"]
    query="Which users have earned specific badges,and what are their reputation scores"
    response = session.process_query(
        query=query,
        prompt_template_str1=prompt_template_str1,
        prompt_template_str2=prompt_template_str2,
        prompt_template_str3=prompt_template_str3
    )

    # Print response
    print("Final Response:", response)

# Persist chat stores and graph store
if __name__ == "__main__":
    main()

