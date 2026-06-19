from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
import os
import asyncio
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.agent.workflow import FunctionAgent
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, text, Table, MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.schema import CreateTable
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage, MessageRole
import uuid
from pathlib import Path
import json
# from llama_index.text_splitter import TextBlock

# Initialize LLM
load_dotenv()
from llama_index.llms.azure_openai import AzureOpenAI

# Simple initialization without context
DB_CONNECTION_URL = os.environ.get("DB_CONNECTION_URL", None)
if not DB_CONNECTION_URL:
    raise ValueError("DB_CONNECTION_URL environment variable is not set")

AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "llama3.1:8b")
AZURE_OPENAI_ENGINE_NAME = os.environ.get("AZURE_OPENAI_ENGINE_NAME", "gpt-4o-mini")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3.1:8b")

# Global variables
available_tables = []
inspector = None
sql_database = None
async_engine = None
async_session = None

# Initialize chat store directory and files
chat_store_dir = Path(r'D:\Airtel bot\text-to-sql')
chat_store_dir.mkdir(parents=True, exist_ok=True)

# Define chat store file paths
public_store_path = chat_store_dir/'chat_store_public.json'
private_store_path = chat_store_dir/'chat_store_private.json'
# Initialize empty chat store structure with conversation key
initial_store = {"conversation": []}

# Create JSON files if they don't exist
if not public_store_path.exists():   
    with open(public_store_path, 'w') as f:
        json.dump(initial_store, f, indent=2)

if not private_store_path.exists():    
    with open(private_store_path, 'w') as f:
        json.dump(initial_store, f, indent=2)

# Initialize chat stores
print("Debug: Initializing chat stores")
chat_store = SimpleChatStore()
chat_store_public = chat_store.from_persist_path(str(public_store_path))
chat_store_private = chat_store.from_persist_path(str(private_store_path))
print("Debug: Chat stores initialized")

async def initialize_database(
    uri: Optional[str] = None,
    scheme: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    dbname: Optional[str] = None,
    tables: Optional[List[str]] = None,
    *args: Optional[Any],
    **kwargs: Optional[Any]
) -> Dict[str, Any]:
    """Initialize the database connection and return metadata."""
    global available_tables, inspector, sql_database, async_engine, async_session
    metadata = {
        "uri": None,
        "tables": [],
        "sql_database": None,
        "inspector": None,
        "session": None,
        "initialization_status": {"status": "pending", "tables_initialized": [], "errors": []}
    }
    
    try:
        # Construct URI
        if uri:
            metadata["uri"] = uri
        elif scheme and host and port and user and password and dbname:
            metadata["uri"] = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
        else:
            raise ValueError(
                "You must provide either a valid connection URI or a valid set of credentials "
                "(scheme, host, port, user, password, dbname)."
            )
        
        # Convert PostgreSQL URI to use asyncpg driver
        if 'postgresql' in metadata["uri"]:
            metadata["uri"] = metadata["uri"].replace('postgresql://', 'postgresql+asyncpg://')
        
        # Create sync engine for inspector (using psycopg2)
        sync_uri = metadata["uri"].replace('+asyncpg://', '://')
        metadata["sql_database"] = create_engine(sync_uri, *args, **kwargs)
        sql_database = metadata["sql_database"]
        
        # Create async engine for queries
        async_engine = create_async_engine(metadata["uri"], *args, **kwargs)
        
        # Create async session factory
        async_session = sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create inspector
        metadata["inspector"] = inspect(metadata["sql_database"].engine)
        inspector = metadata["inspector"]
        
        # Get available tables
        available_tables = tables or metadata["inspector"].get_table_names() + metadata["inspector"].get_view_names()
        
        # Initialize tables
        for table in available_tables:
            metadata_obj = MetaData()
            try:
                table_obj = Table(table, metadata_obj, autoload_with=metadata["sql_database"])
                metadata["tables"].append(table_obj)
                metadata["initialization_status"]["tables_initialized"].append(table)
                print(f"Initialized table: {table}")
            except NoSuchTableError as e:
                print(f'"{e}" table not found!')
                metadata["initialization_status"]["errors"].append({
                    "table": table,
                    "error": str(e),
                    "type": "NoSuchTableError"
                })
                continue
        
        # Update status
        metadata["initialization_status"]["status"] = "completed"
        if not metadata["tables"]:
            metadata["initialization_status"]["errors"].append({
                "error": "No tables found in the database.",
                "type": "EmptyDatabase"
            })
        
        return metadata
        
    except Exception as e:
        metadata["initialization_status"]["errors"].append({
            "error": str(e),
            "type": "InitializationError"
        })
        metadata["initialization_status"]["status"] = "failed"
        return metadata

async def list_tables(ctx: Context) -> str:
        """
        Returns a list of available tables in the database.
        To retrieve details about the columns of specific tables, use
        the describe_tables endpoint.
        """
        global available_tables, inspector
        current_state = await ctx.get("state")
        try:
            table_list = list()
            for table_name in available_tables:
                table_info = inspector.get_table_comment(table_name=table_name).get('text', '')
                table_info = f'Table Description: {table_info}\n' if table_info else ''
                table_list.append(f'Table Name: {table_name}\n{table_info}\n')
            
            result = ''.join(table_list)
            current_state["available_tables"] = result
            await ctx.set("state", current_state)
            return result
        except Exception as e:
            current_state["last_error"] = str(e)
            await ctx.set("state", current_state)
            return f"Error listing tables: {str(e)}"

async def describe_tables(ctx: Context, tables: Optional[List[str]] = None) -> str:
        """
        Describes the specified tables in the database.

        Args:
            ctx: The context object containing state
            tables (List[str]): A list of multiple table names to retrieve details about
        """
        global available_tables, inspector, sql_database
        current_state = await ctx.get("state")
        try:
            table_names = tables or available_tables
            table_schemas = []

            for table_name in table_names:
                try:
                    # Create a new metadata object for each table
                    metadata_obj = MetaData()
                    # Create table object
                    table = Table(table_name, metadata_obj, autoload_with=sql_database)
                    
                    # Get schema
                    schema = str(CreateTable(table).compile(sql_database.engine))
                    
                    # Get columns
                    columns = inspector.get_columns(table_name=table_name)
                    column_info = '\n'
                    for column in columns:
                        comment = column.get("comment", "")
                        if comment:
                            column_info += f'{comment}\n'
                    
                    table_schemas.append(f"{schema}{column_info}\n")
                except NoSuchTableError as e:
                    print(f"Table '{table_name}' not found: {str(e)}")
                    continue

            if not table_schemas:
                return "No valid tables found to describe."

            result = "\n".join(table_schemas)
            current_state["last_schema"] = result
            await ctx.set("state", current_state)
            return result
        except Exception as e:
            current_state["last_error"] = str(e)
            await ctx.set("state", current_state)
            return f"Error describing tables: {str(e)}"

async def load_data(ctx: Context, query: str) -> List[str]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            ctx: The context object containing state
            query (str): an SQL query to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.
        """
        global async_session
        current_state = await ctx.get("state")
        try:
            documents = []
            print(query)
            
            async with async_session() as session:
                if query is None:
                    raise ValueError("A query parameter is necessary to filter the data")
                else:
                    result = await session.execute(text(query))
                    rows = result.fetchall()

                for item in rows:
                    # fetch each item
                    doc_str = ", ".join([str(entry) for entry in item])
                    documents.append(doc_str)
            
            current_state["last_results"] = documents
            await ctx.set("state", current_state)
            return documents
        except Exception as e:
            current_state["last_error"] = str(e)
            await ctx.set("state", current_state)
            return [f"Error loading data: {str(e)}"]

async def validate_sql_query(ctx: Context, query: str) -> dict:
        """
        Validate a SQL query, without executing it.
        
        Args:
            ctx: The context object containing state
            query (str): The SQL query to validate
            
        Returns:
            dict: A dictionary containing validation results with:
                - valid (bool): Whether the query is valid
                - reason (str): Description of error if invalid
                - plan (str): The execution plan if valid
        """
        if DB_CONNECTION_URL:
            try:
                # Get current state
                current_state = await ctx.get("state")
                
                # PostgreSQL
                if 'postgresql' in DB_CONNECTION_URL:
                    explain_query = f"EXPLAIN {query}"
                    results = await load_data(ctx, explain_query)
                    if results:
                        # Update state with validation results
                        current_state["last_validation"] = {"valid": True, "reason": "Query syntax is valid", "plan": results}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]
                    else:
                        current_state["last_validation"] = {"valid": False, "reason": "Invalid query"}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]

                # MySQL
                elif 'mysql' in DB_CONNECTION_URL:
                    explain_query = f"EXPLAIN {query}"
                    results = await load_data(ctx, explain_query)
                    if results:
                        current_state["last_validation"] = {"valid": True, "reason": "Query syntax is valid", "plan": results}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]
                    else:
                        current_state["last_validation"] = {"valid": False, "reason": "Invalid query"}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]

                # SQLite
                elif 'sqlite' in DB_CONNECTION_URL:
                    explain_query = f"EXPLAIN QUERY PLAN {query}"
                    results = await load_data(ctx, explain_query)
                    if results:
                        current_state["last_validation"] = {"valid": True, "reason": "Query syntax is valid", "plan": results}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]
                    else:
                        current_state["last_validation"] = {"valid": False, "reason": "Invalid query"}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]

                # MS SQL Server
                elif 'sqlserver' in DB_CONNECTION_URL or 'mssql' in DB_CONNECTION_URL:
                    explain_query = f"SET SHOWPLAN_ALL ON; {query} SET SHOWPLAN_ALL OFF;"
                    results = await load_data(ctx, explain_query)
                    if results:
                        current_state["last_validation"] = {"valid": True, "reason": "Query syntax is valid", "plan": results}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]
                    else:
                        current_state["last_validation"] = {"valid": False, "reason": "Invalid query"}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]

                # Oracle
                elif 'oracle' in DB_CONNECTION_URL:
                    explain_query = f"EXPLAIN PLAN FOR {query}"
                    await load_data(ctx, explain_query)
                    plan_query = "SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)"
                    results = await load_data(ctx, plan_query)
                    if results:
                        current_state["last_validation"] = {"valid": True, "reason": "Query syntax is valid", "plan": results}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]
                    else:
                        current_state["last_validation"] = {"valid": False, "reason": "Invalid query"}
                        await ctx.set("state", current_state)
                        return current_state["last_validation"]

                else:
                    current_state["last_validation"] = {"valid": False, "reason": "Query validation not supported for this database type"}
                    await ctx.set("state", current_state)
                    return current_state["last_validation"]
            
            except Exception as e:
                error_message = str(e)
                print(f"Error validating query: {error_message}")
                current_state["last_validation"] = {"valid": False, "reason": f"Error validating query: {error_message}"}
                await ctx.set("state", current_state)
                return current_state["last_validation"]
        else:
            return {"valid": False, "reason": "Database URI not provided"}

async def feedback(ctx: Context, details: str):
        """Records feedback about the current query.

        Args:
            ctx: The context object containing state
            details (str): Feedback details
        """
        current_state = await ctx.get("state")
        if "feedback_history" not in current_state:
            current_state["feedback_history"] = []
        
        feedback_entry = {
            "query": current_state.get("user_query", ""),
            "message": details,
            "timestamp": datetime.now().isoformat()
        }
        current_state["feedback_history"].append(feedback_entry)
        await ctx.set("state", current_state)
        
        feedback_str = f"FEEDBACK: {details} Unable to find the requested information. Please contact support."
        return feedback_str

azure_openai_llm = AzureOpenAI(
model=AZURE_OPENAI_MODEL_NAME,
engine=AZURE_OPENAI_ENGINE_NAME,
temperature=0.3,
max_tokens=200
)

def generate_session_id() -> str:
    """Generate a unique session ID using UUID4."""
    return str(uuid.uuid4())

def get_chat_history(session_id: str) -> tuple:
    """
    Get existing session ID from chat stores or create a new one.
    Only creates session info when chat JSONs are first created.
    Returns the session ID.
    """    
    # Try to get existing session from public store
    public_messages = chat_store_public.get_messages(key="conversation")
    private_messages = chat_store_private.get_messages(key="conversation")    
    # If no messages exist for this session, initialize them
    if not public_messages:       
        # Create session message
        session_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=session_id
        )
               
        chat_store_public.add_message(
            key="conversation",
            message=session_message
        )
        # Persist immediately after adding message
        chat_store_public.persist(str(public_store_path))
        public_messages = chat_store_public.get_messages(key="conversation")
        
    if not private_messages:       
        # Create session message
        session_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=session_id
        )
              
        chat_store_private.add_message(
            key="conversation",
            message=session_message
        )
        # Persist immediately after adding message
        chat_store_private.persist(str(private_store_path))
        private_messages = chat_store_private.get_messages(key="conversation")
           
    return public_messages, private_messages, session_id

# Get or create session

public_chat_history,private_chat_history,session_id = get_chat_history(generate_session_id())
# Update initial state structure
initial_state = {
    "session_id": generate_session_id(),  # Initialize with a session ID
    "user_query": "",  # Will be updated with each query
    "result": ""  # Will store the result
}

async def fetch_context(ctx: Context, session_id: str, user_query: str) -> dict:
    """
    Fetches the current context from state and chat history.
    Returns a dictionary containing:
    - current_state: The current state from context
    - chat_history: The chat history from SimpleChatStore
    """
    try:  
        # Get current state
        current_state = await ctx.get("state")           
        # Update state with new query
        current_state["user_query"] = user_query
        await ctx.set("state", current_state)
        
        # Get chat history
        public_messages = chat_store_public.get_messages(key="conversation")
        private_messages = chat_store_private.get_messages(key="conversation")
        
        
        # # Print message contents for debugging
        # for msg in private_messages:
        #     print(f"Debug: Private message - Role: {msg.role}, Content: {msg.content}")
        
        return {
            "current_state": current_state,
            "chat_history": {
                "private": private_messages,
                "public": public_messages
            }
        }
    except Exception as e:
        print(f"Error fetching context: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "current_state": initial_state.copy(),
            "chat_history": {"private": [], "public": []}
        }

async def update_context(ctx: Context, query: str, result: str) -> None:
    """
    Updates both the state context and chat stores with new query and result.
    Only stores the final result from ResultPresenterAgent.
    
    Args:
        ctx: The context object
        query: The user's query
        result: The result/answer to the query
    """
    try:       
        # Get current state
        current_state = await ctx.get("state")       
        # Get session_id from state or generate new one if not exists
        session_id = current_state.get("session_id")
        if not session_id:
            session_id = generate_session_id()
            current_state["session_id"] = session_id         
        # Update state
        current_state["user_query"] = query
        current_state["result"] = result
        await ctx.set("state", current_state)
              
        # Create session message
        session_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=session_id
        )        
        # Create user message
        user_message = ChatMessage(
            role=MessageRole.USER,
            content=query
        )        
        # Create assistant message
        assistant_message = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=result
        )
              
        # Get existing messages first
        existing_messages = chat_store_private.get_messages(key="conversation")     
        # Update chat store with all messages    
        chat_store_private.add_message(key="conversation", message=session_message)
        chat_store_private.add_message(key="conversation", message=user_message)
        chat_store_private.add_message(key="conversation", message=assistant_message)
        
        # Verify messages were added
        updated_messages = chat_store_private.get_messages(key="conversation")       
        # Persist changes immediately
        persist_path = str(private_store_path)       
        # Ensure the directory exists
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        
        # Write to file directly first to ensure we have write permissions
        # try:
        #     # Convert messages to a format that can be serialized
        #     messages_to_save = []
        #     for msg in updated_messages:
        #         message_dict = {
        #             "role": msg.role.value,
        #             "content": msg.content
        #         }
        #         messages_to_save.append(message_dict)
            
        #     with open(persist_path, 'w') as f:
        #         json.dump({"conversation": messages_to_save}, f, indent=2)
        #     print(f"Debug: Successfully wrote to file directly")
        # except Exception as e:
        #     print(f"Debug: Error writing to file directly: {str(e)}")
        
        # Now try the persist method
        try:
            chat_store_private.persist(str(persist_path))
            print(f"Debug: Successfully persisted using chat_store_private.persist")
        except Exception as e:
            print(f"Debug: Error persisting using chat_store_private.persist: {str(e)}")
        
        # Verify the file was written
        if os.path.exists(persist_path):
            with open(persist_path, 'r') as f:
                content = f.read()
                print(f"Debug: File content after write: {content}")
        else:
            print(f"Debug: File not found at {persist_path}")
            
        print("=== Debug: update_context completed ===\n")
        
    except Exception as e:
        print(f"Error updating context: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

# Setup agents with the initialized db_spec
research_agent = FunctionAgent(
    name="MemoryAgent",
    description="Responsible for retrieving relevant answers from conversation history, providing consistent responses to previously addressed queries while efficiently routing new questions to specialized agents.",
system_prompt= """
1. You are an AI assistant responsible for evaluating whether you can answer a user's query based on prior chat history and current state.
2. You have access to the chat history and current state through fetch_context tool, which provides:
   - private messages in chat_history["private"]
   - public messages in chat_history["public"]
   - current state in current_state
3. Your role is to:
   - Check if the current query matches any previous query in the chat history or current state
   - Look for answers in both message content and previous_queries
   - If a match is found, return the answer with "FOUND: " prefix
4. When checking chat history:
   - Compare the current query with previous queries in each message which is stored as text in role as "user"
   - If a match is found, use the corresponding result which is stored in role as "assistant"
   - Return: "FOUND: {result}" and stop the workflow
5. If no match is found:
   - Analyze the chat history and context to understand the query better
   - If the query contains pronouns (e.g., "he," "she," "it," "they"), replace them with correct names/entities based on context
   - If the query is unclear or needs more context, rephrase it using information from previous queries and chat history
   - Respond with: "No, {rephrased_user_query}"
6. When rephrasing queries:
   - Use information from previous queries in the chat history
   - Consider the context from current state
   - Make the query more specific and clear
   - Preserve the original intent while making it more precise
7. Do not guess or infer beyond what is explicitly stated in the chat history.
8. IMPORTANT: When handing off to QueryCraftAgent, you MUST use the handoff tool.
""",
    llm=azure_openai_llm,
tools=[fetch_context],
    verbose=False,
    chat_history=private_chat_history,
    can_handoff_to=["QueryCraftAgent"],
)

query_agent = FunctionAgent(
    name="QueryCraftAgent",
    description="Constructs optimized SQL queries by systematically exploring database structure, generating valid queries based on user intent, and validating execution plans to ensure efficient data retrieval with proper constraints",
system_prompt="""You are a SQL AI assistant specialized in generating SQL queries. 
**Always** generate SQL queries by following these systematic steps: 
1. **List All Tables:**  
    - Retrieve a complete list of all available tables without filtering any out.  

2. **Describe Tables:**  
    - Get detailed schema information for the tables to understand their structure, including column names, data types, and constraints.  

3. **Generate the SQL Query:**  
    - Formulate a clear and efficient SQL query using the available table information. Ensure it returns meaningful results.  
    - Limit the results to a **maximum of 100 rows** to prevent excessive data retrieval.  

4. **Validate the SQL Query:**  
    - Always check the execution plan using the `validate_sql_query()` function before suggesting the query. This ensures the query is efficient and accurate.  

5. **Output Requirements:**  
    - Your final output should be a **valid and optimized SQL query** ready for execution.  
    - Do **not** generate any unnecessary tool calls if the query can be directly inferred.  
    - Provide a brief, clear explanation of the query if necessary.  

**Additional Guidelines:**  
- Do **not** assume table relationships unless explicitly stated.  
- Prioritize clarity and accuracy in both SQL generation and explanations.  
- Avoid redundant operations and reduce unnecessary complexity.
- once the query is generated, you must handoff to the ResultPresenterAgent to present the results""",
    llm=azure_openai_llm,
    tools=[validate_sql_query,describe_tables,list_tables],
    chat_history=public_chat_history,
    verbose=False,
    can_handoff_to=["ResultPresenterAgent"],
)

execute_agent = FunctionAgent(
    name="ResultPresenterAgent",
description="Executes and Translates technical SQL results into natural, conversational responses for users, handling edge cases gracefully and providing appropriate feedback when information cannot be retrieved",
system_prompt="""
You are a SQL AI assistant specialized in executing and translating technical SQL results into natural, conversational responses for users, handling edge cases gracefully and providing appropriate feedback when information cannot be retrieved
- Respond like a chatbot with simple sentences. DO NOT mention the purpose of the SQL query.
- Use the provided SQL query to load the data as is without escaping special characters.
- If no SQL query is provided, display the same message.
- DO NOT reply with one word answers.
    - For unclear answers please call the feedback function with exact details and respond to the user with 'Unable to find the requested information. Please contact support'.
    - IMPORTANT: You MUST call update_context after executing the query to store the result.
    - The session_id is available in the current state, make sure to use it.
    """,
    llm=azure_openai_llm,
tools=[load_data, feedback, update_context],
    chat_history=public_chat_history,
    verbose=False,
)
    
agent_workflow = AgentWorkflow(
    agents=[research_agent, query_agent, execute_agent],
    root_agent=research_agent.name,
    initial_state=initial_state,
)

async def main():
    # Initialize database first
    db_metadata = await initialize_database(uri=DB_CONNECTION_URL)
    if db_metadata["initialization_status"]["status"] == "failed":
        print("Database initialization failed:", db_metadata["initialization_status"]["errors"])
        return

    main_prompt = """Process user queries through a three-agent workflow in the following order: MemoryAgent, QueryCraftAgent, ResultPresenterAgent. Follow these steps exactly:

1. **MemoryAgent**:
   - Use fetch_context to get current context and chat history
   - Check if the query can be answered based on prior interactions
   - If an answer exists:
     - Return: "FOUND: {conversational_answer}" and stop the workflow
   - If no answer exists:
     - Try to rephrase the query using context
     - Use handoff tool to pass to QueryCraftAgent

2. **QueryCraftAgent**:
   - Generate a valid SQL query to answer the query
   - Use list_tables and describe_tables to understand schema
   - Validate query using validate_sql_query
   - If valid, hand off to ResultPresenterAgent

3. **ResultPresenterAgent**:
   - Execute the SQL query using load_data
   - Translate results into conversational response
   - Use update_context to store the result
   - Return response to user

**Rules**:
- Each agent must follow its system prompt
- Keep responses simple and conversational
- Handle errors gracefully
- Limit results to 100 rows
"""

    # Print welcome message only once
    print("\nWelcome to the SQL Assistant! Type 'quit' to exit.")
    print("You can ask questions about the database, and I'll help you find the answers.")
    print("Example questions:")
    print("- What badge did most users earn?")
    print("- How many users have earned badges?")
    print("- What is the average rating of posts?")
    print("\nWhat would you like to know?")

    current_state = initial_state.copy()
    while True:
        try:
            # Get user input
            user_question = input("\nYour question: ").strip()
            if user_question.lower() == 'quit':
                print("\nGoodbye! Have a great day!")
                break

            # Update the initial state with the new query
            current_state["user_query"] = user_question
            
            # Run the workflow with the updated state
            handler = agent_workflow.run(user_msg=f"{main_prompt}\n\nUser Query: {user_question}")
            
            current_agent = None
            final_result = None
            should_continue = True
            
            async for event in handler.stream_events():
                if (
                    hasattr(event, "current_agent_name")
                    and event.current_agent_name != current_agent
                ):
                    current_agent = event.current_agent_name
                    print(f"\n{'='*50}")
                    print(f"🤖 Agent: {current_agent}")
                    print(f"{'='*50}\n")
                elif isinstance(event, AgentOutput):
                    if event.response.content:
                        # Check if this is a final answer from MemoryAgent
                        if current_agent == "MemoryAgent":
                            if "FOUND:" in event.response.content:
                                final_result = event.response.content.split("FOUND:")[1].strip()
                                print(f"\nAnswer: {final_result}")
                                should_continue = False
                            elif "No answer found for query" in event.response.content:
                                print(f"MemoryAgent: {event.response.content}")
                                should_continue = True
                            else:
                                print("📤 Output:", event.response.content)
                        # Check if this is a final answer from ResultPresenterAgent
                        elif current_agent == "ResultPresenterAgent":
                            final_result = event.response.content
                            print(f"\nAnswer: {final_result}")
                            should_continue = False
                        else:
                            print("📤 Output:", event.response.content)
                                        
                    if event.tool_calls:
                        print(
                            "🛠️  Planning to use tools:",
                            [call.tool_name for call in event.tool_calls],
                        )
                elif isinstance(event, ToolCallResult):
                    print(f"🔧 Tool Result ({event.tool_name}):")
                    print(f"  Arguments: {event.tool_kwargs}")
                    print(f"  Output: {event.tool_output}")
                elif isinstance(event, ToolCall):
                    print(f"🔨 Calling Tool: {event.tool_name}")
                    print(f"  With arguments: {event.tool_kwargs}")
            
                # Only break if we have a final result and shouldn't continue
                if final_result and not should_continue:
                    break
            
            print("\nWhat else would you like to know?")
                
        except Exception as e:
            print(f"Error during workflow execution: {str(e)}")
            continue

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
