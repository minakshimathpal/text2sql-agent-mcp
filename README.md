# Text-to-SQL Agent with Multi-Agent Workflow

A sophisticated text-to-SQL conversion system that uses a three-agent workflow to process natural language queries and convert them into executable SQL statements. The system includes memory capabilities to remember previous conversations and provide consistent responses.

## 🚀 Features

- **Multi-Agent Architecture**: Three specialized agents working together
- **Memory System**: Remembers previous conversations and queries
- **Database Agnostic**: Supports PostgreSQL, MySQL, SQLite, MS SQL Server, and Oracle
- **Query Validation**: Validates SQL queries before execution
- **Conversational Interface**: Natural language responses
- **Session Management**: Maintains conversation context across sessions
- **Error Handling**: Graceful handling of database errors and edge cases

## 🏗️ Architecture

### Agent Workflow

1. **MemoryAgent**: 
   - Checks if the query can be answered from previous conversations
   - Provides consistent responses to previously addressed queries
   - Rephrases unclear queries using context
   - Routes new questions to specialized agents

2. **QueryCraftAgent**:
   - Explores database structure systematically
   - Generates optimized SQL queries based on user intent
   - Validates execution plans for efficiency
   - Ensures proper constraints and data retrieval

3. **ResultPresenterAgent**:
   - Executes validated SQL queries
   - Translates technical results into conversational responses
   - Handles edge cases gracefully
   - Stores results in conversation history

## 📋 Prerequisites

- Python 3.8+
- Database connection (PostgreSQL, MySQL, SQLite, MS SQL Server, or Oracle)
- Azure OpenAI API access (or Ollama for local models)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd text2sql-agent-mcp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Database Configuration
   DB_CONNECTION_URL=your_database_connection_string
   
   # Azure OpenAI Configuration
   AZURE_OPENAI_MODEL_NAME=llama3.1:8b
   AZURE_OPENAI_ENGINE_NAME=gpt-4o-mini
   
   # Ollama Configuration (for local models)
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL_NAME=llama3.1:8b
   ```

## 🚀 Usage

### Running the Application

```bash
python Agent/trial1.py
```

### Example Queries

Once the application is running, you can ask questions like:

- "What badge did most users earn?"
- "How many users have earned badges?"
- "What is the average rating of posts?"
- "Show me all users who joined in the last month"
- "Which posts have the highest engagement?"

### Interactive Session

The application provides an interactive command-line interface:

```
Welcome to the SQL Assistant! Type 'quit' to exit.
You can ask questions about the database, and I'll help you find the answers.

Your question: What badge did most users earn?

==================================================
🤖 Agent: MemoryAgent
==================================================

==================================================
🤖 Agent: QueryCraftAgent
==================================================

==================================================
🤖 Agent: ResultPresenterAgent
==================================================

Answer: Based on the database results, the most commonly earned badge is...

What else would you like to know?
```

## 🔧 Configuration

### Database Connection

The system supports various database types through SQLAlchemy:

- **PostgreSQL**: `postgresql://user:password@host:port/dbname`
- **MySQL**: `mysql://user:password@host:port/dbname`
- **SQLite**: `sqlite:///path/to/database.db`
- **MS SQL Server**: `mssql://user:password@host:port/dbname`
- **Oracle**: `oracle://user:password@host:port/dbname`

### Model Configuration

You can configure different LLM providers:

- **Azure OpenAI**: Set `AZURE_OPENAI_MODEL_NAME` and `AZURE_OPENAI_ENGINE_NAME`
- **Ollama (Local)**: Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL_NAME`

## 📁 Project Structure

```
text2sql-agent-mcp/
├── Agent/
│   └── trial1.py          # Main application file
├── chat_store_public.json # Public conversation history
├── chat_store_private.json # Private conversation history
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔍 Key Components

### Database Functions

- `initialize_database()`: Sets up database connection and metadata
- `list_tables()`: Retrieves available tables
- `describe_tables()`: Gets detailed table schemas
- `load_data()`: Executes SQL queries and returns results
- `validate_sql_query()`: Validates SQL syntax and execution plans

### Context Management

- `fetch_context()`: Retrieves conversation history and current state
- `update_context()`: Stores new queries and results
- `get_chat_history()`: Manages session persistence

### Agent Tools

Each agent has access to specific tools:

- **MemoryAgent**: `fetch_context`
- **QueryCraftAgent**: `validate_sql_query`, `describe_tables`, `list_tables`
- **ResultPresenterAgent**: `load_data`, `feedback`, `update_context`

## 🛡️ Error Handling

The system includes comprehensive error handling:

- Database connection failures
- Invalid SQL queries
- Missing tables or columns
- Network timeouts
- Model API errors

## 🔄 Session Management

- **Session IDs**: Unique identifiers for each conversation
- **Chat History**: Persistent storage of conversations
- **State Management**: Maintains context across agent interactions
- **Memory Persistence**: Stores conversations in JSON files

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   - Verify `DB_CONNECTION_URL` in `.env`
   - Check database server status
   - Ensure proper credentials

2. **Model API Errors**:
   - Verify Azure OpenAI credentials
   - Check model name and engine configuration
   - Ensure API quota is available

3. **File Permission Errors**:
   - Ensure write permissions for chat store directory
   - Check file paths in configuration

### Debug Mode

Enable verbose logging by setting `verbose=True` in agent configurations.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LlamaIndex for agent orchestration
- Uses SQLAlchemy for database operations
- Powered by Azure OpenAI for natural language processing
- Inspired by modern conversational AI systems

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the error logs for specific details

---

**Note**: This system is designed for educational and research purposes. Always ensure proper security measures when connecting to production databases. 