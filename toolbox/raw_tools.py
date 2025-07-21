from llama_index.core.tools.tool_spec.base import BaseToolSpec
from typing import Any, List, Optional
from sqlalchemy import create_engine, text, Table, MetaData, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.schema import CreateTable

class RawDatabaseToolSpec(BaseToolSpec):
    """Simple Database tool.

    Concatenates each row into Document used by LlamaIndex.

    Args:
        uri (Optional[str]): uri of the database connection.

        OR

        scheme (Optional[str]): scheme of the database connection.
        host (Optional[str]): host of the database connection.
        port (Optional[int]): port of the database connection.
        user (Optional[str]): user of the database connection.
        password (Optional[str]): password of the database connection.
        dbname (Optional[str]): dbname of the database connection.

    """
    def __init__(
        self,
        uri: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dbname: Optional[str] = None,
        tables: Optional[list[str]] = None,
        *args: Optional[Any],
        **kwargs: Optional[Any],
    ) -> None:
        """Initialize with parameters."""
        self.tables = list()
        if uri:
            self.uri = uri
            self.sql_database = create_engine(uri, *args, **kwargs)
        elif scheme and host and port and user and password and dbname:
            uri = f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"
            self.uri = uri
            self.sql_database = create_engine(uri, *args, **kwargs)
        else:
            raise ValueError(
                "You must provide either a SQLDatabase, "
                "a SQL Alchemy Engine, a valid connection URI, or a valid "
                "set of credentials."
            )
        self.inspector = inspect(self.sql_database.engine)
        tables = tables or self.inspector.get_table_names() + self.inspector.get_view_names()
        for table in tables:
            metadata = MetaData()
            try:
                self.tables.append(Table(table, metadata, autoload_with=self.sql_database))
            except NoSuchTableError as e:
                print(f'"{e}" table not found!')
                pass
        self.spec_functions = ["load_data", "describe_tables", "list_tables", 'validate_sql_query', 'feedback']
        self.session = sessionmaker(bind=self.sql_database.engine)
    
    
    def list_tables(self) -> List[str]:
        """
        Returns a list of available tables in the database.
        To retrieve details about the columns of specific tables, use
        the describe_tables endpoint.
        """
        table_list = list()
        for table in self.tables:
            table_info = self.inspector.get_table_comment(table_name=table.name).get('text', '')
            table_info = f'Table Description: {table_info}\n' if table_info else ''
            table_list.append(f'Table Name: {table.name}\n{table_info}\n')
        return ''.join(table_list)
    
    def describe_tables(self, tables: Optional[List[str]] = None) -> str:
        """
        Describes the specified tables in the database.

        Args:
            tables (List[str]): A list of multiple table names to retrieve details about
        """
        table_names = tables or [table.name for table in self.tables]
        table_schemas = []

        for table_name in table_names:
            table = next(
                (
                    table
                    for table in self.tables
                    if table.name == table_name
                ),
                None,
            )
            if table is None:
                raise NoSuchTableError(f"Table '{table_name}' does not exist.")
            schema = str(CreateTable(table).compile(self.sql_database.engine))
            
            columns = self.inspector.get_columns(table_name=table_name)
            column_info = '\n'
            for column in columns:
                comment = column.get("comment", "")
                if comment:
                    column_info += f'{comment}\n'
            table_schemas.append(f"{schema}{column_info}\n")

        return "\n".join(table_schemas)
    
    def load_data(self, query: str) -> List[str]:
        """Query and load data from the Database, returning a list of Documents.

        Args:
            query (str): an SQL query to filter tables and rows.

        Returns:
            List[Document]: A list of Document objects.
        """
        documents = []
        with self.session() as session:
            if query is None:
                raise ValueError("A query parameter is necessary to filter the data")
            else:
                result = session.execute(text(query))

            for item in result.fetchall():
                # fetch each item
                doc_str = ", ".join([str(entry) for entry in item])
                documents.append(doc_str)
        return documents
    
    def validate_sql_query(self, query: str) -> dict:
        """
        Validate a SQL query, without executing it.
        
        Args:
            query (str): The SQL query to validate
            
        Returns:
            dict: A dictionary containing validation results with:
                - valid (bool): Whether the query is valid
                - reason (str): Description of error if invalid
                - plan (str): The execution plan if valid
        """
        if self.uri:
            try:
                # PostgreSQL
                if 'postgresql' in self.uri:
                    explain_query = f"EXPLAIN {query}"
                    results = self.load_data(explain_query)
                    if results:
                        return {"valid": True, "reason": "Query syntax is valid", "plan": results}
                    else:
                        return {"valid": False, "reason": "Invalid query"}

                # MySQL
                elif 'mysql' in self.uri:
                    explain_query = f"EXPLAIN {query}"
                    results = self.load_data(explain_query)
                    if results:
                        return {"valid": True, "reason": "Query syntax is valid", "plan": results}
                    else:
                        return {"valid": False, "reason": "Invalid query"}

                # SQLite
                elif 'sqlite' in self.uri:
                    explain_query = f"EXPLAIN QUERY PLAN {query}"
                    results = self.load_data(explain_query)
                    if results:
                        return {"valid": True, "reason": "Query syntax is valid", "plan": results}
                    else:
                        return {"valid": False, "reason": "Invalid query"}

                # MS SQL Server
                elif 'sqlserver' in self.uri or 'mssql' in self.uri:
                    explain_query = f"SET SHOWPLAN_ALL ON; {query} SET SHOWPLAN_ALL OFF;"
                    results = self.load_data(explain_query)
                    if results:
                        return {"valid": True, "reason": "Query syntax is valid", "plan": results}
                    else:
                        return {"valid": False, "reason": "Invalid query"}

                # Oracle
                elif 'oracle' in self.uri:
                    explain_query = f"EXPLAIN PLAN FOR {query}"
                    self.load_data(explain_query)  # Execute EXPLAIN PLAN FOR to store the plan in the PLAN_TABLE
                    plan_query = "SELECT * FROM TABLE(DBMS_XPLAN.DISPLAY)"
                    results = self.load_data(plan_query)
                    if results:
                        return {"valid": True, "reason": "Query syntax is valid", "plan": results}
                    else:
                        return {"valid": False, "reason": "Invalid query"}

                else:
                    return {"valid": False, "reason": "Query validation not supported for this database type"}
            
            except Exception as e:
                error_message = str(e)
                print(f"Error validating query: {error_message}")
                return {"valid": False, "reason": f"Error validating query: {error_message}"}
        else:
            return {"valid": False, "reason": "Database URI not provided"}
        
    def feedback(self, details: str):
        feedback_str= f"FEEDBACK: {details} Unable to find the requested information. Please contact support."
        return feedback_str
