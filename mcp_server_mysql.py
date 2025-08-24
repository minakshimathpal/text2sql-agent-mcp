import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from sqlalchemy import Engine
import argparse
from operator import attrgetter

import os
import subprocess
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.fastmcp import FastMCP


from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
import smtplib
from email.mime.text import MIMEText
import uvicorn

from Agent.database_module.database_manager import DatabaseManager
db = DatabaseManager()
engine = db.engine
# instantiate an MCP server client
mcp = FastMCP("SQL Tools")

# ===================== BASIC SELECT OPERATIONS =====================

@mcp.tool()
def get_table_schemas(db_name):
    """Get all the schemas of all the tables listed in a database"""
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

@mcp.tool()
def select_all(table_name: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Select all columns from a table.
    
    Args:
        table_name (str): Name of the database table
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: All rows and columns from the table
        
    Example:
        >>> select_all('users')
        # Returns entire users table
    """
    if df is not None:
        return df.copy()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


@mcp.tool()
def select_columns(table_name: str, columns: List[str], 
                  df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Select specific columns from a table.
    
    Args:
        table_name (str): Name of the database table
        columns (List[str]): List of column names to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Selected columns from the table
        
    Example:
        >>> select_columns('users', ['name', 'email'])
        # Returns only name and email columns
    """
    if df is not None:
        return df[columns].copy()
    cols_str = ', '.join(columns)
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name}", engine)


@mcp.tool()
def select_distinct(table_name: str, columns: List[str], 
                   df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Select distinct values from specified columns.
    
    Args:
        table_name (str): Name of the database table
        columns (List[str]): List of column names for distinct selection
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Unique combinations of values from specified columns
        
    Example:
        >>> select_distinct('orders', ['customer_id', 'status'])
        # Returns unique customer_id/status combinations
    """
    if df is not None:
        return df[columns].drop_duplicates().reset_index(drop=True)
    cols_str = ', '.join(columns)
    return pd.read_sql(f"SELECT DISTINCT {cols_str} FROM {table_name}", engine)


@mcp.tool()
def select_top(table_name: str, n: int, columns: List[str] = None,
              df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Select top N rows from a table.
    
    Args:
        table_name (str): Name of the database table
        n (int): Number of rows to return
        columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: First N rows from the table
        
    Example:
        >>> select_top('products', 10, ['name', 'price'])
        # Returns first 10 rows with name and price columns
    """
    if df is not None:
        result = df.head(n)
        return result[columns] if columns else result
    
    cols_str = ', '.join(columns) if columns else '*'
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} LIMIT {n}", engine)


@mcp.tool()
def where_condition(table_name: str, condition_column: str, 
                   operator: str, value: Any, columns: List[str] = None,
                   df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filter rows based on a condition.
    
    Args:
        table_name (str): Name of the database table
        condition_column (str): Column to apply condition on
        operator (str): Comparison operator (=, !=, >, <, >=, <=, LIKE)
        value (Any): Value to compare against
        columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Filtered rows matching the condition
        
    Example:
        >>> where_condition('users', 'age', '>', 18, ['name', 'email'])
        # Returns users older than 18 with name and email columns
    """
    if df is not None:
        if operator == '=':
            mask = df[condition_column] == value
        elif operator == '!=':
            mask = df[condition_column] != value
        elif operator == '>':
            mask = df[condition_column] > value
        elif operator == '<':
            mask = df[condition_column] < value
        elif operator == '>=':
            mask = df[condition_column] >= value
        elif operator == '<=':
            mask = df[condition_column] <= value
        elif operator.upper() == 'LIKE':
            mask = df[condition_column].str.contains(str(value).replace('%', ''), na=False)
        else:
            raise ValueError(f"Operator {operator} not supported")
        
        result = df[mask]
        return result[columns] if columns else result
    
    cols_str = ', '.join(columns) if columns else '*'
    if operator.upper() == 'LIKE':
        condition = f"{condition_column} LIKE '{value}'"
    else:
        condition = f"{condition_column} {operator} '{value}'" if isinstance(value, str) else f"{condition_column} {operator} {value}"
    
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} WHERE {condition}", engine)


@mcp.tool()
def where_in(table_name: str, column: str, values: List[Any],
            columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filter rows where column value is in a list of values.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column to check values against
        values (List[Any]): List of values to match
        columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Rows where column value is in the provided list
        
    Example:
        >>> where_in('orders', 'status', ['pending', 'shipped'])
        # Returns orders with pending or shipped status
    """
    if df is not None:
        result = df[df[column].isin(values)]
        return result[columns] if columns else result
    
    cols_str = ', '.join(columns) if columns else '*'
    values_str = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in values])
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} WHERE {column} IN ({values_str})", engine)


@mcp.tool()
def where_between(table_name: str, column: str, start_value: Any, end_value: Any,
                 columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filter rows where column value is between two values (inclusive).
    
    Args:
        table_name (str): Name of the database table
        column (str): Column to check range on
        start_value (Any): Lower bound (inclusive)
        end_value (Any): Upper bound (inclusive)
        columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Rows where column value is within the range
        
    Example:
        >>> where_between('products', 'price', 10, 100)
        # Returns products with price between 10 and 100
    """
    if df is not None:
        result = df[(df[column] >= start_value) & (df[column] <= end_value)]
        return result[columns] if columns else result
    
    cols_str = ', '.join(columns) if columns else '*'
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} WHERE {column} BETWEEN {start_value} AND {end_value}", engine)


@mcp.tool()
def where_null(table_name: str, column: str, is_null: bool = True,
              columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filter rows based on null values in a column.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column to check for null values
        is_null (bool): True for IS NULL, False for IS NOT NULL
        columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Rows with null or non-null values in specified column
        
    Example:
        >>> where_null('users', 'phone_number', False)
        # Returns users with non-null phone numbers
    """
    if df is not None:
        if is_null:
            result = df[df[column].isnull()]
        else:
            result = df[df[column].notnull()]
        return result[columns] if columns else result
    
    cols_str = ', '.join(columns) if columns else '*'
    null_condition = "IS NULL" if is_null else "IS NOT NULL"
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} WHERE {column} {null_condition}", engine)


@mcp.tool()
def order_by(table_name: str, columns: List[str], ascending: List[bool] = None,
            select_columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Sort table rows by specified columns.
    
    Args:
        table_name (str): Name of the database table
        columns (List[str]): Columns to sort by
        ascending (List[bool], optional): Sort direction for each column (True=ASC, False=DESC)
        select_columns (List[str], optional): Specific columns to select
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Sorted rows from the table
        
    Example:
        >>> order_by('users', ['age', 'name'], [False, True])
        # Sorts by age descending, then name ascending
    """
    if df is not None:
        if ascending is None:
            ascending = [True] * len(columns)
        result = df.sort_values(by=columns, ascending=ascending).reset_index(drop=True)
        return result[select_columns] if select_columns else result
    
    cols_str = ', '.join(select_columns) if select_columns else '*'
    if ascending is None:
        ascending = [True] * len(columns)
    
    order_parts = []
    for col, asc in zip(columns, ascending):
        order_parts.append(f"{col} {'ASC' if asc else 'DESC'}")
    order_str = ', '.join(order_parts)
    
    return pd.read_sql(f"SELECT {cols_str} FROM {table_name} ORDER BY {order_str}", engine)


@mcp.tool()
def group_by_count(table_name: str, group_columns: List[str], 
                  count_column: str = '*', df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Group rows and count occurrences.
    
    Args:
        table_name (str): Name of the database table
        group_columns (List[str]): Columns to group by
        count_column (str): Column to count (default: '*' for row count)
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Grouped data with count column
        
    Example:
        >>> group_by_count('orders', ['status'])
        # Returns count of orders by status
    """
    if df is not None:
        if count_column == '*':
            return df.groupby(group_columns).size().reset_index(name='count')
        else:
            return df.groupby(group_columns)[count_column].count().reset_index()
    
    group_str = ', '.join(group_columns)
    count_col = count_column if count_column != '*' else '*'
    return pd.read_sql(f"SELECT {group_str}, COUNT({count_col}) as count FROM {table_name} GROUP BY {group_str}", engine)


@mcp.tool()
def group_by_sum(table_name: str, group_columns: List[str], sum_columns: List[str],
                df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Group rows and calculate sum of specified columns.
    
    Args:
        table_name (str): Name of the database table
        group_columns (List[str]): Columns to group by
        sum_columns (List[str]): Columns to sum
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Grouped data with sum columns
        
    Example:
        >>> group_by_sum('sales', ['region'], ['amount'])
        # Returns total sales amount by region
    """
    if df is not None:
        return df.groupby(group_columns)[sum_columns].sum().reset_index()
    
    group_str = ', '.join(group_columns)
    sum_str = ', '.join([f"SUM({col}) as sum_{col}" for col in sum_columns])
    return pd.read_sql(f"SELECT {group_str}, {sum_str} FROM {table_name} GROUP BY {group_str}", engine)


@mcp.tool()
def group_by_avg(table_name: str, group_columns: List[str], avg_columns: List[str],
                df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Group rows and calculate average of specified columns.
    
    Args:
        table_name (str): Name of the database table
        group_columns (List[str]): Columns to group by
        avg_columns (List[str]): Columns to average
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Grouped data with average columns
        
    Example:
        >>> group_by_avg('students', ['grade'], ['score'])
        # Returns average score by grade
    """
    if df is not None:
        return df.groupby(group_columns)[avg_columns].mean().reset_index()
    
    group_str = ', '.join(group_columns)
    avg_str = ', '.join([f"AVG({col}) as avg_{col}" for col in avg_columns])
    return pd.read_sql(f"SELECT {group_str}, {avg_str} FROM {table_name} GROUP BY {group_str}", engine)


@mcp.tool()
def group_by_multiple_agg(table_name: str, group_columns: List[str], 
                         agg_dict: Dict[str, List[str]], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Group rows and apply multiple aggregation functions.
    
    Args:
        table_name (str): Name of the database table
        group_columns (List[str]): Columns to group by
        agg_dict (Dict[str, List[str]]): Dictionary mapping columns to aggregation functions
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Grouped data with multiple aggregations
        
    Example:
        >>> group_by_multiple_agg('sales', ['region'], {'amount': ['sum', 'avg']})
        # Returns sum and average amount by region
    """
    if df is not None:
        return df.groupby(group_columns).agg(agg_dict).reset_index()
    
    group_str = ', '.join(group_columns)
    agg_parts = []
    for col, funcs in agg_dict.items():
        for func in funcs:
            agg_parts.append(f"{func.upper()}({col}) as {func}_{col}")
    agg_str = ', '.join(agg_parts)
    
    return pd.read_sql(f"SELECT {group_str}, {agg_str} FROM {table_name} GROUP BY {group_str}", engine)


@mcp.tool()
def having_condition(table_name: str, group_columns: List[str], 
                    agg_column: str, agg_function: str, operator: str, value: Any,
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filter grouped results based on aggregate conditions.
    
    Args:
        table_name (str): Name of the database table
        group_columns (List[str]): Columns to group by
        agg_column (str): Column to aggregate
        agg_function (str): Aggregation function (sum, avg, count, etc.)
        operator (str): Comparison operator (>, <, >=, <=, =, !=)
        value (Any): Value to compare against
        df (pd.DataFrame, optional): DataFrame to operate on instead of database
    
    Returns:
        pd.DataFrame: Grouped data filtered by aggregate condition
        
    Example:
        >>> having_condition('orders', ['customer_id'], 'amount', 'sum', '>', 1000)
        # Returns customers with total orders > 1000
    """
    if df is not None:
        grouped = df.groupby(group_columns).agg({agg_column: agg_function}).reset_index()
        
        if operator == '>':
            return grouped[grouped[agg_column] > value]
        elif operator == '<':
            return grouped[grouped[agg_column] < value]
        elif operator == '>=':
            return grouped[grouped[agg_column] >= value]
        elif operator == '<=':
            return grouped[grouped[agg_column] <= value]
        elif operator == '=':
            return grouped[grouped[agg_column] == value]
        elif operator == '!=':
            return grouped[grouped[agg_column] != value]
    
    group_str = ', '.join(group_columns)
    agg_func_upper = agg_function.upper()
    return pd.read_sql(f"""
        SELECT {group_str}, {agg_func_upper}({agg_column}) as {agg_function}_{agg_column}
        FROM {table_name} 
        GROUP BY {group_str} 
        HAVING {agg_func_upper}({agg_column}) {operator} {value}
    """, engine)


@mcp.tool()
def inner_join(left_table: str, right_table: str, join_column: str,
              left_columns: List[str] = None, right_columns: List[str] = None,
              left_df: Optional[pd.DataFrame] = None, right_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform inner join between two tables.
    
    Args:
        left_table (str): Name of the left table
        right_table (str): Name of the right table
        join_column (str): Column to join on
        left_columns (List[str], optional): Columns to select from left table
        right_columns (List[str], optional): Columns to select from right table
        left_df (pd.DataFrame, optional): Left DataFrame to operate on
        right_df (pd.DataFrame, optional): Right DataFrame to operate on
    
    Returns:
        pd.DataFrame: Inner joined result
        
    Example:
        >>> inner_join('users', 'orders', 'user_id')
        # Returns users with their orders
    """
    if left_df is not None and right_df is not None:
        result = pd.merge(left_df, right_df, on=join_column, how='inner')
        if left_columns and right_columns:
            return result[left_columns + right_columns]
        return result
    
    left_cols = ', '.join([f"l.{col}" for col in left_columns]) if left_columns else 'l.*'
    right_cols = ', '.join([f"r.{col}" for col in right_columns]) if right_columns else 'r.*'
    
    return pd.read_sql(f"""
        SELECT {left_cols}, {right_cols}
        FROM {left_table} l
        INNER JOIN {right_table} r ON l.{join_column} = r.{join_column}
    """, engine)


@mcp.tool()
def left_join(left_table: str, right_table: str, join_column: str,
             left_columns: List[str] = None, right_columns: List[str] = None,
             left_df: Optional[pd.DataFrame] = None, right_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform left join between two tables.
    
    Args:
        left_table (str): Name of the left table
        right_table (str): Name of the right table
        join_column (str): Column to join on
        left_columns (List[str], optional): Columns to select from left table
        right_columns (List[str], optional): Columns to select from right table
        left_df (pd.DataFrame, optional): Left DataFrame to operate on
        right_df (pd.DataFrame, optional): Right DataFrame to operate on
    
    Returns:
        pd.DataFrame: Left joined result (all left table rows, matched right rows)
        
    Example:
        >>> left_join('users', 'profiles', 'user_id')
        # Returns all users with their profiles (if they exist)
    """
    if left_df is not None and right_df is not None:
        result = pd.merge(left_df, right_df, on=join_column, how='left')
        if left_columns and right_columns:
            return result[left_columns + right_columns]
        return result
    
    left_cols = ', '.join([f"l.{col}" for col in left_columns]) if left_columns else 'l.*'
    right_cols = ', '.join([f"r.{col}" for col in right_columns]) if right_columns else 'r.*'
    
    return pd.read_sql(f"""
        SELECT {left_cols}, {right_cols}
        FROM {left_table} l
        LEFT JOIN {right_table} r ON l.{join_column} = r.{join_column}
    """, engine)


@mcp.tool()
def right_join(left_table: str, right_table: str, join_column: str,
              left_columns: List[str] = None, right_columns: List[str] = None,
              left_df: Optional[pd.DataFrame] = None, right_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform right join between two tables.
    
    Args:
        left_table (str): Name of the left table
        right_table (str): Name of the right table
        join_column (str): Column to join on
        left_columns (List[str], optional): Columns to select from left table
        right_columns (List[str], optional): Columns to select from right table
        left_df (pd.DataFrame, optional): Left DataFrame to operate on
        right_df (pd.DataFrame, optional): Right DataFrame to operate on
    
    Returns:
        pd.DataFrame: Right joined result (all right table rows, matched left rows)
        
    Example:
        >>> right_join('orders', 'products', 'product_id')
        # Returns all products with their orders (if they exist)
    """
    if left_df is not None and right_df is not None:
        result = pd.merge(left_df, right_df, on=join_column, how='right')
        if left_columns and right_columns:
            return result[left_columns + right_columns]
        return result
    
    left_cols = ', '.join([f"l.{col}" for col in left_columns]) if left_columns else 'l.*'
    right_cols = ', '.join([f"r.{col}" for col in right_columns]) if right_columns else 'r.*'
    
    return pd.read_sql(f"""
        SELECT {left_cols}, {right_cols}
        FROM {left_table} l
        RIGHT JOIN {right_table} r ON l.{join_column} = r.{join_column}
    """, engine)

# ===================== SUBQUERIES =====================
@mcp.tool()
def full_outer_join(left_table: str, right_table: str, join_column: str,
                    left_columns: List[str] = None, right_columns: List[str] = None,
                    left_df: Optional[pd.DataFrame] = None, right_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Performs a FULL OUTER JOIN operation between two tables/DataFrames.
    
    Args:
        left_table (str): Name of the left table
        right_table (str): Name of the right table  
        join_column (str): Column name to join on
        left_columns (List[str], optional): Specific columns from left table
        right_columns (List[str], optional): Specific columns from right table
        left_df (pd.DataFrame, optional): Left DataFrame for in-memory operation
        right_df (pd.DataFrame, optional): Right DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Result of full outer join with all matching and non-matching rows
    
    Example:
        >>> full_outer_join('customers', 'orders', 'customer_id', ['name'], ['order_date'])
    """
    if left_df is not None and right_df is not None:
        result = pd.merge(left_df, right_df, on=join_column, how='outer')
        if left_columns and right_columns:
            return result[left_columns + right_columns]
        return result
    
    left_cols = ', '.join([f"l.{col}" for col in left_columns]) if left_columns else 'l.*'
    right_cols = ', '.join([f"r.{col}" for col in right_columns]) if right_columns else 'r.*'
    
    return pd.read_sql(f"""
        SELECT {left_cols}, {right_cols}
        FROM {left_table} l
        FULL OUTER JOIN {right_table} r ON l.{join_column} = r.{join_column}
    """, engine)

# ===================== SUBQUERIES =====================

@mcp.tool()
def exists_subquery(main_table: str, sub_table: str, main_column: str, sub_column: str,
                    main_df: Optional[pd.DataFrame] = None, sub_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filters main table for rows where a related record exists in sub table.
    
    Args:
        main_table (str): Name of the main table
        sub_table (str): Name of the sub table to check existence
        main_column (str): Column in main table for comparison
        sub_column (str): Column in sub table for comparison
        main_df (pd.DataFrame, optional): Main DataFrame for in-memory operation
        sub_df (pd.DataFrame, optional): Sub DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Filtered rows where EXISTS condition is true
    
    Example:
        >>> exists_subquery('customers', 'orders', 'customer_id', 'customer_id')
    """
    if main_df is not None and sub_df is not None:
        return main_df[main_df[main_column].isin(sub_df[sub_column])]
    
    return pd.read_sql(f"""
        SELECT * FROM {main_table} m
        WHERE EXISTS (SELECT 1 FROM {sub_table} s WHERE s.{sub_column} = m.{main_column})
    """, engine)

@mcp.tool()
def in_subquery(main_table: str, sub_table: str, main_column: str, sub_column: str,
                main_df: Optional[pd.DataFrame] = None, sub_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Filters main table where column values are IN a list from sub table.
    
    Args:
        main_table (str): Name of the main table
        sub_table (str): Name of the sub table containing filter values
        main_column (str): Column in main table to filter
        sub_column (str): Column in sub table containing filter values
        main_df (pd.DataFrame, optional): Main DataFrame for in-memory operation
        sub_df (pd.DataFrame, optional): Sub DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Filtered rows where column value exists in subquery results
    
    Example:
        >>> in_subquery('products', 'bestsellers', 'product_id', 'product_id')
    """
    if main_df is not None and sub_df is not None:
        return main_df[main_df[main_column].isin(sub_df[sub_column])]
    
    return pd.read_sql(f"""
        SELECT * FROM {main_table}
        WHERE {main_column} IN (SELECT {sub_column} FROM {sub_table})
    """, engine)

# ===================== UNION OPERATIONS =====================

@mcp.tool()
def union_all(table1: str, table2: str, columns: List[str],
                df1: Optional[pd.DataFrame] = None, df2: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Combines rows from two tables, including duplicates.
    
    Args:
        table1 (str): Name of the first table
        table2 (str): Name of the second table
        columns (List[str]): Columns to select from both tables
        df1 (pd.DataFrame, optional): First DataFrame for in-memory operation
        df2 (pd.DataFrame, optional): Second DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Combined rows from both tables with duplicates preserved
    
    Example:
        >>> union_all('sales_2023', 'sales_2024', ['product', 'amount', 'date'])
    """
    if df1 is not None and df2 is not None:
        return pd.concat([df1[columns], df2[columns]], ignore_index=True)
    
    cols_str = ', '.join(columns)
    return pd.read_sql(f"""
        SELECT {cols_str} FROM {table1}
        UNION ALL
        SELECT {cols_str} FROM {table2}
    """, engine)

@mcp.tool()
def union_distinct(table1: str, table2: str, columns: List[str],
                    df1: Optional[pd.DataFrame] = None, df2: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Combines rows from two tables, removing duplicates.
    
    Args:
        table1 (str): Name of the first table
        table2 (str): Name of the second table
        columns (List[str]): Columns to select from both tables
        df1 (pd.DataFrame, optional): First DataFrame for in-memory operation
        df2 (pd.DataFrame, optional): Second DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Combined rows from both tables with duplicates removed
    
    Example:
        >>> union_distinct('active_users', 'premium_users', ['user_id', 'email'])
    """
    if df1 is not None and df2 is not None:
        return pd.concat([df1[columns], df2[columns]], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    cols_str = ', '.join(columns)
    return pd.read_sql(f"""
        SELECT {cols_str} FROM {table1}
        UNION
        SELECT {cols_str} FROM {table2}
    """, engine)

# ===================== WINDOW FUNCTIONS =====================

@mcp.tool()
def row_number(table_name: str, partition_by: List[str], order_by: List[str],
                df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Assigns sequential row numbers within partitions.
    
    Args:
        table_name (str): Name of the table
        partition_by (List[str]): Columns to partition by (empty for no partitioning)
        order_by (List[str]): Columns to order by
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with row_number column added
    
    Example:
        >>> row_number('sales', ['department'], ['sales_amount'])
    """
    if df is not None:
        return df.assign(row_number=df.groupby(partition_by).cumcount() + 1)
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by)
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, ROW_NUMBER() OVER ({partition_clause} ORDER BY {order_str}) as row_number
        FROM {table_name}
    """, engine)

@mcp.tool()
def rank_function(table_name: str, partition_by: List[str], order_by: List[str],
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Assigns ranks with gaps for tied values within partitions.
    
    Args:
        table_name (str): Name of the table
        partition_by (List[str]): Columns to partition by
        order_by (List[str]): Columns to order by for ranking
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with rank column added (1, 2, 2, 4...)
    
    Example:
        >>> rank_function('employees', ['department'], ['salary'])
    """
    if df is not None:
        return df.assign(rank=df.groupby(partition_by)[order_by[0]].rank(method='min'))
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by)
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, RANK() OVER ({partition_clause} ORDER BY {order_str}) as rank
        FROM {table_name}
    """, engine)

@mcp.tool()
def dense_rank(table_name: str, partition_by: List[str], order_by: List[str],
                df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Assigns consecutive ranks without gaps for tied values within partitions.
    
    Args:
        table_name (str): Name of the table
        partition_by (List[str]): Columns to partition by
        order_by (List[str]): Columns to order by for ranking
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with dense_rank column added (1, 2, 2, 3...)
    
    Example:
        >>> dense_rank('students', ['grade'], ['score'])
    """
    if df is not None:
        return df.assign(dense_rank=df.groupby(partition_by)[order_by[0]].rank(method='dense'))
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by)
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, DENSE_RANK() OVER ({partition_clause} ORDER BY {order_str}) as dense_rank
        FROM {table_name}
    """, engine)

@mcp.tool()
def lag_lead(table_name: str, column: str, partition_by: List[str], order_by: List[str],
            offset: int = 1, lag: bool = True, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Accesses values from previous (LAG) or next (LEAD) rows within partitions.
    
    Args:
        table_name (str): Name of the table
        column (str): Column to lag/lead
        partition_by (List[str]): Columns to partition by
        order_by (List[str]): Columns to order by
        offset (int, optional): Number of rows to offset. Defaults to 1
        lag (bool, optional): True for LAG, False for LEAD. Defaults to True
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with lag/lead column added
    
    Example:
        >>> lag_lead('stock_prices', 'price', ['symbol'], ['date'], offset=1, lag=True)
    """
    if df is not None:
        if lag:
            df_result = df.assign(**{f'lag_{column}': df.groupby(partition_by)[column].shift(offset)})
        else:
            df_result = df.assign(**{f'lead_{column}': df.groupby(partition_by)[column].shift(-offset)})
        return df_result
    
    func_name = 'LAG' if lag else 'LEAD'
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by)
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, {func_name}({column}, {offset}) OVER ({partition_clause} ORDER BY {order_str}) as {func_name.lower()}_{column}
        FROM {table_name}
    """, engine)

# ===================== ANALYTICAL FUNCTIONS =====================

@mcp.tool()
def running_total(table_name: str, sum_column: str, partition_by: List[str], order_by: List[str],
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculates cumulative sum within partitions.
    
    Args:
        table_name (str): Name of the table
        sum_column (str): Column to calculate running total for
        partition_by (List[str]): Columns to partition by
        order_by (List[str]): Columns to order by for accumulation
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with running_total column added
    
    Example:
        >>> running_total('daily_sales', 'revenue', ['store_id'], ['sale_date'])
    """
    if df is not None:
        return df.assign(running_total=df.groupby(partition_by)[sum_column].cumsum())
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by)
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, SUM({sum_column}) OVER ({partition_clause} ORDER BY {order_str}) as running_total
        FROM {table_name}
    """, engine)

@mcp.tool()
def percentile_rank(table_name: str, column: str, partition_by: List[str] = None,
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculates percentile rank of values within partitions.
    
    Args:
        table_name (str): Name of the table
        column (str): Column to calculate percentile rank for
        partition_by (List[str], optional): Columns to partition by
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with percent_rank column (0.0 to 1.0)
    
    Example:
        >>> percentile_rank('test_scores', 'score', ['subject'])
    """
    if df is not None:
        if partition_by:
            return df.assign(percent_rank=df.groupby(partition_by)[column].rank(pct=True))
        else:
            return df.assign(percent_rank=df[column].rank(pct=True))
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, PERCENT_RANK() OVER ({partition_clause} ORDER BY {column}) as percent_rank
        FROM {table_name}
    """, engine)

@mcp.tool()
def ntile(table_name: str, column: str, n: int, partition_by: List[str] = None,
            df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Divides ordered data into n equal buckets within partitions.
    
    Args:
        table_name (str): Name of the table
        column (str): Column to use for bucketing
        n (int): Number of buckets to create
        partition_by (List[str], optional): Columns to partition by
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with ntile column (1 to n)
    
    Example:
        >>> ntile('customer_sales', 'total_spent', 4, ['region'])  # Quartiles by region
    """
    if df is not None:
        if partition_by:
            return df.assign(ntile=df.groupby(partition_by)[column].apply(lambda x: pd.qcut(x, n, labels=False) + 1))
        else:
            return df.assign(ntile=pd.qcut(df[column], n, labels=False) + 1)
    
    partition_str = ', '.join(partition_by) if partition_by else ''
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, NTILE({n}) OVER ({partition_clause} ORDER BY {column}) as ntile
        FROM {table_name}
    """, engine)

# ===================== PIVOT OPERATIONS =====================

@mcp.tool()
def pivot_table(table_name: str, index_cols: List[str], value_col: str, 
                column_col: str, agg_func: str = 'sum', df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Creates pivot table with aggregated values.
    
    Args:
        table_name (str): Name of the table
        index_cols (List[str]): Columns to use as row indices
        value_col (str): Column containing values to aggregate
        column_col (str): Column whose unique values become new columns
        agg_func (str, optional): Aggregation function ('sum', 'mean', 'count'). Defaults to 'sum'
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Pivoted table with aggregated values
    
    Example:
        >>> pivot_table('sales', ['product'], 'revenue', 'month', 'sum')
    """
    if df is not None:
        return pd.pivot_table(df, 
                            values=value_col, 
                            index=index_cols, 
                            columns=column_col, 
                            aggfunc=agg_func, 
                            fill_value=0).reset_index()
    
    # For SQL pivot, this would be database-specific
    # This is a simplified version - actual SQL pivot syntax varies by database
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return pd.pivot_table(data, 
                        values=value_col, 
                        index=index_cols, 
                        columns=column_col, 
                        aggfunc=agg_func, 
                        fill_value=0).reset_index()

# ===================== CASE STATEMENTS =====================

@mcp.tool()
def case_when(table_name: str, case_column: str, conditions: List[Dict], 
                else_value: Any = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Creates conditional logic column based on multiple conditions.
    
    Args:
        table_name (str): Name of the table
        case_column (str): Name for the new conditional column
        conditions (List[Dict]): List of condition dictionaries with 'condition' and 'value' keys
        else_value (Any, optional): Default value when no conditions match
        df (pd.DataFrame, optional): DataFrame for in-memory operation
    
    Returns:
        pd.DataFrame: Original data with new case_column added
    
    Example:
        >>> conditions = [
        ...     {'condition': 'score >= 90', 'value': 'A'},
        ...     {'condition': 'score >= 80', 'value': 'B'}
        ... ]
        >>> case_when('students', 'grade', conditions, 'F')
    """
    if df is not None:
        result_df = df.copy()
        case_series = pd.Series([else_value] * len(df), index=df.index)
        
        for condition in conditions:
            if '>' in condition['condition']:
                col, val = condition['condition'].split('>')
                col, val = col.strip(), float(val.strip())
                mask = df[col] > val
            elif '<' in condition['condition']:
                col, val = condition['condition'].split('<')
                col, val = col.strip(), float(val.strip())
                mask = df[col] < val
            elif '=' in condition['condition']:
                col, val = condition['condition'].split('=')
                col, val = col.strip(), val.strip().strip("'\"")
                mask = df[col] == val
            else:
                continue
            
            case_series[mask] = condition['value']
        
        result_df[case_column] = case_series
        return result_df
    
    # Build SQL CASE statement
    case_parts = []
    for condition in conditions:
        case_parts.append(f"WHEN {condition['condition']} THEN '{condition['value']}'")
    
    case_statement = f"CASE {' '.join(case_parts)}"
    if else_value:
        case_statement += f" ELSE '{else_value}'"
    case_statement += f" END as {case_column}"
    
    return pd.read_sql(f"SELECT *, {case_statement} FROM {table_name}", engine)

# ===================== STRING FUNCTIONS =====================
@mcp.tool()
def string_concat(table_name: str, columns: List[str], separator: str = '',
                    new_column_name: str = 'concatenated', df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Concatenate multiple string columns into a single column.
    
    Args:
        table_name (str): Name of the database table
        columns (List[str]): List of column names to concatenate
        separator (str): String separator between concatenated values (default: '')
        new_column_name (str): Name for the new concatenated column (default: 'concatenated')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new concatenated column added
    
    Example:
        >>> string_concat('users', ['first_name', 'last_name'], ' ', 'full_name')
        # Creates 'full_name' column: "John Doe"
    """
    if df is not None:
        df_result = df.copy()
        df_result[new_column_name] = df[columns].astype(str).agg(separator.join, axis=1)
        return df_result
    
    concat_expr = f"CONCAT({', '.join(columns)})" if separator == '' else f"CONCAT_WS('{separator}', {', '.join(columns)})"
    return pd.read_sql(f"SELECT *, {concat_expr} as {new_column_name} FROM {table_name}", engine)

@mcp.tool()
def string_length(table_name: str, column: str, new_column_name: str = 'length',
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate the length of strings in a column.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column name to calculate length for
        new_column_name (str): Name for the new length column (default: 'length')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new length column added
    
    Example:
        >>> string_length('products', 'product_name', 'name_length')
        # Creates 'name_length' column with character counts
    """
    if df is not None:
        df_result = df.copy()
        df_result[new_column_name] = df[column].astype(str).str.len()
        return df_result
    
    return pd.read_sql(f"SELECT *, LENGTH({column}) as {new_column_name} FROM {table_name}", engine)

@mcp.tool()
def string_upper_lower(table_name: str, column: str, operation: str = 'upper',
                        new_column_name: str = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert string column to uppercase or lowercase.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column name to convert
        operation (str): 'upper' or 'lower' case conversion (default: 'upper')
        new_column_name (str): Name for the new column (default: '{operation}_{column}')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new case-converted column added
    
    Example:
        >>> string_upper_lower('customers', 'email', 'lower', 'email_normalized')
        # Creates 'email_normalized' column: "USER@DOMAIN.COM" → "user@domain.com"
    """
    if df is not None:
        df_result = df.copy()
        new_col = new_column_name or f"{operation}_{column}"
        if operation.lower() == 'upper':
            df_result[new_col] = df[column].astype(str).str.upper()
        else:
            df_result[new_col] = df[column].astype(str).str.lower()
        return df_result
    
    new_col = new_column_name or f"{operation}_{column}"
    func = operation.upper()
    return pd.read_sql(f"SELECT *, {func}({column}) as {new_col} FROM {table_name}", engine)

@mcp.tool()
def string_substring(table_name: str, column: str, start: int, length: int = None,
                    new_column_name: str = 'substring', df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract a substring from a string column.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column name to extract substring from
        start (int): Starting position (1-based indexing)
        length (int): Number of characters to extract (optional, extracts to end if None)
        new_column_name (str): Name for the new substring column (default: 'substring')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new substring column added
    
    Example:
        >>> string_substring('orders', 'order_id', 1, 3, 'order_prefix')
        # Creates 'order_prefix' column: "ORD12345" → "ORD"
    """
    if df is not None:
        df_result = df.copy()
        if length:
            df_result[new_column_name] = df[column].astype(str).str[start-1:start-1+length]
        else:
            df_result[new_column_name] = df[column].astype(str).str[start-1:]
        return df_result
    
    if length:
        return pd.read_sql(f"SELECT *, SUBSTRING({column}, {start}, {length}) as {new_column_name} FROM {table_name}", engine)
    else:
        return pd.read_sql(f"SELECT *, SUBSTRING({column}, {start}) as {new_column_name} FROM {table_name}", engine)

@mcp.tool()
def string_replace(table_name: str, column: str, old_value: str, new_value: str,
                    new_column_name: str = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Replace all occurrences of a substring in a string column.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column name to perform replacement on
        old_value (str): Substring to replace
        new_value (str): Replacement substring
        new_column_name (str): Name for the new column (default: 'replaced_{column}')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new replaced column added
    
    Example:
        >>> string_replace('products', 'description', 'old', 'new', 'updated_desc')
        # Creates 'updated_desc' column: "old product" → "new product"
    """
    if df is not None:
        df_result = df.copy()
        new_col = new_column_name or f"replaced_{column}"
        df_result[new_col] = df[column].astype(str).str.replace(old_value, new_value)
        return df_result
    
    new_col = new_column_name or f"replaced_{column}"
    return pd.read_sql(f"SELECT *, REPLACE({column}, '{old_value}', '{new_value}') as {new_col} FROM {table_name}", engine)

# ===================== DATE/TIME FUNCTIONS =====================
@mcp.tool()
def date_extract(table_name: str, date_column: str, part: str,
                new_column_name: str = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract specific parts from a date/datetime column.
    
    Args:
        table_name (str): Name of the database table
        date_column (str): Column name containing date/datetime values
        part (str): Date part to extract ('YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'DAYOFWEEK', 'QUARTER')
        new_column_name (str): Name for the new column (default: '{part}_{date_column}')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new extracted date part column added
    
    Example:
        >>> date_extract('sales', 'order_date', 'YEAR', 'order_year')
        # Creates 'order_year' column: "2023-05-15" → 2023
    """
    if df is not None:
        df_result = df.copy()
        new_col = new_column_name or f"{part.lower()}_{date_column}"
        date_series = pd.to_datetime(df[date_column])
        
        if part.upper() == 'YEAR':
            df_result[new_col] = date_series.dt.year
        elif part.upper() == 'MONTH':
            df_result[new_col] = date_series.dt.month
        elif part.upper() == 'DAY':
            df_result[new_col] = date_series.dt.day
        elif part.upper() == 'HOUR':
            df_result[new_col] = date_series.dt.hour
        elif part.upper() == 'MINUTE':
            df_result[new_col] = date_series.dt.minute
        elif part.upper() == 'SECOND':
            df_result[new_col] = date_series.dt.second
        elif part.upper() == 'DAYOFWEEK':
            df_result[new_col] = date_series.dt.dayofweek + 1  # SQL usually starts from 1
        elif part.upper() == 'QUARTER':
            df_result[new_col] = date_series.dt.quarter
        
        return df_result
    
    new_col = new_column_name or f"{part.lower()}_{date_column}"
    return pd.read_sql(f"SELECT *, EXTRACT({part} FROM {date_column}) as {new_col} FROM {table_name}", engine)

@mcp.tool()
def date_add_subtract(table_name: str, date_column: str, interval: int, 
                        unit: str, operation: str = 'add', new_column_name: str = None,
                        df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add or subtract time intervals from a date/datetime column.
    
    Args:
        table_name (str): Name of the database table
        date_column (str): Column name containing date/datetime values
        interval (int): Number of units to add/subtract
        unit (str): Time unit ('DAY', 'MONTH', 'YEAR', 'HOUR', 'MINUTE')
        operation (str): 'add' or 'subtract' (default: 'add')
        new_column_name (str): Name for the new column (default: 'modified_{date_column}')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new modified date column added
    
    Example:
        >>> date_add_subtract('orders', 'order_date', 30, 'DAY', 'add', 'delivery_date')
        # Creates 'delivery_date' column: "2023-05-15" → "2023-06-14"
    """
    if df is not None:
        df_result = df.copy()
        new_col = new_column_name or f"modified_{date_column}"
        date_series = pd.to_datetime(df[date_column])
        
        if unit.upper() == 'DAY':
            offset = pd.Timedelta(days=interval)
        elif unit.upper() == 'MONTH':
            offset = pd.DateOffset(months=interval)
        elif unit.upper() == 'YEAR':
            offset = pd.DateOffset(years=interval)
        elif unit.upper() == 'HOUR':
            offset = pd.Timedelta(hours=interval)
        elif unit.upper() == 'MINUTE':
            offset = pd.Timedelta(minutes=interval)
        else:
            offset = pd.Timedelta(days=interval)
        
        if operation.lower() == 'add':
            df_result[new_col] = date_series + offset
        else:
            df_result[new_col] = date_series - offset
        
        return df_result
    
    new_col = new_column_name or f"modified_{date_column}"
    sign = '+' if operation.lower() == 'add' else '-'
    return pd.read_sql(f"SELECT *, {date_column} {sign} INTERVAL {interval} {unit} as {new_col} FROM {table_name}", engine)

@mcp.tool()
def date_diff(table_name: str, date1_column: str, date2_column: str, 
                unit: str = 'DAY', new_column_name: str = 'date_diff',
                df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate the difference between two date/datetime columns.
    
    Args:
        table_name (str): Name of the database table
        date1_column (str): First date column (minuend)
        date2_column (str): Second date column (subtrahend)
        unit (str): Unit for difference ('DAY', 'HOUR', 'MINUTE', 'SECOND') (default: 'DAY')
        new_column_name (str): Name for the new difference column (default: 'date_diff')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new date difference column added
    
    Example:
        >>> date_diff('orders', 'delivery_date', 'order_date', 'DAY', 'delivery_days')
        # Creates 'delivery_days' column: delivery_date - order_date in days
    """
    if df is not None:
        df_result = df.copy()
        date1 = pd.to_datetime(df[date1_column])
        date2 = pd.to_datetime(df[date2_column])
        diff = date1 - date2
        
        if unit.upper() == 'DAY':
            df_result[new_column_name] = diff.dt.days
        elif unit.upper() == 'HOUR':
            df_result[new_column_name] = diff.dt.total_seconds() / 3600
        elif unit.upper() == 'MINUTE':
            df_result[new_column_name] = diff.dt.total_seconds() / 60
        elif unit.upper() == 'SECOND':
            df_result[new_column_name] = diff.dt.total_seconds()
        else:
            df_result[new_column_name] = diff.dt.days
        
        return df_result
    
    return pd.read_sql(f"SELECT *, DATEDIFF({unit}, {date2_column}, {date1_column}) as {new_column_name} FROM {table_name}", engine)

# ===================== MATHEMATICAL FUNCTIONS =====================

@mcp.tool()
def math_operations(table_name: str, column: str, operation: str,
                    new_column_name: str = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Apply mathematical functions to a numeric column.
    
    Args:
        table_name (str): Name of the database table
        column (str): Column name to apply mathematical function to
        operation (str): Mathematical operation ('ABS', 'CEIL', 'FLOOR', 'ROUND', 'SQRT', 'LOG', 'EXP')
        new_column_name (str): Name for the new column (default: '{operation}_{column}')
        df (Optional[pd.DataFrame]): DataFrame to operate on (if None, uses SQL query)
    
    Returns:
        pd.DataFrame: DataFrame with new mathematical operation column added
    
    Example:
        >>> math_operations('products', 'price', 'ROUND', 'rounded_price')
        # Creates 'rounded_price' column: 19.99 → 20.0
    """
    if df is not None:
        df_result = df.copy()
        new_col = new_column_name or f"{operation}_{column}"
        
        if operation.upper() == 'ABS':
            df_result[new_col] = df[column].abs()
        elif operation.upper() == 'CEIL':
            df_result[new_col] = np.ceil(df[column])
        elif operation.upper() == 'FLOOR':
            df_result[new_col] = np.floor(df[column])
        elif operation.upper() == 'ROUND':
            df_result[new_col] = df[column].round()
        elif operation.upper() == 'SQRT':
            df_result[new_col] = np.sqrt(df[column])
        elif operation.upper() == 'LOG':
            df_result[new_col] = np.log(df[column])
        elif operation.upper() == 'EXP':
            df_result[new_col] = np.exp(df[column])
        
        return df_result
    
    new_col = new_column_name or f"{operation}_{column}"
    return pd.read_sql(f"SELECT *, {operation.upper()}({column}) as {new_col} FROM {table_name}", engine)

# ===================== CONDITIONAL AGGREGATION =====================
@mcp.tool()
def conditional_sum(table_name: str, sum_column: str, condition_column: str,
                    condition_value: Any, group_columns: List[str] = None,
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Sum values in a column where another column meets a specific condition (like SUMIF in Excel).
    
    Args:
        table_name: Name of the database table
        sum_column: Column to sum values from
        condition_column: Column to check condition against
        condition_value: Value that condition_column must equal
        group_columns: Optional columns to group by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with conditional sum results
    
    Example:
        conditional_sum('sales', 'amount', 'region', 'North', ['product'])
        # Returns sum of amounts where region='North', grouped by product
    """
    if df is not None:
        mask = df[condition_column] == condition_value
        if group_columns:
            return df[mask].groupby(group_columns)[sum_column].sum().reset_index()
        else:
            return pd.DataFrame({'conditional_sum': [df[mask][sum_column].sum()]})
    
    if group_columns:
        group_str = ', '.join(group_columns)
        return pd.read_sql(f"""
            SELECT {group_str}, SUM(CASE WHEN {condition_column} = '{condition_value}' THEN {sum_column} ELSE 0 END) as conditional_sum
            FROM {table_name}
            GROUP BY {group_str}
        """, engine)
    else:
        return pd.read_sql(f"""
            SELECT SUM(CASE WHEN {condition_column} = '{condition_value}' THEN {sum_column} ELSE 0 END) as conditional_sum
            FROM {table_name}
        """, engine)

@mcp.tool()
def conditional_count(table_name: str, condition_column: str, condition_value: Any,
                        group_columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Count records where a column meets a specific condition (like COUNTIF in Excel).
    
    Args:
        table_name: Name of the database table
        condition_column: Column to check condition against
        condition_value: Value that condition_column must equal
        group_columns: Optional columns to group by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with conditional count results
    
    Example:
        conditional_count('orders', 'status', 'completed', ['customer_id'])
        # Returns count of completed orders per customer
    """
    if df is not None:
        mask = df[condition_column] == condition_value
        if group_columns:
            return df[mask].groupby(group_columns).size().reset_index(name='conditional_count')
        else:
            return pd.DataFrame({'conditional_count': [mask.sum()]})
    
    if group_columns:
        group_str = ', '.join(group_columns)
        return pd.read_sql(f"""
            SELECT {group_str}, COUNT(CASE WHEN {condition_column} = '{condition_value}' THEN 1 END) as conditional_count
            FROM {table_name}
            GROUP BY {group_str}
        """, engine)
    else:
        return pd.read_sql(f"""
            SELECT COUNT(CASE WHEN {condition_column} = '{condition_value}' THEN 1 END) as conditional_count
            FROM {table_name}
        """, engine)

# ===================== STATISTICAL FUNCTIONS =====================
@mcp.tool()
def correlation(table_name: str, column1: str, column2: str,
                group_columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate Pearson correlation coefficient between two numeric columns.
    
    Args:
        table_name: Name of the database table
        column1: First column for correlation analysis
        column2: Second column for correlation analysis
        group_columns: Optional columns to group by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with correlation values (range: -1 to 1)
    
    Example:
        correlation('sales', 'price', 'quantity', ['category'])
        # Returns correlation between price and quantity for each category
    """
    if df is not None:
        if group_columns:
            return df.groupby(group_columns)[[column1, column2]].corr().unstack().iloc[:, 1].reset_index(name='correlation')
        else:
            corr_value = df[column1].corr(df[column2])
            return pd.DataFrame({'correlation': [corr_value]})
    
    # Note: SQL correlation functions vary by database
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    if group_columns:
        return data.groupby(group_columns)[[column1, column2]].corr().unstack().iloc[:, 1].reset_index(name='correlation')
    else:
        corr_value = data[column1].corr(data[column2])
        return pd.DataFrame({'correlation': [corr_value]})

@mcp.tool()
def percentiles(table_name: str, column: str, percentiles: List[float],
                group_columns: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate specified percentiles for a numeric column.
    
    Args:
        table_name: Name of the database table
        column: Column to calculate percentiles for
        percentiles: List of percentile values (0.0 to 1.0)
        group_columns: Optional columns to group by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with percentile values as columns
    
    Example:
        percentiles('sales', 'amount', [0.25, 0.5, 0.75], ['region'])
        # Returns 25th, 50th, 75th percentiles of amount by region
    """
    if df is not None:
        if group_columns:
            return df.groupby(group_columns)[column].quantile(percentiles).unstack().reset_index()
        else:
            result = df[column].quantile(percentiles).to_frame().T
            result.columns = [f'p{int(p*100)}' for p in percentiles]
            return result
    
    # For SQL, this would be database-specific
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    if group_columns:
        return data.groupby(group_columns)[column].quantile(percentiles).unstack().reset_index()
    else:
        result = data[column].quantile(percentiles).to_frame().T
        result.columns = [f'p{int(p*100)}' for p in percentiles]
        return result

# ===================== ADVANCED ANALYTICS =====================
@mcp.tool()
def moving_average(table_name: str, column: str, window: int, 
                    partition_by: List[str] = None, order_by: List[str] = None,
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate rolling/moving average over a specified window of rows.
    
    Args:
        table_name: Name of the database table
        column: Column to calculate moving average for
        window: Number of rows to include in rolling calculation
        partition_by: Optional columns to partition data by
        order_by: Optional columns to order data by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        Original DataFrame with additional 'moving_avg' column
    
    Example:
        moving_average('stock_prices', 'price', 7, ['symbol'], ['date'])
        # Returns 7-day moving average of price for each stock symbol
    """
    if df is not None:
        df_result = df.copy()
        if partition_by:
            df_result['moving_avg'] = df.groupby(partition_by)[column].rolling(window=window, min_periods=1).mean().reset_index(drop=True)
        else:
            df_result['moving_avg'] = df[column].rolling(window=window, min_periods=1).mean()
        return df_result
    
    # For SQL window functions
    partition_str = ', '.join(partition_by) if partition_by else ''
    order_str = ', '.join(order_by) if order_by else column
    partition_clause = f"PARTITION BY {partition_str}" if partition_by else ""
    
    return pd.read_sql(f"""
        SELECT *, AVG({column}) OVER ({partition_clause} ORDER BY {order_str} 
                                        ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW) as moving_avg
        FROM {table_name}
    """, engine)

@mcp.tool()
def year_over_year_growth(table_name: str, value_column: str, date_column: str,
                            partition_by: List[str] = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate year-over-year growth rate as a percentage.
    
    Args:
        table_name: Name of the database table
        value_column: Column containing values to compare
        date_column: Column containing date information
        partition_by: Optional columns to partition data by
        df: Optional DataFrame to use instead of database query
    
    Returns:
        Original DataFrame with additional 'yoy_growth' column (percentage)
    
    Example:
        year_over_year_growth('revenue', 'amount', 'date', ['product'])
        # Returns YoY growth rate for each product's revenue
    """
    if df is not None:
        df_result = df.copy()
        df_result[date_column] = pd.to_datetime(df_result[date_column])
        df_result['year'] = df_result[date_column].dt.year
        
        if partition_by:
            group_cols = partition_by + ['year']
        else:
            group_cols = ['year']
        
        # Calculate previous year value
        df_result['prev_year_value'] = df_result.groupby(partition_by if partition_by else [None])[value_column].shift(1)
        df_result['yoy_growth'] = ((df_result[value_column] - df_result['prev_year_value']) / df_result['prev_year_value'] * 100)
        
        return df_result
    
    # This would require more complex SQL with LAG function
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return year_over_year_growth(table_name, value_column, date_column, partition_by, data)

@mcp.tool()
def cohort_analysis(table_name: str, user_column: str, date_column: str,
                    value_column: str = None, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform cohort analysis to track user behavior over time periods.
    
    Args:
        table_name: Name of the database table
        user_column: Column identifying unique users
        date_column: Column containing transaction dates
        value_column: Optional column for value-based analysis (default: user count)
        df: Optional DataFrame to use instead of database query
    
    Returns:
        Pivot table with cohort months as index and period numbers as columns
    
    Example:
        cohort_analysis('purchases', 'user_id', 'purchase_date', 'amount')
        # Returns cohort analysis showing revenue retention by month
    """
    if df is not None:
        df_cohort = df.copy()
        df_cohort[date_column] = pd.to_datetime(df_cohort[date_column])
        
        # Get first purchase date for each user
        df_cohort['first_purchase'] = df_cohort.groupby(user_column)[date_column].transform('min')
        df_cohort['cohort_month'] = df_cohort['first_purchase'].dt.to_period('M')
        df_cohort['period_month'] = df_cohort[date_column].dt.to_period('M')
        df_cohort['period_number'] = (df_cohort['period_month'] - df_cohort['cohort_month']).apply(attrgetter('n'))
        
        if value_column:
            cohort_data = df_cohort.groupby(['cohort_month', 'period_number'])[value_column].sum().reset_index()
        else:
            cohort_data = df_cohort.groupby(['cohort_month', 'period_number'])[user_column].nunique().reset_index()
        
        return cohort_data.pivot(index='cohort_month', columns='period_number', values=value_column if value_column else user_column)
    
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return cohort_analysis(table_name, user_column, date_column, value_column, data)

# ===================== DATA QUALITY FUNCTIONS =====================
@mcp.tool()
def data_profiling(table_name: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate comprehensive statistical profile for all columns in a dataset.
    
    Args:
        table_name: Name of the database table
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with column statistics including data types, null counts, 
        unique values, and basic statistics for numeric columns
    
    Example:
        data_profiling('customers')
        # Returns profile with stats for each column in customers table
    """
    if df is not None:
        profile_data = []
        for col in df.columns:
            profile_data.append({
                'column_name': col,
                'data_type': str(df[col].dtype),
                'non_null_count': df[col].count(),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'min_value': df[col].min() if df[col].dtype in ['int64', 'float64'] else None,
                'max_value': df[col].max() if df[col].dtype in ['int64', 'float64'] else None,
                'mean_value': df[col].mean() if df[col].dtype in ['int64', 'float64'] else None
            })
        return pd.DataFrame(profile_data)
    
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return data_profiling(table_name, data)

@mcp.tool()
def find_duplicates(table_name: str, columns: List[str], 
                    df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Identify and return all duplicate records based on specified columns.
    
    Args:
        table_name: Name of the database table
        columns: List of column names to check for duplicates
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame containing all rows that have duplicates, sorted by duplicate columns
    
    Example:
        find_duplicates('orders', ['customer_id', 'order_date'])
        # Returns all orders where same customer has multiple orders on same date
    """
    if df is not None:
        return df[df.duplicated(subset=columns, keep=False)].sort_values(columns)
    
    cols_str = ', '.join(columns)
    return pd.read_sql(f"""
        SELECT * FROM {table_name}
        WHERE ({cols_str}) IN (
            SELECT {cols_str}
            FROM {table_name}
            GROUP BY {cols_str}
            HAVING COUNT(*) > 1
        )
        ORDER BY {cols_str}
    """, engine)

@mcp.tool()
def missing_data_analysis(table_name: str, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Analyze missing data patterns across all columns in a dataset.
    
    Args:
        table_name: Name of the database table
        df: Optional DataFrame to use instead of database query
    
    Returns:
        DataFrame with missing data statistics per column, sorted by missing percentage
        (includes count, percentage, and non-missing count)
    
    Example:
        missing_data_analysis('survey_responses')
        # Returns analysis showing which survey questions have most missing answers
    """
    if df is not None:
        missing_data = []
        total_rows = len(df)
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            missing_data.append({
                'column_name': col,
                'missing_count': null_count,
                'missing_percentage': (null_count / total_rows) * 100,
                'non_missing_count': total_rows - null_count
            })
        
        return pd.DataFrame(missing_data).sort_values('missing_percentage', ascending=False)
    
    data = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    return missing_data_analysis(table_name, data)


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """
    Constructs a Starlette app with SSE and message endpoints.

    Args:
        mcp_server (Server): The core MCP server instance.
        debug (bool): Enable debug mode for verbose logs.

    Returns:
        Starlette: The full Starlette app with routes.
    """
    # Create SSE transport handler to manage long-lived SSE connections
    sse = SseServerTransport("/messages/")

    # This function is triggered when a client connects to `/sse`
    async def handle_sse(request: Request) -> None:
        """
        Handles a new SSE client connection and links it to the MCP server.
        """
        # Open an SSE connection, then hand off read/write streams to MCP
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # Low-level send function provided by Starlette
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # Return the Starlette app with configured endpoints
    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),          # For initiating SSE connection
            Mount("/messages/", app=sse.handle_post_message),  # For POST-based communication
        ],
    )

if __name__ == "__main__":
    # Get the underlying MCP server instance from FastMCP
    mcp_server = mcp._mcp_server  # Accessing private member (acceptable here)

    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8081, help='Port to listen on')
    args = parser.parse_args()

    # Build the Starlette app with debug mode enabled
    starlette_app = create_starlette_app(mcp_server, debug=True)

    # Launch the server using Uvicorn
    uvicorn.run(starlette_app, host=args.host, port=args.port)
    
# Example usage:
"""
# Initialize the SQL operations class
sql_ops = SQLOperations(engine)

# Basic operations
all_data = sql_ops.select_all('users')
specific_columns = sql_ops.select_columns('users', ['name', 'email'])
filtered_data = sql_ops.where_condition('users', 'age', '>', 25)

# Aggregations
user_counts = sql_ops.group_by_count('orders', ['user_id'])
sales_by_region = sql_ops.group_by_sum('sales', ['region'], ['amount'])

# Joins
user_orders = sql_ops.inner_join('users', 'orders', 'user_id')

# Window functions
ranked_data = sql_ops.rank_function('sales', ['region'], ['amount'])
running_totals = sql_ops.running_total('sales', 'amount', ['region'], ['date'])

# Working with DataFrames
df_result = sql_ops.select_columns('users', ['name', 'age'], df=existing_dataframe
"""