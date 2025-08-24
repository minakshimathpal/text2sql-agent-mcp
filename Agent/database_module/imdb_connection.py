import pandas as pd
import sqlite3
from typing import Optional

class IMDBDatabase:
    """
    A class to handle IMDB database connection and data cleaning operations.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the database cleaner with the path to the database file.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
    
    def connect(self):
        """
        Establish connection to the database.
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Successfully connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("Database connection closed.")
    
    def clean_database(self, commit: bool = False):
        """
        Perform data cleaning operations on the database.
        
        Args:
            commit (bool): Whether to commit changes to the database. 
                          Default is False for safety.
        """
        if not self.conn or not self.cursor:
            raise Exception("Database connection not established. Call connect() first.")
        
        try:
            print("Starting database cleaning operations...")
            
            # Clean Movie table
            print("Cleaning Movie table...")
            self.cursor.execute('UPDATE Movie SET year = REPLACE(year, "I", "");')
            self.cursor.execute('UPDATE Movie SET year = REPLACE(year, "V", "");')
            self.cursor.execute('UPDATE Movie SET year = REPLACE(year, "X ", "");')
            self.cursor.execute('UPDATE Movie SET title = LTRIM(title);')
            self.cursor.execute('UPDATE Movie SET year = RTRIM(LTRIM(year));')
            self.cursor.execute('UPDATE Movie SET rating = RTRIM(LTRIM(rating));')
            self.cursor.execute('UPDATE Movie SET num_votes = RTRIM(LTRIM(num_votes));')
            
            # Clean M_Producer table
            print("Cleaning M_Producer table...")
            self.cursor.execute('UPDATE M_Producer SET pid = RTRIM(LTRIM(pid));')
            self.cursor.execute('UPDATE M_Producer SET mid = RTRIM(LTRIM(mid));')
            
            # Clean M_Director table
            print("Cleaning M_Director table...")
            self.cursor.execute('UPDATE M_Director SET pid = RTRIM(LTRIM(pid));')
            self.cursor.execute('UPDATE M_Director SET mid = RTRIM(LTRIM(mid));')
            
            # Clean M_Cast table
            print("Cleaning M_Cast table...")
            self.cursor.execute('UPDATE M_Cast SET pid = RTRIM(LTRIM(pid));')
            self.cursor.execute('UPDATE M_Cast SET mid = RTRIM(LTRIM(mid));')
            
            # Clean M_Genre table
            print("Cleaning M_Genre table...")
            self.cursor.execute('UPDATE M_Genre SET gid = RTRIM(LTRIM(gid));')
            self.cursor.execute('UPDATE M_Genre SET mid = RTRIM(LTRIM(mid));')
            
            # Clean Genre table
            print("Cleaning Genre table...")
            self.cursor.execute('UPDATE Genre SET gid = RTRIM(LTRIM(gid));')
            self.cursor.execute('UPDATE Genre SET name = RTRIM(LTRIM(name));')
            
            # Clean Person table
            print("Cleaning Person table...")
            self.cursor.execute('UPDATE Person SET name = RTRIM(LTRIM(name));')
            self.cursor.execute('UPDATE Person SET pid = RTRIM(LTRIM(pid));')
            self.cursor.execute('UPDATE Person SET gender = RTRIM(LTRIM(gender));')
            
            if commit:
                self.conn.commit()
                print("Database changes committed successfully!")
            else:
                print("Database cleaning completed. Changes not committed (set commit=True to save changes).")
                
        except sqlite3.Error as e:
            print(f"Error during database cleaning: {e}")
            if self.conn:
                self.conn.rollback()
            raise
    

    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Example usage:
if __name__ == "__main__":
    db_path = r"C:\Users\aniru\Downloads\agentic_framework_mcp\text2sql_agent_mcp\employee-sample-database\imdb\IMDB.db"
    cleaner = IMDBDatabase(db_path)
    cleaner.connect()
    try:
        cleaner.clean_database(commit=False)
    finally:
        cleaner.disconnect()