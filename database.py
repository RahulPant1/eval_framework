"""Database module for RAG evaluation framework.

Contains functions for:
- Connecting to database (MongoDB or JSON files)
- Creating collections
- Inserting initial data
- Managing datasets and chunks

This module now uses the database_adapter module to provide a flexible
database abstraction layer that supports both MongoDB and JSON files.
"""

import logging
from database_adapter import connect_to_database, create_collections, insert_initial_data

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_mongodb(uri=None, db_name=None):
    """Connects to database (MongoDB or JSON files).
    
    This function is maintained for backward compatibility.
    It now uses the database_adapter module to connect to either MongoDB or JSON files.
    """
    return connect_to_database(uri, db_name)

# The following functions are now imported from database_adapter
# They are kept here as empty functions for backward compatibility

# def create_collections(db):
#     """Creates the necessary collections if they don't already exist."""
#     pass

# def insert_initial_data(db):
#     """Inserts initial data into the collections if it doesn't already exist."""
#     pass