#!/usr/bin/env python3
"""
Database initialization script for RAG Evaluation Framework.

This script is used to:
1. Create MongoDB collections
2. Insert initial data

Run this script only once to set up the database infrastructure.
"""

import logging
from database import connect_to_mongodb, create_collections, insert_initial_data

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize MongoDB database with collections and initial data."""
    try:
        logger.info("Setting up database connection...")
        db = connect_to_mongodb()
        
        logger.info("Creating collections...")
        create_collections(db)
        
        logger.info("Inserting initial data...")
        insert_initial_data(db)
        
        logger.info("Database initialization completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    initialize_database()