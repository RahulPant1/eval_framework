#!/usr/bin/env python3
"""
JSON Database initialization script for RAG Evaluation Framework.

This script is used to:
1. Create JSON file collections
2. Insert initial data

Run this script only once to set up the JSON database infrastructure.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the JSON database directory
JSON_DB_DIR = Path('json_db')

def ensure_json_db_dir():
    """Ensure the json_db directory exists."""
    JSON_DB_DIR.mkdir(exist_ok=True)
    logger.info(f"Ensured JSON database directory exists at {JSON_DB_DIR}")

def create_json_collections():
    """Creates the necessary JSON files if they don't already exist."""
    collections = [
        "llms",
        "llm_endpoints",
        "model_deployments",
        "datasets",
        "prompt_templates",
        "synthetic_data",
        "evaluation_harnesses",
        "test_suites",
        "evaluation_runs",
        "test_results",
        "metrics",
        "metrics_definitions",
        "vector_store"
    ]
    
    for collection_name in collections:
        json_file = JSON_DB_DIR / f"{collection_name}.json"
        if not json_file.exists():
            with open(json_file, 'w') as f:
                json.dump([], f)
            logger.info(f"JSON file '{collection_name}.json' created.")
        else:
            logger.info(f"JSON file '{collection_name}.json' already exists and will be reused.")

def read_json_file(collection_name: str) -> List[Dict[str, Any]]:
    """Read data from a JSON file."""
    json_file = JSON_DB_DIR / f"{collection_name}.json"
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def write_json_file(collection_name: str, data: List[Dict[str, Any]]):
    """Write data to a JSON file."""
    json_file = JSON_DB_DIR / f"{collection_name}.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def insert_initial_data():
    """Inserts initial data into the JSON files if they don't already exist."""
    # llms
    llms_data = read_json_file("llms")
    if not llms_data:
        llm1 = {"_id": "llm_1", "llm_name": "Gemini 1.5 Flash", "llm_description": "Google's Gemini 1.5 Flash", "llm_provider": "Google"}
        llm2 = {"_id": "llm_2", "llm_name": "Llama 2", "llm_description": "Meta's Llama 2", "llm_provider": "Meta"}
        write_json_file("llms", [llm1, llm2])
        logger.info("LLMs data inserted.")
    else:
        logger.info("LLMs collection already has data, skipping insertion.")

    # llm_endpoints
    endpoints_data = read_json_file("llm_endpoints")
    if not endpoints_data:
        endpoint1 = {
            "_id": "endpoint_1",
            "llm_id": "llm_1",
            "endpoint_url": "https://ai.googleapis.com/v1beta1/models/gemini-1.5-flash:generateContent",
            "endpoint_type": "REST API"
        }
        endpoint2 = {
            "_id": "endpoint_2",
            "llm_id": "llm_2",
            "endpoint_url": "http://localhost:8080/completions",
            "endpoint_type": "REST API"
        }
        write_json_file("llm_endpoints", [endpoint1, endpoint2])
        logger.info("LLM endpoints data inserted.")
    else:
        logger.info("LLM endpoints collection already has data, skipping insertion.")

def initialize_json_database():
    """Initialize JSON database with collections and initial data."""
    try:
        logger.info("Ensuring JSON database directory exists...")
        ensure_json_db_dir()
        
        logger.info("Creating JSON collections...")
        create_json_collections()
        
        logger.info("Inserting initial data...")
        insert_initial_data()
        
        logger.info("JSON database initialization completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing JSON database: {e}")
        return False

if __name__ == "__main__":
    initialize_json_database()