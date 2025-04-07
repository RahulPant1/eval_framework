"""
Database module for MongoDB operations in RAG evaluation framework.

Contains functions for:
- Connecting to MongoDB
- Creating collections
- Inserting initial data
- Managing datasets and chunks
"""

import pymongo
from bson.objectid import ObjectId
from datetime import datetime
import logging

# Configuration (will be moved to config.py later)
config = {
    "mongodb_uri": "mongodb://localhost:27017/",
    "mongodb_db": "rag_evaluation",
    "log_file": "rag_evaluation.log"
}

# Set up logging
logging.basicConfig(filename=config["log_file"], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_mongodb(uri=config["mongodb_uri"], db_name=config["mongodb_db"]):
    """Connects to MongoDB."""
    try:
        client = pymongo.MongoClient(uri)
        db = client[db_name]
        logger.info(f"Connected to MongoDB: {uri}, Database: {db_name}")
        return db
    except pymongo.errors.ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def create_collections(db):
    """Creates the necessary collections in MongoDB if they don't already exist."""
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
    existing_collections = db.list_collection_names()
    for collection_name in collections:
        if collection_name not in existing_collections:
            db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created.")
        else:
            logger.info(f"Collection '{collection_name}' already exists and will be reused.")


def insert_initial_data(db):
    """Inserts initial data into the collections if it doesn't already exist."""
    # llms
    if db.llms.count_documents({}) == 0:
        llm1 = {"llm_name": "Gemini 1.5 Flash", "llm_description": "Google's Gemini 1.5 Flash", "llm_provider": "Google"}
        llm2 = {"llm_name": "Llama 2", "llm_description": "Meta's Llama 2", "llm_provider": "Meta"}
        db.llms.insert_many([llm1, llm2])
        logger.info("LLMs inserted.")
    else:
        logger.info("LLMs collection already has data, skipping insertion.")

    # llm_endpoints
    if db.llm_endpoints.count_documents({}) == 0:
        # Make sure we have the LLMs first
        gemini_llm = db.llms.find_one({"llm_name": "Gemini 1.5 Flash"})
        llama_llm = db.llms.find_one({"llm_name": "Llama 2"})
        
        if gemini_llm and llama_llm:
            endpoint1 = {"llm_id": gemini_llm["_id"],
                        "endpoint_url": "https://ai.googleapis.com/v1beta1/models/gemini-1.5-flash:generateContent",
                        "endpoint_type": "REST API"}
            endpoint2 = {"llm_id": llama_llm["_id"],
                        "endpoint_url": "http://localhost:8080/completions", "endpoint_type": "REST API"}
            db.llm_endpoints.insert_many([endpoint1, endpoint2])
            logger.info("LLM endpoints inserted.")
        else:
            logger.warning("Could not find required LLMs for endpoints, skipping endpoint insertion.")
    else:
        logger.info("LLM endpoints collection already has data, skipping insertion.")

    # model_deployments
    if db.model_deployments.count_documents({}) == 0:
        # Make sure we have the LLMs first
        gemini_llm = db.llms.find_one({"llm_name": "Gemini 1.5 Flash"})
        llama_llm = db.llms.find_one({"llm_name": "Llama 2"})
        
        if gemini_llm and llama_llm:
            deployment1 = {
                "llm_id": gemini_llm["_id"],
                "application_name": "Customer Service Chatbot",
                "application_description": "Chatbot for customer queries",
                "status": "Active",
                "deployment_timestamp": datetime.now()
            }
            deployment2 = {
                "llm_id": llama_llm["_id"],
                "application_name": "Document Summarization Tool",
                "application_description": "Tool for summarizing documents",
                "status": "Active",
                "deployment_timestamp": datetime.now()
            }
            db.model_deployments.insert_many([deployment1, deployment2])
            logger.info("Model deployments inserted.")
        else:
            logger.warning("Could not find required LLMs for deployments, skipping deployment insertion.")
    else:
        logger.info("Model deployments collection already has data, skipping insertion.")

    # Vector Store
    if db.vector_store.count_documents({}) == 0:
        vector_store_data = {
            "store_name": "Chroma",
            "store_type": "Vector Database",
            "connection_details": {
                "persist_directory": "./chroma_db",
                "client_settings": {},
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }
        db.vector_store.insert_one(vector_store_data)
        logger.info("Vector store inserted")
    else:
        logger.info("Vector store collection already has data, skipping insertion.")

    # evaluation_harnesses
    if db.evaluation_harnesses.count_documents({}) == 0:
        harness1 = {"harness_name": "RAG Faithfulness", "harness_type": "RAG Evaluation",
                    "description": "Evaluates the faithfulness of RAG responses"}
        harness2 = {"harness_name": "Context Relevance", "harness_type": "RAG Evaluation",
                    "description": "Evaluates the relevance of retrieved context"}
        db.evaluation_harnesses.insert_many([harness1, harness2])
        logger.info("Evaluation harnesses inserted.")
    else:
        logger.info("Evaluation harnesses collection already has data, skipping insertion.")

    # test_suites
    if db.test_suites.count_documents({}) == 0:
        # Check if we have evaluation harnesses first
        rag_harness = db.evaluation_harnesses.find_one({"harness_name": "RAG Faithfulness"})
        context_harness = db.evaluation_harnesses.find_one({"harness_name": "Context Relevance"})
        
        if rag_harness and context_harness:
            # Use a parameter for dataset_id instead of hardcoding
            # This allows the function to work with any dataset
            datasets = list(db.datasets.find().limit(1))
            if datasets:
                dataset_id = datasets[0]["_id"]
                
                suite1 = {
                    "harness_id": rag_harness["_id"],
                    "suite_name": "Faithfulness Test Suite",
                    "suite_description": "Suite for testing faithfulness",
                    "suite_type": "End-to-end",
                    "suite_parameters": {"threshold": 0.9},
                    "dataset_id": dataset_id
                }
                suite2 = {
                    "harness_id": context_harness["_id"],
                    "suite_name": "Retrieval Relevance Test Suite",
                    "suite_description": "Suite for testing retrieval",
                    "suite_type": "Component",
                    "suite_parameters": {"top_k": 5},
                    "dataset_id": dataset_id
                }
                db.test_suites.insert_many([suite1, suite2])
                logger.info("Test suites inserted.")
            else:
                logger.warning("No datasets found, skipping test suite insertion.")
        else:
            logger.warning("Could not find required evaluation harnesses, skipping test suite insertion.")
    else:
        logger.info("Test suites collection already has data, skipping insertion.")