"""Database adapter module for RAG evaluation framework.

Provides a flexible database abstraction layer that supports:
- MongoDB (primary database)
- JSON files (fallback when MongoDB is unavailable)

This adapter maintains the same API regardless of the underlying storage mechanism.
"""

import os
import json
import logging
import pymongo
from bson.objectid import ObjectId
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import pathlib

# Configuration (will be moved to config.py later)
config = {
    "mongodb_uri": "mongodb://localhost:27017/",
    "mongodb_db": "rag_evaluation",
    "json_db_path": "./json_db",  # Directory to store JSON files
    "log_file": "rag_evaluation.log"
}

# Set up logging
logging.basicConfig(filename=config["log_file"], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseDatabase:
    """Base database interface that defines common operations."""
    
    def __init__(self):
        self.collections = {}
    
    def list_collection_names(self) -> List[str]:
        """Returns a list of collection names."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def create_collection(self, collection_name: str) -> None:
        """Creates a collection if it doesn't exist."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_collection(self, collection_name: str) -> Any:
        """Returns a collection object."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def insert_one(self, collection_name: str, document: Dict) -> str:
        """Inserts a document into a collection and returns its ID."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        """Inserts multiple documents into a collection and returns their IDs."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def find_one(self, collection_name: str, query: Dict = None) -> Optional[Dict]:
        """Finds a single document matching the query."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def find(self, collection_name: str, query: Dict = None, limit: int = 0) -> List[Dict]:
        """Finds all documents matching the query."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def update_one(self, collection_name: str, query: Dict, update: Dict) -> int:
        """Updates a single document matching the query."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def count_documents(self, collection_name: str, query: Dict = None) -> int:
        """Counts documents matching the query."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def drop_collection(self, collection_name: str) -> None:
        """Drops a collection."""
        raise NotImplementedError("Subclasses must implement this method")


class MongoDBDatabase(BaseDatabase):
    """MongoDB implementation of the database interface."""
    
    def __init__(self, uri: str = config["mongodb_uri"], db_name: str = config["mongodb_db"]):
        super().__init__()
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB: {uri}, Database: {db_name}")
    
    def __getattr__(self, name):
        """Dynamically create collection accessors to mimic MongoDB style access."""
        # Create the collection if it doesn't exist
        if name not in self.db.list_collection_names():
            self.create_collection(name)
            logger.info(f"Automatically created collection '{name}' on first access")
        
        # Return a CollectionAccessor that provides MongoDB-like methods
        return CollectionAccessor(self, name)
    
    def list_collection_names(self) -> List[str]:
        return self.db.list_collection_names()
    
    def create_collection(self, collection_name: str) -> None:
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created in MongoDB.")
        else:
            logger.info(f"Collection '{collection_name}' already exists in MongoDB and will be reused.")
    
    def get_collection(self, collection_name: str) -> Any:
        return self.db[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict) -> str:
        result = self.db[collection_name].insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        result = self.db[collection_name].insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def find_one(self, collection_name: str, query: Dict = None) -> Optional[Dict]:
        if query is None:
            query = {}
        
        # Handle ID conversion for MongoDB
        if '_id' in query and isinstance(query['_id'], str):
            try:
                query['_id'] = ObjectId(query['_id'])
            except:
                logger.error(f"Invalid ID format: {query['_id']}")
                return None
                
        return self.db[collection_name].find_one(query)
    
    def find(self, collection_name: str, query: Dict = None, limit: int = 0) -> List[Dict]:
        if query is None:
            query = {}
        cursor = self.db[collection_name].find(query)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def update_one(self, collection_name: str, query: Dict, update: Dict) -> int:
        result = self.db[collection_name].update_one(query, update)
        return result.modified_count
    
    def count_documents(self, collection_name: str, query: Dict = None) -> int:
        if query is None:
            query = {}
        return self.db[collection_name].count_documents(query)
    
    def drop_collection(self, collection_name: str) -> None:
        self.db.drop_collection(collection_name)
        logger.info(f"Collection '{collection_name}' dropped from MongoDB.")


class JSONFileDatabase(BaseDatabase):
    """JSON file implementation of the database interface."""
    
    def __init__(self, db_path: str = config["json_db_path"]):
        super().__init__()
        self.db_path = pathlib.Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collections = {}
        logger.info(f"Using JSON file database at: {db_path}")
        
        # Load existing collections
        for file_path in self.db_path.glob("*.json"):
            collection_name = file_path.stem
            self.collections[collection_name] = self._load_collection(collection_name)
    
    def __getattr__(self, name):
        """Dynamically create collection accessors to mimic MongoDB style access."""
        # Create the collection if it doesn't exist
        if name not in self.collections:
            self.create_collection(name)
            logger.info(f"Automatically created collection '{name}' on first access")
        
        # Return a CollectionAccessor that provides MongoDB-like methods
        return CollectionAccessor(self, name)
    
    def _get_collection_path(self, collection_name: str) -> pathlib.Path:
        """Returns the path to a collection file."""
        return self.db_path / f"{collection_name}.json"
    
    def _load_collection(self, collection_name: str) -> List[Dict]:
        """Loads a collection from a JSON file."""
        path = self._get_collection_path(collection_name)
        if path.exists():
            with open(path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {path}")
                    return []
        return []
    
    def _save_collection(self, collection_name: str) -> None:
        """Saves a collection to a JSON file."""
        path = self._get_collection_path(collection_name)
        with open(path, 'w') as f:
            json.dump(self.collections[collection_name], f, default=str, indent=2)
    
    def _ensure_collection(self, collection_name: str) -> None:
        """Ensures a collection exists in memory."""
        if collection_name not in self.collections:
            self.collections[collection_name] = self._load_collection(collection_name)
    
    def _generate_id(self) -> str:
        """Generates a unique ID for a document."""
        return str(ObjectId())
    
    def list_collection_names(self) -> List[str]:
        return list(self.collections.keys())
    
    def create_collection(self, collection_name: str) -> None:
        if collection_name not in self.collections:
            self.collections[collection_name] = []
            self._save_collection(collection_name)
            logger.info(f"Collection '{collection_name}' created in JSON file database.")
        else:
            logger.info(f"Collection '{collection_name}' already exists in JSON file database and will be reused.")
    
    def get_collection(self, collection_name: str) -> List[Dict]:
        self._ensure_collection(collection_name)
        return self.collections[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict) -> str:
        self._ensure_collection(collection_name)
        
        # Create a copy of the document to avoid modifying the original
        doc_copy = document.copy()
        
        # Add _id if not present
        if '_id' not in doc_copy:
            doc_copy['_id'] = ObjectId()
        elif isinstance(doc_copy['_id'], str):
            try:
                doc_copy['_id'] = ObjectId(doc_copy['_id'])
            except Exception:
                doc_copy['_id'] = ObjectId()
        
        self.collections[collection_name].append(doc_copy)
        self._save_collection(collection_name)
        return str(doc_copy['_id'])
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        self._ensure_collection(collection_name)
        ids = []
        
        for document in documents:
            # Create a copy of the document to avoid modifying the original
            doc_copy = document.copy()
            
            # Add _id if not present
            if '_id' not in doc_copy:
                doc_copy['_id'] = self._generate_id()
            
            self.collections[collection_name].append(doc_copy)
            ids.append(str(doc_copy['_id']))
        
        self._save_collection(collection_name)
        return ids
    
    def find_one(self, collection_name: str, query: Dict = None) -> Optional[Dict]:
        self._ensure_collection(collection_name)
        # Process query - convert ObjectIds in query values to strings if needed for comparison
        processed_query = self._process_query(query)

        for doc in self.collections[collection_name]:
            if self._matches_query(doc, processed_query):
                return self._stringify_ids(doc) # Return doc with string IDs
        return None
    
    def find(self, collection_name: str, query: Dict = None, limit: int = 0) -> List[Dict]:
        self._ensure_collection(collection_name)
        # Process query - convert ObjectIds in query values to strings if needed for comparison
        processed_query = self._process_query(query)

        results = []
        count = 0
        for doc in self.collections[collection_name]:
           if self._matches_query(doc, processed_query):
                results.append(self._stringify_ids(doc)) # Return docs with string IDs
                count += 1
                if limit > 0 and count >= limit:
                    break

        return results
    
    def _process_query(self, query: Optional[Dict]) -> Dict:
        """Converts ObjectId values in query to strings for consistent comparison."""
        if query is None:
            return {}
        processed = {}
        for key, value in query.items():
            # Convert top-level ObjectIds
            if isinstance(value, ObjectId):
                processed[key] = str(value)
            # Handle operators like $in containing ObjectIds
            elif isinstance(value, dict) and '$in' in value and isinstance(value['$in'], list):
                 processed[key] = {
                     '$in': [str(item) if isinstance(item, ObjectId) else item for item in value['$in']]
                 }
            # TODO: Handle other potential operators or nested ObjectIds if necessary
            else:
                processed[key] = value
        return processed

    def update_one(self, collection_name: str, query: Dict, update: Dict) -> int:
        self._ensure_collection(collection_name)
        
        for i, doc in enumerate(self.collections[collection_name]):
            if self._matches_query(doc, query):
                # Handle MongoDB-style update operators
                if '$set' in update:
                    for key, value in update['$set'].items():
                        doc[key] = value
                else:
                    # Replace the entire document except _id
                    doc_id = doc['_id']
                    self.collections[collection_name][i] = update
                    self.collections[collection_name][i]['_id'] = doc_id
                
                self._save_collection(collection_name)
                return 1
        
        return 0
    
    def count_documents(self, collection_name: str, query: Dict = None) -> int:
        self._ensure_collection(collection_name)
        
        if query is None:
            query = {}
        
        return len([doc for doc in self.collections[collection_name] if self._matches_query(doc, query)])
    
    def drop_collection(self, collection_name: str) -> None:
        if collection_name in self.collections:
            del self.collections[collection_name]
            
            # Remove the JSON file
            path = self._get_collection_path(collection_name)
            if path.exists():
                path.unlink()
            
            logger.info(f"Collection '{collection_name}' dropped from JSON file database.")
    
    def _matches_query(self, document: Dict, query: Dict) -> bool:
        """Checks if a document matches a query, comparing ID fields as strings."""
        for key, query_value in query.items():
            doc_value = document.get(key)

            # Determine if the key likely represents an ID
            is_id_key = key.endswith('_id')

            # Standardize comparison values:
            # Convert query value to string if it's an ObjectId
            query_value_comp = str(query_value) if isinstance(query_value, ObjectId) else query_value
            # Convert document value to string if it's an ObjectId
            doc_value_comp = str(doc_value) if isinstance(doc_value, ObjectId) else doc_value

            # --- Comparison Logic ---

            # Handle operators like $in
            if isinstance(query_value_comp, dict) and '$in' in query_value_comp:
                 # Ensure the list values are comparable (convert potential ObjectIds in the list)
                op_list = [str(item) if isinstance(item, ObjectId) else item for item in query_value_comp['$in']]
                # Compare the potentially stringified document value against the processed list
                if doc_value_comp not in op_list:
                     return False
                # Add handling for other operators ($gt, $lt, etc.) here if needed
            # Direct comparison
            elif is_id_key:
                # Always compare ID fields as strings
                if str(doc_value_comp) != str(query_value_comp):
                    return False
            elif doc_value_comp != query_value_comp:
                # Standard comparison for non-ID fields
                return False
        # If all query conditions passed
        return True
    # Also ensure returned documents have string IDs for consistency
    def _stringify_ids(self, document: Dict) -> Dict:
        """Converts all ObjectId values in a document to strings."""
        if not document: return document
        stringified = {}
        for key, value in document.items():
             if isinstance(value, ObjectId):
                 stringified[key] = str(value)
             # Recursively stringify nested dictionaries (if any)
             elif isinstance(value, dict):
                 stringified[key] = self._stringify_ids(value)
             # Recursively stringify lists containing dictionaries (if any)
             elif isinstance(value, list):
                 stringified[key] = [self._stringify_ids(item) if isinstance(item, dict) else item for item in value]
             else:
                stringified[key] = value
        return stringified


class InsertOneResult:
    """Mimics MongoDB's InsertOneResult class."""
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id

class InsertManyResult:
    """Mimics MongoDB's InsertManyResult class."""
    def __init__(self, inserted_ids):
        self.inserted_ids = inserted_ids

class CollectionAccessor:
    """Provides MongoDB-like access to a collection in the JSON file database."""
    
    def __init__(self, db: 'JSONFileDatabase', collection_name: str):
        self.db = db
        self.collection_name = collection_name
    
    def insert_one(self, document: Dict) -> InsertOneResult:
        """Inserts a document into the collection."""
        inserted_id = self.db.insert_one(self.collection_name, document)
        return InsertOneResult(inserted_id)
    
    def insert_many(self, documents: List[Dict]) -> InsertManyResult:
        """Inserts multiple documents into the collection."""
        inserted_ids = self.db.insert_many(self.collection_name, documents)
        return InsertManyResult(inserted_ids)
    
    def find_one(self, query: Dict = None) -> Optional[Dict]:
        """Finds a single document matching the query."""
        return self.db.find_one(self.collection_name, query)
    
    def find(self, query: Dict = None, limit: int = 0) -> List[Dict]:
        """Finds all documents matching the query."""
        return self.db.find(self.collection_name, query, limit)
    
    def update_one(self, query: Dict, update: Dict) -> Dict:
        """Updates a single document matching the query."""
        modified_count = self.db.update_one(self.collection_name, query, update)
        return {"modified_count": modified_count}
    
    def count_documents(self, query: Dict = None) -> int:
        """Counts documents matching the query."""
        return self.db.count_documents(self.collection_name, query)
    
    def drop(self) -> None:
        """Drops the collection."""
        self.db.drop_collection(self.collection_name)


def connect_to_database(uri: str = config["mongodb_uri"], 
                       db_name: str = config["mongodb_db"],
                       json_db_path: str = config["json_db_path"]) -> BaseDatabase:
    """Connects to the database, trying MongoDB first and falling back to JSON files."""
    try:
        # Validate database name is a string
        if not isinstance(db_name, str):
            db_name = str(db_name) if db_name is not None else config["mongodb_db"]
        
        # Try to connect to MongoDB first
        logger.info(f"Attempting to connect to MongoDB at {uri}")
        
        # Create client with a shorter timeout for faster fallback
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        
        # Test the connection with a quick command
        client.admin.command('ismaster')
        
        # If we get here, connection was successful
        logger.info(f"Successfully connected to MongoDB at {uri}")
        return MongoDBDatabase(uri, db_name)
    except pymongo.errors.ConnectionFailure as e:
        logger.warning(f"Failed to connect to MongoDB: {e}. Falling back to JSON file database.")
        return JSONFileDatabase(json_db_path)
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.warning(f"MongoDB server selection timeout: {e}. Falling back to JSON file database.")
        return JSONFileDatabase(json_db_path)
    except Exception as e:
        logger.warning(f"Unexpected error connecting to MongoDB: {e}. Falling back to JSON file database.")
        return JSONFileDatabase(json_db_path)


def create_collections(db: BaseDatabase) -> None:
    """Creates the necessary collections if they don't already exist."""
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


def insert_initial_data(db: BaseDatabase) -> None:
    """Inserts initial data into the collections if it doesn't already exist."""
    # llms
    if db.count_documents("llms", {}) == 0:
        llm1 = {"llm_name": "Gemini 1.5 Flash", "llm_description": "Google's Gemini 1.5 Flash", "llm_provider": "Google"}
        llm2 = {"llm_name": "Llama 2", "llm_description": "Meta's Llama 2", "llm_provider": "Meta"}
        db.insert_many("llms", [llm1, llm2])
        logger.info("LLMs inserted.")
    else:
        logger.info("LLMs collection already has data, skipping insertion.")

    # llm_endpoints
    if db.count_documents("llm_endpoints", {}) == 0:
        # Make sure we have the LLMs first
        gemini_llm = db.find_one("llms", {"llm_name": "Gemini 1.5 Flash"})
        llama_llm = db.find_one("llms", {"llm_name": "Llama 2"})
        
        if gemini_llm and llama_llm:
            endpoint1 = {"llm_id": gemini_llm["_id"],
                        "endpoint_url": "https://ai.googleapis.com/v1beta1/models/gemini-1.5-flash:generateContent",
                        "endpoint_type": "REST API"}
            endpoint2 = {"llm_id": llama_llm["_id"],
                        "endpoint_url": "http://localhost:8080/completions", "endpoint_type": "REST API"}
            db.insert_many("llm_endpoints", [endpoint1, endpoint2])
            logger.info("LLM endpoints inserted.")
        else:
            logger.warning("Could not find required LLMs for endpoints, skipping endpoint insertion.")
    else:
        logger.info("LLM endpoints collection already has data, skipping insertion.")

    # model_deployments
    if db.count_documents("model_deployments", {}) == 0:
        # Make sure we have the LLMs first
        gemini_llm = db.find_one("llms", {"llm_name": "Gemini 1.5 Flash"})
        llama_llm = db.find_one("llms", {"llm_name": "Llama 2"})
        
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
            db.insert_many("model_deployments", [deployment1, deployment2])
            logger.info("Model deployments inserted.")
        else:
            logger.warning("Could not find required LLMs for deployments, skipping deployment insertion.")
    else:
        logger.info("Model deployments collection already has data, skipping insertion.")

    # Vector Store
    if db.count_documents("vector_store", {}) == 0:
        vector_store_data = {
            "store_name": "Chroma",
            "store_type": "Vector Database",
            "collection_name": "eval_docs",
            "connection_details": {
                "persist_directory": "./chroma_db",
                "client_settings": {},
                "embedding_model": "all-MiniLM-L6-v2"
            }
        }
        db.insert_one("vector_store", vector_store_data)
        logger.info("Vector store inserted")
    else:
        logger.info("Vector store collection already has data, skipping insertion.")

    # evaluation_harnesses
    if db.count_documents("evaluation_harnesses", {}) == 0:
        harness1 = {"harness_name": "RAG Faithfulness", "harness_type": "RAG Evaluation",
                    "description": "Evaluates the faithfulness of RAG responses"}
        harness2 = {"harness_name": "Context Relevance", "harness_type": "RAG Evaluation",
                    "description": "Evaluates the relevance of retrieved context"}
        db.insert_many("evaluation_harnesses", [harness1, harness2])
        logger.info("Evaluation harnesses inserted.")
    else:
        logger.info("Evaluation harnesses collection already has data, skipping insertion.")

    # test_suites
    if db.count_documents("test_suites", {}) == 0:
        # Check if we have evaluation harnesses first
        rag_harness = db.find_one("evaluation_harnesses", {"harness_name": "RAG Faithfulness"})
        context_harness = db.find_one("evaluation_harnesses", {"harness_name": "Context Relevance"})
        
        if rag_harness and context_harness:
            # Use a parameter for dataset_id instead of hardcoding
            # This allows the function to work with any dataset
            datasets = db.find("datasets", {}, limit=1)
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
                db.insert_many("test_suites", [suite1, suite2])
                logger.info("Test suites inserted.")
            else:
                logger.warning("No datasets found, skipping test suite insertion.")
        else:
            logger.warning("Could not find required evaluation harnesses, skipping test suite insertion.")
    else:
        logger.info("Test suites collection already has data, skipping insertion.")