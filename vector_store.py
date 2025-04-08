"""
Vector Store module for handling vector embeddings in RAG evaluation.

Contains functions for:
- Storing document embeddings
- Retrieving similar documents
- Managing vector indexes
"""
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaError
from database import connect_to_mongodb

class VectorStore:
    """
    Handles vector storage and retrieval for RAG systems.
    """

    def __init__(self):
        """
        Initialize vector store with embedding model and ChromaDB local store.
        Configuration is fetched from MongoDB vector_store collection.
        """
        # Get configuration from MongoDB
        db = connect_to_mongodb()
        vector_store_config = db.vector_store.find_one()
        
        if not vector_store_config:
            raise RuntimeError("Vector store configuration not found in MongoDB")
            
        # Extract configuration
        connection_details = vector_store_config.get('connection_details', {})
        persist_directory = connection_details.get('persist_directory', './chroma_db')
        model_name = connection_details.get('embedding_model', 'all-MiniLM-L6-v2')
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB client with configuration from MongoDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None # Initialize self.collection to None
        
        # Get collection name from MongoDB config
        collection_name = vector_store_config.get('collection_name', 'eval_docs')

        # Atomic collection initialization with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.collection = self.client.get_collection(collection_name)
                logging.info(f"Successfully retrieved existing {collection_name} collection.")
                break # Exit loop if collection is retrieved successfully
            except Exception as e:
                # Check if the error indicates the collection doesn't exist
                # Different Chroma versions might raise different errors (NotFoundError, ValueError, etc.)
                # We check the specific message content for robustness.
                error_message = str(e).lower()
                # Use more specific check for ChromaDB's typical message
                is_not_found_error = f"collection {collection_name} does not exist" in error_message

                if is_not_found_error:
                    logging.warning(f"Collection '{collection_name}' not found (Error type: {type(e).__name__}, message: {e}). Attempting to create (Attempt {attempt+1}/{max_retries})...")
                    if attempt == max_retries - 1:
                        logging.error("Failed to create collection after multiple retries.")
                        raise # Re-raise the original error if all creation attempts fail
                    try:
                        self.collection = self.client.create_collection(
                            collection_name,
                            metadata={"hnsw:space": "cosine"},
                            embedding_function=None # Assuming SentenceTransformer handles embeddings separately
                        )
                        logging.info(f"Successfully created new {collection_name} collection.")
                        break # Exit loop after successful creation
                    except Exception as create_error:
                        logging.warning(f"Collection creation attempt {attempt+1} failed: {create_error}")
                        # If creation fails, loop will continue to retry getting/creating
                else:
                    # Handle other unexpected errors during get_collection
                    logging.error(f"Unexpected error getting collection (Attempt {attempt+1}/{max_retries}, Error type: {type(e).__name__}): {e}")
                    if attempt == max_retries - 1:
                        logging.error("Failed to get collection due to unexpected errors after multiple retries.")
                        raise # Re-raise the last unexpected error
                    # Consider adding a small delay before retrying unexpected errors
                    # import time
                    # time.sleep(1)

        # Final check after the loop
        if self.collection is None:
            raise RuntimeError(f"Failed to initialize collection '{collection_name}' after all retries.")

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector store and generate embeddings.
        """
        try:
            embeddings = self.model.encode(documents)
            ids = [str(i) for i in range(len(documents))]
            self.collection.add(
                embeddings=[embedding.tolist() for embedding in embeddings],
                documents=documents,
                ids=ids
            )
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            raise

    def get_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve most similar documents to a query.
        """
        try:
            query_embedding = self.model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )

            if not results['documents'] or not results['documents'][0]:
                logging.warning(f"No similar documents found for query: {query}")
                return []

            return [
                {
                    "text": doc,
                    "score": score
                }
                for doc, score in zip(results['documents'][0], results['distances'][0])
            ]
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []

    def cleanup_collection(self) -> None:
        """
        Delete all documents from the collection by removing and recreating the collection.
        """
        try:
            if self.collection is not None:
                # Get collection name from MongoDB config
                db = connect_to_mongodb()
                vector_store_config = db.vector_store.find_one()
                collection_name = vector_store_config.get('collection_name', 'eval_docs')
                
                # Delete the entire collection
                self.client.delete_collection(collection_name)
                # Recreate the collection
                self.collection = self.client.create_collection(
                    collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None
                )
                logging.info(f"Successfully cleaned up {collection_name} collection.")
        except Exception as e:
            logging.error(f"Error cleaning up collection: {e}")
            raise
