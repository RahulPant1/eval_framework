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

class VectorStore:
    """
    Handles vector storage and retrieval for RAG systems.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector store with embedding model and ChromaDB local store.
        """
        self.model = SentenceTransformer(model_name)

        # Initialize ChromaDB client with new configuration
        self.client = chromadb.PersistentClient(path="./chroma_dev")
        self.collection = None # Initialize self.collection to None

        # Atomic collection initialization with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.collection = self.client.get_collection("eval_docs")
                logging.info("Successfully retrieved existing eval_docs collection.")
                break # Exit loop if collection is retrieved successfully
            except Exception as e:
                # Check if the error indicates the collection doesn't exist
                # Different Chroma versions might raise different errors (NotFoundError, ValueError, etc.)
                # We check the specific message content for robustness.
                error_message = str(e).lower()
                # Use more specific check for ChromaDB's typical message
                is_not_found_error = f"collection eval_docs does not exist" in error_message

                if is_not_found_error:
                    logging.warning(f"Collection 'eval_docs' not found (Error type: {type(e).__name__}, message: {e}). Attempting to create (Attempt {attempt+1}/{max_retries})...")
                    if attempt == max_retries - 1:
                        logging.error("Failed to create collection after multiple retries.")
                        raise # Re-raise the original error if all creation attempts fail
                    try:
                        self.collection = self.client.create_collection(
                            "eval_docs",
                            metadata={"hnsw:space": "cosine"},
                            embedding_function=None # Assuming SentenceTransformer handles embeddings separately
                        )
                        logging.info("Successfully created new eval_docs collection.")
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
            raise RuntimeError("Failed to initialize collection 'eval_docs' after all retries.")

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

            return [
                {
                    "text": doc,
                    "score": score
                }
                for doc, score in zip(results['documents'][0], results['distances'][0])
            ]
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            raise
