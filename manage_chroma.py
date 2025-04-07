import chromadb
import argparse
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the path to the ChromaDB persistent storage
CHROMA_PATH = "./chroma_dev"

def delete_collection(collection_name: str):
    """
    Connects to the persistent ChromaDB client and deletes the specified collection.
    """
    if not os.path.exists(CHROMA_PATH):
        logging.error(f"ChromaDB path does not exist: {CHROMA_PATH}")
        logging.error("Please ensure the path is correct or that ChromaDB has been initialized.")
        return

    try:
        # Ensure the client is initialized correctly
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        logging.info(f"Connected to ChromaDB at: {CHROMA_PATH}")
    except Exception as e:
        logging.error(f"Failed to connect to ChromaDB client at {CHROMA_PATH}: {e}")
        return

    try:
        # Check if collection exists before attempting deletion
        # Listing collections is a good way to verify connection and existence
        collections = client.list_collections()
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            logging.info(f"Attempting to delete collection: '{collection_name}'...")
            client.delete_collection(name=collection_name)
            logging.info(f"Successfully deleted collection: '{collection_name}'")
        else:
            logging.warning(f"Collection '{collection_name}' not found. No action taken.")

    except Exception as e:
        logging.error(f"An error occurred while trying to delete collection '{collection_name}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete a specific collection from ChromaDB.")
    parser.add_argument(
        "collection_name",
        type=str,
        help="The name of the collection to delete."
    )

    args = parser.parse_args()

    delete_collection(args.collection_name)
