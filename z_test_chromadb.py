import chromadb

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Get list of all collections
collections = client.list_collections()
print("\nAll persisted collections:")
for collection in collections:
    print(f"Collection name: {collection.name}, metadata: {collection.metadata}")