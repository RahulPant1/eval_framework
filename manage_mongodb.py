import pymongo
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default MongoDB connection URI
MONGO_URI = "mongodb://localhost:27017/"

def delete_all_collections(db_name: str):
    """
    Connects to MongoDB and drops all collections within the specified database.
    """
    try:
        client = pymongo.MongoClient(MONGO_URI)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        logging.info(f"Successfully connected to MongoDB at: {MONGO_URI}")
    except pymongo.errors.ConnectionFailure as e:
        logging.error(f"Failed to connect to MongoDB at {MONGO_URI}: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during MongoDB connection: {e}")
        return

    try:
        # Check if the database exists
        db_list = client.list_database_names()
        if db_name not in db_list:
            logging.warning(f"Database '{db_name}' not found. No action taken.")
            client.close()
            return

        db = client[db_name]
        collections = db.list_collection_names()

        if not collections:
            logging.info(f"Database '{db_name}' contains no collections. No action needed.")
        else:
            logging.warning(f"Preparing to delete ALL collections from database: '{db_name}'")
            logging.warning(f"Collections to be deleted: {', '.join(collections)}")

            # Confirmation step
            confirm = input("Are you absolutely sure you want to delete all collections? Type 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                logging.info("Deletion cancelled by user.")
                client.close()
                return

            logging.info(f"Proceeding with deletion of collections in database '{db_name}'...")
            for collection_name in collections:
                try:
                    db.drop_collection(collection_name)
                    logging.info(f"Successfully dropped collection: '{collection_name}'")
                except Exception as e:
                    logging.error(f"Failed to drop collection '{collection_name}': {e}")

            logging.info(f"Finished deleting collections from database: '{db_name}'")

    except Exception as e:
        logging.error(f"An error occurred while operating on database '{db_name}': {e}")
    finally:
        client.close()
        logging.info("MongoDB connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete ALL collections within a specified MongoDB database.",
        epilog="WARNING: This operation is irreversible and will delete all data in the target database's collections."
    )
    parser.add_argument(
        "db_name",
        type=str,
        help="The name of the MongoDB database whose collections should be deleted."
    )

    args = parser.parse_args()

    delete_all_collections(args.db_name)
