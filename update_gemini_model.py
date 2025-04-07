#!/usr/bin/env python3
"""
Script to update the Gemini model name in the MongoDB database from 'Gemini Pro' to 'Gemini 1.5 Flash'.
"""

import pymongo
import logging
from database import connect_to_mongodb, config

# Set up logging
logging.basicConfig(filename=config["log_file"], level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_gemini_model():
    """Updates the Gemini model name from 'Gemini Pro' to 'Gemini 1.5 Flash' in the llms collection."""
    try:
        # Connect to MongoDB
        db = connect_to_mongodb()
        
        # Find the Gemini Pro entry
        gemini_pro = db.llms.find_one({"llm_name": "Gemini Pro"})
        
        if gemini_pro:
            # Update the entry
            result = db.llms.update_one(
                {"_id": gemini_pro["_id"]},
                {"$set": {
                    "llm_name": "Gemini 1.5 Flash",
                    "llm_description": "Google's Gemini 1.5 Flash"
                }}
            )
            
            if result.modified_count > 0:
                logger.info("Successfully updated Gemini model from 'Gemini Pro' to 'Gemini 1.5 Flash'")
                print("Successfully updated Gemini model from 'Gemini Pro' to 'Gemini 1.5 Flash'")
                
                # Update related entries in llm_endpoints collection
                endpoints_result = db.llm_endpoints.update_many(
                    {"llm_id": gemini_pro["_id"]},
                    {"$set": {"endpoint_url": "https://ai.googleapis.com/v1beta1/models/gemini-1.5-flash:generateContent"}}
                )
                
                if endpoints_result.modified_count > 0:
                    logger.info(f"Updated {endpoints_result.modified_count} endpoint(s) for the Gemini model")
                    print(f"Updated {endpoints_result.modified_count} endpoint(s) for the Gemini model")
            else:
                logger.info("No update was needed for the Gemini model")
                print("No update was needed for the Gemini model")
        else:
            logger.warning("Could not find 'Gemini Pro' entry in the llms collection")
            print("Could not find 'Gemini Pro' entry in the llms collection")
            
    except Exception as e:
        logger.error(f"Error updating Gemini model: {e}")
        print(f"Error updating Gemini model: {e}")

if __name__ == "__main__":
    update_gemini_model()