#!/usr/bin/env python3
"""
Main entry point for the RAG Evaluation Framework.

This script demonstrates how to use the framework to:
1. Connect to MongoDB
2. Set up necessary collections
3. Load data from CSV
4. Generate synthetic QA pairs
5. Run evaluation harnesses
"""

import os
import logging
import argparse
from database import connect_to_mongodb
from database_adapter import BaseDatabase  # Import the BaseDatabase type for type hints
from vector_store import VectorStore
from llm_handlers import get_llm_handler
from evaluation import run_evaluation_harness, generate_rag_response, generate_metrics_report
from utils import load_data_from_csv

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """Set up MongoDB database connection"""
    logger.info("Setting up database connection...")
    db = connect_to_mongodb()
    return db

def store_dataset(db, csv_file_path):
    """Store dataset from CSV file and add to vector store"""
    logger.info(f"Loading data from {csv_file_path}...")
    try:
        # Load chunks from CSV
        chunks = load_data_from_csv(csv_file_path)
        
        if not chunks or len(chunks) == 0:
            logger.error("No data found in CSV file or invalid format")
            raise ValueError("CSV file contains no valid data or is missing 'chunk_content' column")
        
        # Store dataset in MongoDB
        dataset = {
            "dataset_name": os.path.basename(csv_file_path),
            "dataset_description": "Dataset loaded from CSV",
            "chunks": [{"chunk_content": chunk} for chunk in chunks]
        }
        
        dataset_id = db.datasets.insert_one(dataset).inserted_id
        logger.info(f"Dataset stored with ID: {dataset_id}")
        
        # Add chunks to vector store
        logger.info("Adding chunks to vector store...")
        from vector_store import VectorStore
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        logger.info("Chunks successfully added to vector store")
        
        return dataset_id
    except Exception as e:
        logger.error(f"Error storing dataset: {e}")
        raise

def generate_synthetic_data(db, dataset_id, llm_name="Gemini 1.5 Flash", num_questions_per_chunk=3):
    """Generate synthetic QA pairs for dataset"""
    logger.info(f"Generating synthetic QA pairs using {llm_name}...")
    
    try:
        # Get dataset chunks
        dataset = db.datasets.find_one({"_id": dataset_id})
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
        
        # Get LLM handler
        llm_handler = get_llm_handler(db, llm_name)
        
        # Store prompt template if not exists
        template = db.prompt_templates.find_one({"template_type": "qa_generation"})
        if not template:
            template = {
                "template_type": "qa_generation",
                "template_content": "Generate {num_questions} question-answer pairs from the following text. Format each pair as 'Question: <question>\nAnswer: <answer>'\n\nText: {context}"
            }
            template_id = db.prompt_templates.insert_one(template).inserted_id
        else:
            template_id = template["_id"]
        
        # Generate QA pairs for each chunk
        synthetic_data = []
        for chunk in dataset["chunks"]:
            prompt = template["template_content"].format(
                context=chunk["chunk_content"],
                num_questions=num_questions_per_chunk
            )
            qa_text = llm_handler.generate_qa_pairs(prompt, chunk["chunk_content"], num_questions_per_chunk)
            
            # Parse QA pairs
            from utils import QAPairParser
            qa_pairs = QAPairParser.parse_qa_pairs(qa_text, num_questions_per_chunk)
            
            # Check if qa_pairs is not None before iterating
            if qa_pairs is not None:
                # Store QA pairs
                for question, answer in qa_pairs:
                    synthetic_data.append({
                        "dataset_id": dataset_id,
                        "chunk_id": chunk.get("_id"),
                        "question": question,
                        "answer": answer,
                        "llm_id": db.llms.find_one({"llm_name": llm_name})["_id"]
                    })
            else:
                logging.warning(f"Failed to parse QA pairs for chunk: {chunk.get('_id')}")
        
        # Insert synthetic data
        if synthetic_data:
            db.synthetic_data.insert_many(synthetic_data)
            logger.info(f"Generated {len(synthetic_data)} QA pairs")
        
        return synthetic_data
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise

def create_test_suite(db, dataset_id, synthetic_data):
    """Create a test suite from synthetic data"""
    logger.info("Creating test suite...")
    try:
        # Create test suite
        test_suite = {
            "test_suite_name": f"Test Suite for Dataset {dataset_id}",
            "test_suite_description": "Automatically generated test suite",
            "dataset_id": dataset_id,
            "tests": []
        }
        
        # Add tests from synthetic data
        for qa_pair in synthetic_data:
            test_suite["tests"].append({
                "question": qa_pair["question"],
                "expected_answer": qa_pair["answer"],
                "test_type": "qa"
            })
        
        # Insert test suite
        test_suite_id = db.test_suites.insert_one(test_suite).inserted_id
        logger.info(f"Test suite created with ID: {test_suite_id}")
        return test_suite_id
    except Exception as e:
        logger.error(f"Error creating test suite: {e}")
        raise

def run_evaluation(db, test_suite_id, llm_name="Gemini 1.5 Flash"):
    """Run evaluation harness"""
    logger.info(f"Running evaluation with {llm_name}...")
    try:
        # Run evaluation harness
        results = run_evaluation_harness(db, test_suite_id, llm_name)
        
        # Check if results is a dictionary (expected format) or a list (possible format when using JSON DB)
        if isinstance(results, list):
            logger.info(f"Evaluation completed with {len(results)} results")
            # Convert list results to the expected dictionary format
            return {
                "run_id": None,  # This might be None when using JSON DB
                "test_suite_id": test_suite_id,
                "llm_name": llm_name,
                "num_tests": len(results),
                "success_rate": sum(1 for r in results if r.get("success", False)) / len(results) if results else 0
            }
        else:
            # Standard dictionary format
            logger.info(f"Evaluation completed with {results.get('num_tests', 0)} results")
            return results
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise

def prompt_cleanup():
    """Prompt user about cleaning up ChromaDB after execution"""
    while True:
        response = input("\nDo you want to clean up the ChromaDB collection? (yes/no): ").lower().strip()
        if response in ['yes', 'no']:
            return response == 'yes'
        print("Please enter 'yes' or 'no'")

def main():
    """Main function to run the RAG evaluation framework"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--csv", help="Path to CSV file with document chunks", default="data.csv")
    parser.add_argument("--llm", help="LLM to use", default="Gemini 1.5 Flash")
    parser.add_argument("--questions", type=int, help="Number of questions per chunk", default=3)
    parser.add_argument('--no-cleanup-prompt', action='store_true', help='Skip cleanup prompt after execution')
    args = parser.parse_args()
    
    # Check for API key
    if args.llm == "Gemini 1.5 Flash" and not os.environ.get("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY environment variable not set")
        print("\nError: GEMINI_API_KEY environment variable not set.")
        print("Please set it using: export GEMINI_API_KEY=your_api_key")
        return
    
    try:
        # Setup database connection
        db = setup_database()
        
        # Check if CSV file exists and is valid
        if not args.csv:
            logger.error("No CSV file path provided")
            print("\nError: No CSV file path provided.")
            print("Please provide a valid CSV file with document chunks.")
            return
            
        # Clean up file path (remove any trailing spaces)
        csv_path = args.csv.strip()
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            print(f"\nError: CSV file not found: {csv_path}")
            print("Please provide a valid CSV file with document chunks.")
            return
        
        # Store dataset with cleaned path
        dataset_id = store_dataset(db, csv_path)
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(db, dataset_id, args.llm, args.questions)
        
        # Create test suite
        test_suite_id = create_test_suite(db, dataset_id, synthetic_data)
        
        # Run evaluation
        results = run_evaluation(db, test_suite_id, args.llm)
        
        # Check if results contain run_id
        if "run_id" not in results or results["run_id"] is None:
            logger.error("Evaluation failed: No run_id in results")
            print("\nError: Evaluation failed to generate results")
            return
            
        # Generate metrics report
        report_path = generate_metrics_report(db, results["run_id"], output_format="html")
        
        print("\nRAG Evaluation completed successfully!")
        print(f"Dataset ID: {dataset_id}")
        print(f"Test Suite ID: {test_suite_id}")
        print(f"Generated {len(synthetic_data)} QA pairs")
        print(f"Ran {results['num_tests']} evaluation tests")
        if report_path:
            print(f"Generated metrics report: {report_path}")
        
        # Prompt for cleanup if not disabled
        if not args.no_cleanup_prompt and prompt_cleanup():
            logger.info("Cleaning up ChromaDB collection...")
            vector_store = VectorStore()
            vector_store.cleanup_collection()
            logger.info("ChromaDB cleanup completed.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()