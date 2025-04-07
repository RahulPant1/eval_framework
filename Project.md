# RAG Evaluation System Function Summaries

## Core Database Functions
1. `connect_to_mongodb(uri, db_name)`
    - Establishes connection to MongoDB using provided URI and database name
    - Returns database object or raises ConnectionFailure exception

2. `create_collections(db)`
    - Creates necessary collections in MongoDB if they don't exist
    - Collections include: llms, llm_endpoints, model_deployments, datasets, etc.

3. `insert_initial_data(db)`
    - Populates collections with initial data like LLM definitions and endpoints
    - Creates sample data for evaluation harnesses and test suites

4. `get_collection(db, collection_name)`
    - Retrieves a specific collection from the database
    - Returns collection object for further operations

## Data Management Functions
5. `load_data_from_csv(csv_file_path)`
    - Loads document chunks from CSV file
    - Validates that CSV contains required 'chunk_content' column

6. `store_dataset_and_chunks(db, csv_file_path)`
    - Stores dataset metadata and document chunks from CSV file in MongoDB
    - Returns the ID of the inserted dataset

7. `store_prompt_template(db)`
    - Creates and stores QA generation prompt template in MongoDB
    - Returns template ID for later reference

8. `create_sample_data(db, csv_file_path, num_questions_per_chunk, llm_name)`
    - End-to-end pipeline to create dataset from CSV, generate synthetic QA pairs, and populate vector store
    - Returns dataset ID

9. `populate_vector_store(db, dataset_id)`
    - Embeds document chunks and stores them in ChromaDB vector database
    - Uses SentenceTransformer for generating embeddings

10. `retrieve_chunks(db, dataset_id, query, num_results=3)`
    - Retrieves relevant chunks from vector store based on query similarity
    - Returns specified number of most relevant chunks

11. `get_dataset_by_id(db, dataset_id)`
    - Retrieves dataset metadata and associated chunks by ID
    - Returns dataset object or None if not found

## LLM Handler Classes
12. `BaseLLMHandler` (Class)
    - Abstract base class defining interface for LLM operations
    - Methods for QA generation, modified questions, and RAG responses

13. `GeminiLLMHandler` (Class)
    - Implementation of BaseLLMHandler for Google's Gemini 1.5 Flash
    - Uses Google's generative AI API for text generation

14. `Llama2LLMHandler` (Class)
    - Implementation of BaseLLMHandler for Meta's Llama 2
    - Simplified implementation with placeholder functionality

15. `GPTLLMHandler` (Class)
    - Implementation of BaseLLMHandler for OpenAI's GPT models
    - Uses OpenAI API for text generation

16. `get_llm_handler(db, llm_name)`
    - Factory function to return appropriate LLM handler based on name
    - Falls back to Gemini 1.5 Flash if requested LLM not found

## Synthetic Data Generation
17. `generate_synthetic_data(db, dataset_id, template_id, num_questions_per_chunk, llm_name, generate_modified)`
    - Generates QA pairs for each chunk in dataset using specified LLM
    - Optionally generates modified versions of questions for robustness testing

18. `generate_modified_question(question, llm_name)`
    - Creates variations of a question while preserving meaning
    - Generates paraphrased, noisy, and restructured versions

19. `QAPairParser` (Class)
    - Static methods to parse QA pairs from LLM outputs
    - Handles multiple format patterns using regex

20. `format_prompt_with_context(prompt_template, context, question=None)`
    - Formats prompt template by inserting context and optional question
    - Returns formatted prompt ready for LLM consumption

## RAG Evaluation Functions
21. `generate_rag_response(question, db, dataset_id, llm_name, num_results)`
    - Retrieves relevant context chunks from vector store
    - Generates answer to question using retrieved context and specified LLM

22. `evaluate_response(rag_response, expected_answer)`
    - Basic evaluation of whether response contains expected information
    - Returns boolean result and explanation

23. `calculate_bleu(reference, candidate)`
    - Calculates BLEU score between reference and candidate texts
    - Handles tokenization and smoothing

24. `calculate_rouge(reference, candidate)`
    - Calculates ROUGE scores for precision, recall, and F1
    - Returns dictionary of ROUGE metrics

25. `calculate_meteor(reference, candidate)`
    - Calculates METEOR score for semantic similarity
    - Handles tokenization

26. `calculate_bertscore(reference, candidate)`
    - Calculates BERTScore for semantic similarity using BERT embeddings
    - Returns precision, recall, and F1 scores

27. `run_evaluation_harness(db, test_suite_id, llm_name)`
    - Main evaluation pipeline that runs tests for a given test suite
    - Processes original and modified questions, calculates metrics
    - Stores detailed results in MongoDB

28. `create_test_suite(db, dataset_id, name, description)`
    - Creates a new test suite associated with a dataset
    - Returns test suite ID for later reference

29. `add_test_case(db, test_suite_id, question, expected_answer, metadata=None)`
    - Adds a test case to an existing test suite
    - Optionally includes metadata about the test case

## Additional Utility Functions
30. `setup_logging(log_level)`
    - Configures logging with specified level
    - Sets up file and console handlers

31. `validate_config(config)`
    - Validates configuration parameters
    - Checks for required fields and correct data types

32. `cleanup_resources(db)`
    - Properly closes database connections
    - Cleans up any temporary files or resources

33. `format_evaluation_results(results)`
    - Formats evaluation results for display or export
    - Generates summary statistics and detailed breakdowns

34. `export_results_to_csv(results, output_path)`
    - Exports evaluation results to CSV file
    - Includes all metrics and test case details
