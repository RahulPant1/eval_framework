# RAG Evaluation Framework: Execution Flow Documentation

## Prerequisites
- Python 3.8+
- MongoDB Running on localhost (sudo systemctl start mongod)
- Gemini API key (export GEMINI_API_KEY="") for Gemini 1.5 Flash model
- Disk space for persistent ChromaDB storage
- JSON database files in `json_db/` directory

## Execution Flow

Step 1: Initialize the databse: python init_database.py

Step 2: Given Below:
### Help
python3 main.py --h
usage: main.py [-h] [--csv CSV] [--llm LLM] [--questions QUESTIONS] [--no-cleanup-prompt]

RAG Evaluation Framework

options:
  -h, --help            show this help message and exit
  --csv CSV             Path to CSV file with document chunks
  --llm LLM             LLM to use
  --questions QUESTIONS
                        Number of questions per chunk
  --no-cleanup-prompt   Skip cleanup prompt after execution


This document explains the execution flow when running the command:
```bash
python3 main.py --csv /home/rahul/eval_framework/sample_data.csv
```

## Overview

The RAG Evaluation Framework is designed to evaluate Retrieval-Augmented Generation systems by:
1. Loading document chunks from a CSV file
2. Adding document chunks to a persistent ChromaDB vector store
3. Generating synthetic question-answer pairs using an LLM
4. Creating a test suite from these pairs
5. Running evaluations using a RAG pipeline
6. Calculating performance metrics
7. Generating comprehensive evaluation reports

## Execution Flow

### 1. Entry Point: `main.py`

The execution begins in the `main()` function of `main.py`, which:

- Parses command-line arguments using `argparse`
- Validates the presence of required API keys
- Orchestrates the entire evaluation process

### 2. Database Setup

The database setup is now split into two parts:

#### 2.1 Database Initialization (One-time Setup)

Before running the main evaluation, initialize the database using:

```bash
python init_database.py
```

or for JSON database:

```bash
python init_jsondb.py
```

This script:
- Creates all required MongoDB collections or JSON files
- Populates initial reference data
- Sets up the following collections/files:
  - llms
  - llm_endpoints
  - model_deployments
  - datasets
  - prompt_templates
  - synthetic_data
  - evaluation_harnesses
  - test_suites
  - evaluation_runs
  - test_results
  - metrics
  - metrics_definitions
  - vector_store

#### 2.2 Database Connection: `setup_database()`

```python
db = setup_database()
```

This function:
- Establishes a connection to MongoDB or loads JSON files
- Returns the database connection object for use in subsequent operations

### 3. Dataset Storage and Vector Indexing: `store_dataset()`

```python
dataset_id = store_dataset(db, csv_path)
```

This function:
- Calls `load_data_from_csv()` from `utils.py` to read document chunks
- Creates a dataset document with metadata and chunks
- Inserts the dataset into MongoDB and returns its ID
- **Initializes the persistent ChromaDB vector store**
- **Adds all document chunks to the vector store in a single operation**

This optimization ensures that document chunks are indexed only once during initial data loading, rather than repeatedly during each evaluation question.

### 4. Synthetic Data Generation: `generate_synthetic_data()`

```python
synthetic_data = generate_synthetic_data(db, dataset_id, args.llm, args.questions)
```

This function:
- Retrieves the dataset chunks from MongoDB
- Gets the appropriate LLM handler using `get_llm_handler()` from `llm_handlers.py`
- For each chunk:
  - Formats a prompt using a template from the database
  - Calls `generate_qa_pairs()` on the LLM handler to generate QA pairs
  - Uses `QAPairParser.parse_qa_pairs()` from `utils.py` to extract structured QA pairs
  - Stores the QA pairs in the synthetic_data collection

### 5. Test Suite Creation: `create_test_suite()`

```python
test_suite_id = create_test_suite(db, dataset_id, synthetic_data)
```

This function:
- Creates a test suite document with metadata
- Adds tests from the synthetic QA pairs
- Inserts the test suite into MongoDB and returns its ID

### 6. Evaluation Execution: `run_evaluation()`

```python
results = run_evaluation(db, test_suite_id, args.llm)
```

This function:
- Calls `run_evaluation_harness()` from `evaluation.py`, which:
  - Retrieves the test suite from MongoDB
  - For each test question:
    - **Reuses the existing vector store** (no need to reinitialize for each question)
    - Calls `generate_rag_response()` which:
      - Retrieves similar documents using vector similarity search
      - Generates an answer using the LLM with the retrieved context
    - Evaluates the response using metrics like BLEU, ROUGE, etc.
    - Stores the results in the test_results collection

### 7. Metrics Reporting: `generate_metrics_report()`

```python
report_path = generate_metrics_report(db, results["run_id"], output_format="html")
```

This new function:
- Retrieves the evaluation run results from MongoDB
- Calculates aggregate metrics across all test questions
- Generates detailed reports in multiple formats (HTML, Markdown, etc.)
- Saves reports to the `reports/` directory with timestamps
- Returns the path to the generated report

## Key Components

### Vector Store (vector_store.py)

The `VectorStore` class:
- Uses SentenceTransformer to generate embeddings
- **Stores embeddings in persistent ChromaDB**
  - Configuration managed through MongoDB vector_store collection or JSON files
  - Persistent storage in configurable directory (default: `./chroma_db/`)
  - Collection name and other settings fetched from database or JSON
- Provides similarity search functionality with k-nearest neighbors (kNN)
- **Implements robust error handling and retries for ChromaDB operations**
- Supports dynamic configuration updates through MongoDB or JSON

### LLM Handlers (llm_handlers.py)

The framework supports multiple LLMs through handler classes:
- `GeminiLLMHandler`: Interfaces with Google's Gemini 1.5 Flash model
- Each handler implements methods for:
  - Generating QA pairs
  - Generating modified questions
  - Generating RAG responses

### Evaluation Metrics (evaluation.py)

The framework calculates various metrics:
- BLEU: Measures n-gram precision
- ROUGE: Measures recall of n-grams
- Semantic similarity: Uses embedding-based similarity
- Custom RAG metrics like faithfulness and relevance

### Report Generator (report_generator.py)

The new report generator module:
- Calculates aggregate metrics across all test questions
- Generates detailed per-question metrics
- Supports multiple output formats (HTML, Markdown, JSON, TXT)
- Creates visually appealing reports with tables and charts
- Includes modified question performance analysis

## Logging

The framework logs detailed information to `rag_evaluation.log`, including:
- Database operations
- LLM prompts and responses
- Vector store operations
- Evaluation results

## Conclusion

The RAG Evaluation Framework provides an end-to-end pipeline for evaluating RAG systems. It handles data loading, synthetic QA generation, persistent vector storage, retrieval, response generation, evaluation metrics calculation, and comprehensive reporting in a modular and extensible way.

==============================================================================================================================================================================================================

## Overall Codebase Structure

The framework is modular, separating concerns into distinct Python files:

1.  **`main.py`**: The main entry point and orchestrator of the evaluation pipeline. It handles argument parsing, setup, and calls functions from other modules in sequence.
2.  **`database_adapter.py`**: Provides a crucial abstraction layer for database operations. It defines a `BaseDatabase` interface and implements it for both MongoDB (`MongoDBDatabase`) and local JSON files (`JSONFileDatabase`), allowing the framework to function even without a MongoDB instance. Includes helper classes (`CollectionAccessor`, `InsertOneResult`, `InsertManyResult`) to mimic MongoDB's API style.
3.  **`database.py`**: Acts as a backward-compatibility layer. Its `connect_to_mongodb` function now uses the `database_adapter`. Other functions are either wrappers or placeholders. Primarily used by older parts of the code or for potential direct MongoDB interaction if needed outside the adapter pattern.
4.  **`vector_store.py`**: Manages interactions with the vector database (ChromaDB). Handles initialization (reading config from the database adapter), adding documents (encoding + storing), retrieving similar documents, and cleanup.
5.  **`llm_handlers.py`**: Contains handlers for interacting with different Large Language Models (LLMs). It defines a base class (`BaseLLMHandler`) and specific implementations (e.g., `GeminiLLMHandler`). Includes a factory function (`get_llm_handler`) to retrieve the correct handler. Responsible for generating synthetic QA data and generating answers during RAG.
6.  **`utils.py`**: Holds shared utility functions, notably `load_data_from_csv` for data ingestion and the `QAPairParser` class for robustly parsing QA pairs from potentially messy LLM text output.
7.  **`evaluation.py`**: Contains the core logic for running the evaluation harness. It includes functions to generate a RAG response for a single question, calculate various evaluation metrics (BLEU, ROUGE, METEOR), and the main `run_evaluation_harness` function which orchestrates the testing loop and stores results in the database via the adapter.
8.  **`report_generator.py`**: Responsible for fetching evaluation run data from the database (via adapter) and generating comprehensive reports in different formats (HTML, Markdown, JSON, TXT). Includes functions for calculating aggregate metrics.
9.  **`init_database.py`**: A utility script to perform one-time initialization of the *MongoDB* database (creating collections, inserting initial seed data) using the database adapter.
10. **`init_jsondb.py`**: A utility script to perform one-time initialization of the *JSON file* database backend (creating the directory and empty JSON files for each collection, inserting initial seed data). *Note: This script directly manipulates files for setup, whereas `JSONFileDatabase` handles runtime interaction.*

### Developer-Friendly Content:

*   **Modularity:** The separation of concerns makes the code easier to navigate, maintain, and extend. Need to change the vector store? Look in `vector_store.py`. Add a new LLM? Modify `llm_handlers.py`.
*   **Database Abstraction:** The `database_adapter.py` is key. It allows developers to run the framework with or without MongoDB setup, significantly lowering the barrier to entry for testing or local development. The adapter ensures that modules like `evaluation.py` or `vector_store.py` don't need to know the underlying database type; they interact through the consistent `BaseDatabase` interface or the `CollectionAccessor`.
*   **Configuration:** Configuration (like DB URIs, paths, LLM names) is mostly centralized or passed via arguments/environment variables, although moving constants from `database_adapter.py` and other files into a dedicated `config.py` would be a good improvement.
*   **Error Handling:** Basic error handling (e.g., `try...except` blocks, logging warnings/errors) is present, particularly around external dependencies like database connections, API calls, and file operations.
*   **Logging:** Standard Python `logging` is used throughout, providing insights into the execution flow and potential issues.

---

## Detailed Function Explanations

Here's a breakdown of each function/method within the Python files:

### `database_adapter.py`

*   `BaseDatabase.__init__(self)`: Initializes the base class, setting up an empty `collections` dictionary (though not strictly used by subclasses).
*   `BaseDatabase.list_collection_names(self)` / `create_collection(self, ...)` / `get_collection(self, ...)` / `insert_one(self, ...)` / `insert_many(self, ...)` / `find_one(self, ...)` / `find(self, ...)` / `update_one(self, ...)` / `count_documents(self, ...)` / `drop_collection(self, ...)`: Abstract methods defining the required database operations. Subclasses *must* implement these.
*   `MongoDBDatabase.__init__(self, uri, db_name)`: Constructor for the MongoDB adapter. Initializes the `pymongo.MongoClient` and selects the database. Logs connection info.
*   `MongoDBDatabase.__getattr__(self, name)`: Allows accessing MongoDB collections using dot notation (e.g., `db.datasets`) by dynamically returning a `CollectionAccessor`. Creates the collection if it doesn't exist on first access.
*   `MongoDBDatabase.list_collection_names(self)`: Returns a list of collection names directly from the MongoDB database.
*   `MongoDBDatabase.create_collection(self, collection_name)`: Creates a collection in MongoDB if it doesn't already exist.
*   `MongoDBDatabase.get_collection(self, collection_name)`: Returns the underlying `pymongo` collection object.
*   `MongoDBDatabase.insert_one(self, collection_name, document)`: Inserts a single document into the specified MongoDB collection. Returns the string representation of the inserted `ObjectId`.
*   `MongoDBDatabase.insert_many(self, collection_name, documents)`: Inserts a list of documents. Returns a list of string representations of the inserted `ObjectId`s.
*   `MongoDBDatabase.find_one(self, collection_name, query)`: Finds a single document matching the query. Handles string-to-ObjectId conversion for the `_id` field in the query. Returns the document dictionary or `None`.
*   `MongoDBDatabase.find(self, collection_name, query, limit)`: Finds multiple documents matching the query, optionally limiting the results. Returns a list of document dictionaries.
*   `MongoDBDatabase.update_one(self, collection_name, query, update)`: Updates a single document matching the query using MongoDB's update operators (like `$set`). Returns the number of documents modified (0 or 1).
*   `MongoDBDatabase.count_documents(self, collection_name, query)`: Counts documents matching the query in the specified collection.
*   `MongoDBDatabase.drop_collection(self, collection_name)`: Drops (deletes) the specified collection from MongoDB.
*   `JSONFileDatabase.__init__(self, db_path)`: Constructor for the JSON file adapter. Creates the base directory if needed, initializes an in-memory `collections` dictionary, and pre-loads any existing `.json` files from the `db_path`.
*   `JSONFileDatabase.__getattr__(self, name)`: Allows accessing JSON collections using dot notation (e.g., `db.datasets`) by dynamically returning a `CollectionAccessor`. Creates the collection (in memory and as a file) if it doesn't exist on first access.
*   `JSONFileDatabase._get_collection_path(self, collection_name)`: Helper to get the full file path for a given collection name.
*   `JSONFileDatabase._load_collection(self, collection_name)`: Helper to load data for a specific collection from its JSON file into memory. Handles file-not-found and JSON decoding errors.
*   `JSONFileDatabase._save_collection(self, collection_name)`: Helper to save the current in-memory state of a collection back to its JSON file. Uses `json.dump` with indentation and `default=str` to handle non-serializable types like `ObjectId` and `datetime`.
*   `JSONFileDatabase._ensure_collection(self, collection_name)`: Helper to make sure a collection exists in the in-memory dictionary, loading it from file if necessary.
*   `JSONFileDatabase._generate_id(self)`: Helper to generate a new unique ID (as a stringified `ObjectId`) for documents that don't have one.
*   `JSONFileDatabase.list_collection_names(self)`: Returns a list of keys from the in-memory `collections` dictionary.
*   `JSONFileDatabase.create_collection(self, collection_name)`: Creates an empty list for the collection in the in-memory dictionary and saves an empty list to the corresponding JSON file if it doesn't exist.
*   `JSONFileDatabase.get_collection(self, collection_name)`: Returns the list representing the collection from the in-memory dictionary (ensuring it's loaded first).
*   `JSONFileDatabase.insert_one(self, collection_name, document)`: Adds a document to the in-memory list for the collection. Assigns a new string `_id` if one isn't present. Saves the updated collection list back to the JSON file. Returns the string `_id`.
*   `JSONFileDatabase.insert_many(self, collection_name, documents)`: Adds multiple documents to the in-memory list. Assigns string `_id`s if needed. Saves the updated collection list to the JSON file. Returns a list of string `_id`s.
*   `JSONFileDatabase._process_query(self, query)`: Helper function to preprocess query dictionaries. It converts any `ObjectId` values (including those within `$in` lists) to their string representations to ensure consistent comparisons.
*   `JSONFileDatabase._stringify_ids(self, document)`: Helper function to recursively traverse a document (or list) and convert any `ObjectId` values to their string representations before returning results. Ensures consistent output format.
*   `JSONFileDatabase.find_one(self, collection_name, query)`: Iterates through the in-memory list for the collection. Preprocesses the query using `_process_query`. Uses `_matches_query` to find the first matching document. Returns a copy of the document with IDs stringified using `_stringify_ids`, or `None`.
*   `JSONFileDatabase.find(self, collection_name, query, limit)`: Iterates through the in-memory list. Preprocesses the query using `_process_query`. Uses `_matches_query` to find all matching documents. Returns a list of document copies with IDs stringified using `_stringify_ids`, respecting the optional `limit`.
*   `JSONFileDatabase.update_one(self, collection_name, query, update)`: Finds the first document matching the query using `_matches_query`. Applies the update (handles `$set` operator, otherwise replaces the document). Saves the collection. Returns 1 if updated, 0 otherwise.
*   `JSONFileDatabase.count_documents(self, collection_name, query)`: Counts documents in the in-memory list that match the query using `_matches_query`.
*   `JSONFileDatabase.drop_collection(self, collection_name)`: Removes the collection from the in-memory dictionary and deletes the corresponding JSON file.
*   `JSONFileDatabase._matches_query(self, document, query)`: Core query matching logic for the JSON adapter. Compares document fields against query values. Explicitly handles ID field comparisons (`_id`, `dataset_id`, etc.) by comparing string representations. Handles the `$in` operator.
*   `InsertOneResult.__init__(self, inserted_id)`: Simple class to hold the result of an `insert_one` operation, mimicking MongoDB's result object.
*   `InsertManyResult.__init__(self, inserted_ids)`: Simple class to hold the result of an `insert_many` operation.
*   `CollectionAccessor.__init__(self, db, collection_name)`: Constructor for the accessor, holding references to the database adapter instance and the collection name.
*   `CollectionAccessor.insert_one(...)` / `insert_many(...)` / `find_one(...)` / `find(...)` / `update_one(...)` / `count_documents(...)` / `drop()`: Methods that provide the familiar MongoDB-like API (e.g., `db.my_collection.find_one(...)`). They simply call the corresponding methods on the underlying database adapter instance (`self.db`), passing the `self.collection_name`.
*   `connect_to_database(uri, db_name, json_db_path)`: The primary factory function. Attempts to connect to MongoDB with a timeout. If successful, returns `MongoDBDatabase`. On failure (connection error, timeout), logs a warning and returns `JSONFileDatabase`. This is the recommended way to get a database connection object.
*   `create_collections(db)`: Utility function (can work with either adapter type) that iterates through a predefined list of collection names and calls `db.create_collection()` for each one if it doesn't exist.
*   `insert_initial_data(db)`: Utility function (can work with either adapter type) that populates collections with initial seed data (LLMs, endpoints, vector store config, etc.) *only if* the respective collections are currently empty (`db.count_documents(...) == 0`).

### `database.py`

*   `connect_to_mongodb(uri, db_name)`: Now simply acts as a pass-through function, calling `database_adapter.connect_to_database`. Maintained for backward compatibility.

### `evaluation.py`

*   NLTK Downloads: Ensures necessary `nltk` data packages (punkt, wordnet, omw-1.4, taggers, stopwords) required for tokenization and METEOR scoring are downloaded.
*   `generate_rag_response(question, db, dataset_id, llm_name, num_results)`: Orchestrates the RAG pipeline for a single query.
    1.  Finds the dataset metadata using `db.datasets.find_one`.
    2.  Initializes the `VectorStore`.
    3.  Retrieves `num_results` similar document chunks using `vector_store.get_similar_documents`.
    4.  Handles the case where no documents are found.
    5.  Concatenates retrieved document texts into a single `context`.
    6.  Gets the appropriate LLM handler using `llm_handlers.get_llm_handler`.
    7.  Calls the handler's `generate_answers` method with the question and context.
    8.  Returns the generated answer text or an error message.
*   `evaluate_response(rag_response, expected_answer)`: Performs a basic evaluation based on token overlap.
    1.  Tokenizes both the RAG response and expected answer (lowercase).
    2.  Calculates the ratio of common tokens to expected tokens.
    3.  Returns `(True, explanation)` if overlap >= 70%, `(False, explanation)` otherwise.
*   `calculate_bleu(reference, candidate)`: Calculates the sentence BLEU score.
    1.  Tokenizes reference (expected answer) and candidate (RAG response).
    2.  Uses `nltk.translate.bleu_score.sentence_bleu` with smoothing (method1).
    3.  Returns the BLEU score (float 0-1) or 0.0 on error.
*   `calculate_rouge(reference, candidate)`: Calculates ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    1.  Uses the `rouge.Rouge` library.
    2.  Calls `rouge.get_scores`.
    3.  Extracts precision (p), recall (r), and f1-score (f) for each ROUGE type.
    4.  Returns a dictionary containing these scores or a dictionary of zeros on error.
*   `calculate_meteor(reference, candidate)`: Calculates the METEOR score.
    1.  Tokenizes reference and candidate texts.
    2.  Uses `nltk.translate.meteor_score.meteor_score`.
    3.  Returns the METEOR score (float 0-1) or 0.0 on error.
*   `run_evaluation_harness(db, test_suite_id, llm_name)`: Executes the main evaluation loop for a given test suite.
    1.  Finds the `test_suite` document using `db.test_suites.find_one` (handles string/ObjectId).
    2.  Finds all associated `synthetic_data` documents using `db.synthetic_data.find`.
    3.  Iterates through each `data_point` (QA pair) from the synthetic data.
    4.  For each data point:
        *   Calls `generate_rag_response` to get the system's answer.
        *   Calls `evaluate_response` for a basic success/failure judgment.
        *   Calls `calculate_bleu`, `calculate_rouge`, `calculate_meteor`.
        *   Compiles these results into a `result` dictionary.
        *   *(Includes commented-out logic for handling modified questions).*
        *   Appends the `result` to the `results` list.
    5.  Creates an `evaluation_run` document containing metadata and the full `results` list.
    6.  Inserts the `evaluation_run` into the database using `db.evaluation_runs.insert_one`.
    7.  Iterates through the `results` again to insert individual `test_result` documents into `db.test_results` and corresponding `metrics_entry` documents into `db.metrics`.
    8.  Inserts default `metrics_definitions` into the database if the collection is empty.
    9.  Returns a summary dictionary containing run ID, counts, success rate, and average metrics, or an error dictionary if exceptions occurred.
*   `generate_metrics_report(db, evaluation_run_id, output_format, output_dir)`: A simple wrapper function that calls `report_generator.generate_metrics_report` to handle the actual report creation process.

### `init_database.py`

*   `initialize_database()`: Orchestrates the MongoDB initialization.
    1.  Calls `database.connect_to_mongodb` (which uses the adapter, potentially falling back to JSON if Mongo fails, although this script is *intended* for Mongo setup).
    2.  Calls `database_adapter.create_collections`.
    3.  Calls `database_adapter.insert_initial_data`.
    4.  Logs success or errors.

### `init_jsondb.py`

*   `ensure_json_db_dir()`: Creates the `json_db` directory if it doesn't exist.
*   `create_json_collections()`: Iterates through a predefined list of collection names and creates empty `[ ]` JSON files for each if they don't exist.
*   `read_json_file(collection_name)`: Reads and parses JSON data from a specific collection file.
*   `write_json_file(collection_name, data)`: Writes Python list/dict data to a specific collection JSON file with indentation.
*   `insert_initial_data()`: Reads current data from `llms.json` and `llm_endpoints.json`. If empty, writes default initial data (LLM names, basic endpoints) to these files. *Note: This only seeds a couple of collections compared to the adapter's `insert_initial_data`.*
*   `initialize_json_database()`: Orchestrates the JSON database file structure setup. Calls `ensure_json_db_dir`, `create_json_collections`, and `insert_initial_data`.

### `llm_handlers.py`

*   `BaseLLMHandler.generate_qa_pairs(self, ...)`: Abstract method signature.
*   `GeminiLLMHandler.__init__(self)`: Configures the `google.generativeai` library with the API key found in the environment variable `GEMINI_API_KEY`. Initializes the `GenerativeModel`.
*   `GeminiLLMHandler.generate_qa_pairs(self, prompt_template, context, num_questions)`:
    1.  Formats the prompt using the provided template, context, and number of questions.
    2.  Calls `self.model.generate_content(prompt)` to interact with the Gemini API.
    3.  Parses the response structure to extract the generated text content.
    4.  Handles potential errors during API call or response parsing.
    5.  Returns the raw text string containing the generated QA pairs (or an error message).
*   `GeminiLLMHandler.generate_answers(self, questions, context)`:
    1.  Iterates through the list of `questions`.
    2.  For each question, formats a prompt asking for an answer based *only* on the provided `context`.
    3.  Calls `self.model.generate_content(prompt)`.
    4.  Extracts the generated text answer from the response.
    5.  Appends the answer (or an error message) to a list.
    6.  Returns the list of generated answers.
*   `Llama2LLMHandler.__init__(self, chunk_content_size)`: Placeholder constructor.
*   `Llama2LLMHandler.generate_qa_pairs(self, ...)`: Placeholder implementation returning templated QA pairs.
*   `get_llm_handler(db, llm_name)`: Factory function.
    1.  Checks if the requested `llm_name` exists in the `db.llms` collection.
    2.  Looks up the name in a hardcoded `handlers` dictionary.
    3.  If found and valid, returns an instance of the corresponding handler class.
    4.  If not found or not supported, logs a warning and defaults to returning a `GeminiLLMHandler` instance.

### `main.py`

*   Argument Parsing: Uses `argparse` to define and parse command-line arguments (`--csv`, `--llm`, `--questions`, `--no-cleanup-prompt`).
*   `setup_database()`: Calls `database.connect_to_mongodb` to get a database adapter instance (could be Mongo or JSON).
*   `store_dataset(db, csv_file_path)`:
    1.  Loads data chunks from the specified CSV using `utils.load_data_from_csv`.
    2.  Creates a `dataset` dictionary containing metadata and the loaded chunks.
    3.  Inserts the dataset document into `db.datasets`.
    4.  Initializes a `VectorStore` instance.
    5.  Calls `vector_store.add_documents` to vectorize and store the chunks.
    6.  Returns the dataset ID.
*   `generate_synthetic_data(db, dataset_id, llm_name, num_questions_per_chunk)`:
    1.  Retrieves the dataset document using `db.datasets.find_one`.
    2.  Gets the appropriate LLM handler using `llm_handlers.get_llm_handler`.
    3.  Finds or creates a QA generation prompt template in `db.prompt_templates`.
    4.  Iterates through each `chunk` in the dataset.
    5.  Formats the prompt and calls `llm_handler.generate_qa_pairs`.
    6.  Parses the returned text into QA pairs using `utils.QAPairParser.parse_qa_pairs`.
    7.  Formats the parsed pairs into documents suitable for the `synthetic_data` collection (linking `dataset_id`, `llm_id`).
    8.  Inserts the list of synthetic data documents into `db.synthetic_data` using `insert_many`.
    9.  Returns the list of generated synthetic data documents.
*   `create_test_suite(db, dataset_id, synthetic_data)`:
    1.  Creates a `test_suite` dictionary containing metadata and a `tests` list populated from the `synthetic_data` (question, expected_answer).
    2.  Inserts the test suite document into `db.test_suites`.
    3.  Returns the test suite ID.
*   `run_evaluation(db, test_suite_id, llm_name)`:
    1.  Calls `evaluation.run_evaluation_harness` to execute the tests.
    2.  Logs the number of results.
    3.  Handles potential differences in return type (dict vs. list) depending on adapter/success.
    4.  Returns the results dictionary/list.
*   `prompt_cleanup()`: Prompts the user via `input()` whether to clean up the ChromaDB collection. Returns `True` for "yes", `False` for "no".
*   `main()`: The main execution function.
    1.  Parses arguments.
    2.  Checks for the `GEMINI_API_KEY` if Gemini is selected.
    3.  Calls `setup_database`.
    4.  Validates the CSV file path.
    5.  Calls `store_dataset`.
    6.  Calls `generate_synthetic_data`.
    7.  Calls `create_test_suite`.
    8.  Calls `run_evaluation`.
    9.  Checks if the evaluation results contain a valid `run_id`.
    10. Calls `evaluation.generate_metrics_report` (which calls `report_generator`).
    11. Prints summary information to the console.
    12. If `--no-cleanup-prompt` is not set, calls `prompt_cleanup` and conditionally cleans the vector store by initializing `VectorStore` and calling `vector_store.cleanup_collection()`.
    13. Catches and logs/prints any top-level exceptions.

### `report_generator.py`

*   `generate_metrics_report(db, evaluation_run_id, output_formats, output_dir)`:
    1.  Fetches the `evaluation_run` document from the database.
    2.  Fetches the associated `test_suite` and `dataset` documents.
    3.  Builds the base `report` dictionary structure with summary info.
    4.  Calculates aggregate metrics using `calculate_aggregate_metrics`.
    5.  Populates `per_question_metrics` by extracting data from the `results` list within the evaluation run document.
    6.  Checks if modified results exist and calls `calculate_modified_questions_summary` if they do.
    7.  Ensures the `output_dir` exists.
    8.  Generates a timestamped base filename.
    9.  Iterates through the requested `output_formats`:
        *   Opens the corresponding file (`.json`, `.txt`, `.html`, `.md`).
        *   Calls the appropriate writer function (`json.dump`, `write_text_report`, `write_html_report`, `write_markdown_report`).
        *   Stores the output path.
    10. Logs the paths of generated reports.
    11. Returns a dictionary mapping format names to file paths.
*   `calculate_aggregate_metrics(results)`: Takes the list of individual test results. Calculates and returns a dictionary containing average, min, and max values for BLEU, ROUGE F1 scores, and METEOR across all results. Handles empty results list.
*   `calculate_modified_questions_summary(results)`: Extracts data specifically from the `modified_results` field (if present) within each result item. Calculates and returns summary statistics (count, success rate, average metrics) for these modified questions. Returns empty dict if no modified results are found.
*   `write_text_report(file, report)`: Writes the report data dictionary to the provided file object in a human-readable plain text format.
*   `write_html_report(file, report)`: Writes the report data to the file object as an HTML document with basic tables and styling for improved readability in a browser. Includes conditional formatting for success/failure.
*   `write_markdown_report(file, report)`: Writes the report data to the file object using Markdown syntax (headers, lists, tables) suitable for rendering on platforms like GitHub.

### `utils.py`

*   `QAPairParser.parse_qa_pairs(text, num_questions)`: The primary parsing function.
    1.  Uses a primary regex (`Question: ... Answer: ...`) designed to handle multi-line content and varying whitespace.
    2.  If the primary regex doesn't find enough pairs, it falls back to splitting the text by double newlines and calling `_extract_qa_from_block` on each block.
    3.  If still not enough pairs are found, it pads the list with "Failed to parse..." entries to match `num_questions`.
    4.  Returns a list of `(question, answer)` tuples, truncated to `num_questions`.
*   `QAPairParser._extract_qa_from_block(block)`: A fallback helper method. Tries multiple simpler regex patterns (e.g., `Q: ... A: ...`, numbered lists) on a text block to extract a single question and answer.
*   `load_data_from_csv(csv_file_path)`:
    1.  Tries reading the CSV file using a list of common encodings (`utf-8`, `latin1`, etc.).
    2.  Checks if the required `chunk_content` column exists.
    3.  If successful, returns the content of the `chunk_content` column as a list of strings.
    4.  Raises appropriate errors for file not found, invalid format, or encoding issues.

### `vector_store.py`

*   `VectorStore.__init__(self)`:
    1.  Connects to the database using `database.connect_to_mongodb` (gets adapter).
    2.  Fetches the vector store configuration document (`db.vector_store.find_one`).
    3.  If config not found (e.g., using JSON DB for the first time), creates and inserts a default configuration.
    4.  Extracts connection details (persist directory, embedding model name). Uses defaults if necessary.
    5.  Initializes the `SentenceTransformer` model.
    6.  Initializes the `chromadb.PersistentClient` using the persist directory.
    7.  Attempts to get or create the ChromaDB collection specified in the config, handling potential `DoesNotExist` errors and retrying creation. Raises an error if collection initialization fails after retries.
*   `VectorStore.add_documents(self, documents)`:
    1.  Encodes the list of `documents` into embeddings using `self.model.encode()`.
    2.  Generates simple string IDs for the documents.
    3.  Calls `self.collection.add()` to store embeddings, documents, and IDs in ChromaDB.
*   `VectorStore.get_similar_documents(self, query, top_k)`:
    1.  Encodes the input `query` string into an embedding.
    2.  Calls `self.collection.query()` with the query embedding and `n_results=top_k`.
    3.  Formats the results from ChromaDB into a list of dictionaries, each containing the document `text` and similarity `score`.
    4.  Handles cases where no documents are found. Returns the list of results or an empty list on error.
*   `VectorStore.cleanup_collection(self)`:
    1.  Gets the collection name from the database configuration (using defaults if needed).
    2.  Calls `self.client.delete_collection()` to remove the ChromaDB collection.
    3.  Calls `self.client.create_collection()` to immediately recreate an empty collection with the same name and settings.
