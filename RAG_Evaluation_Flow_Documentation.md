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