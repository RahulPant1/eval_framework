# RAG Evaluation Framework: Execution Flow Documentation

## Prerequsistes
- Python 3.8+
- MongoDB Running on localhost (sudo systemctl start mongod)
- Gemini API key (export GEMINI_API_KEY="")

## Execution Flow

This document explains the execution flow when running the command:
```bash
python3 main.py --csv /home/rahul/eval_framework/sample_data.csv
```

## Overview

The RAG Evaluation Framework is designed to evaluate Retrieval-Augmented Generation systems by:
1. Loading document chunks from a CSV file
2. Generating synthetic question-answer pairs using an LLM
3. Creating a test suite from these pairs
4. Running evaluations using a RAG pipeline
5. Calculating performance metrics

## Execution Flow

### 1. Entry Point: `main.py`

The execution begins in the `main()` function of `main.py`, which:

- Parses command-line arguments using `argparse`
- Validates the presence of required API keys
- Orchestrates the entire evaluation process

### 2. Database Setup: `setup_database()`

```python
db = setup_database()
```

This function:
- Calls `connect_to_mongodb()` to establish a connection to MongoDB
- Calls `create_collections()` to ensure all required collections exist
- Calls `insert_initial_data()` to populate reference data if collections are empty

Collections created include:
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

### 3. Dataset Storage: `store_dataset()`

```python
dataset_id = store_dataset(db, csv_path)
```

This function:
- Calls `load_data_from_csv()` from `utils.py` to read document chunks
- Creates a dataset document with metadata and chunks
- Inserts the dataset into MongoDB and returns its ID

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
    - Initializes a `VectorStore` from `vector_store.py`
    - Adds documents to the vector store
    - Calls `generate_rag_response()` which:
      - Retrieves similar documents using vector similarity search
      - Generates an answer using the LLM with the retrieved context
    - Evaluates the response using metrics like BLEU, ROUGE, etc.
    - Stores the results in the test_results collection

## Key Components

### Vector Store (vector_store.py)

The `VectorStore` class:
- Uses SentenceTransformer to generate embeddings
- Stores embeddings in ChromaDB
- Provides similarity search functionality

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

## Logging

The framework logs detailed information to `rag_evaluation.log`, including:
- Database operations
- LLM prompts and responses
- Vector store operations
- Evaluation results

## Conclusion

The RAG Evaluation Framework provides an end-to-end pipeline for evaluating RAG systems. It handles data loading, synthetic QA generation, vector storage, retrieval, response generation, and evaluation metrics calculation in a modular and extensible way.