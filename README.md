# RAG Evaluation Framework

This framework provides tools for evaluating Retrieval-Augmented Generation (RAG) systems. It includes components for generating synthetic QA pairs, running evaluations, and calculating metrics.

## Features

- MongoDB integration for storing datasets, QA pairs, and evaluation results
- Persistent collections that are reused across multiple runs
- Vector store implementation using SentenceTransformer embeddings
- Support for multiple LLMs (Gemini 1.5 Flash, Llama 2)
- Evaluation metrics including BLEU, ROUGE, and more
- Synthetic data generation for testing

## Prerequisites

- Python 3.8+
- MongoDB running locally or accessible via URI
- API keys for LLMs (e.g., GEMINI_API_KEY)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install pymongo google-generativeai nltk rouge sentence-transformers numpy pandas
```

3. Set up environment variables:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Project Structure

- `database.py`: MongoDB operations
- `evaluation.py`: Evaluation metrics and harnesses
- `llm_handlers.py`: LLM interaction classes
- `utils.py`: Utility functions
- `vector_store.py`: Vector embedding storage using ChromaDB
- `main.py`: Entry point for running the framework
- `__init__.py`: Marks the directory as a Python package
- `run_evaluation.sh`: Shell script to automate running the evaluation

- `test_chromadb.py`: Contains tests for ChromaDB vector store functionality
- `test_gemini_api.py`: Contains tests for Gemini LLM API interactions
- `update_gemini_model.py`: Utility script for Gemini model configurations (if applicable)
- `manage_chroma.py`: Utility script to delete ChromaDB collections
- `manage_mongodb.py`: Utility script to delete MongoDB collections/databases

## Usage

### Basic Usage

1. Prepare a CSV file with document chunks (must have a 'chunk_content' column)
2. Run the evaluation framework:

```bash
python main.py --csv your_data.csv --llm "Gemini 1.5 Flash" --questions 3
```

This will:
- Connect to MongoDB
- Create necessary collections (or reuse existing ones)
- Load data from the CSV file
- Generate synthetic QA pairs
- Run evaluation harnesses
- Output results

### Advanced Usage

You can also use individual components of the framework in your own code:

```python
from database import connect_to_mongodb
from llm_handlers import get_llm_handler
from evaluation import run_evaluation_harness

# Connect to MongoDB
db = connect_to_mongodb()

# Get LLM handler
llm_handler = get_llm_handler(db, "Gemini 1.5 Flash")

# Generate QA pairs
prompt_template = "Generate {num_questions} question-answer pairs from the following text. Format each pair as 'Question: <question>\nAnswer: <answer>'\n\nText: {context}"
qa_pairs = llm_handler.generate_qa_pairs(prompt_template, your_context, 3)

# Run evaluation
results = run_evaluation_harness(db, test_suite_id, "Gemini 1.5 Flash")
```

## Troubleshooting

- If you encounter MongoDB connection issues, ensure MongoDB is running and accessible
- For LLM API errors, verify your API keys are correctly set in environment variables
- For missing dependencies, install them using pip

## License

This project is licensed under the MIT License - see the LICENSE file for details.