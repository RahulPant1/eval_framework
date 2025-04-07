#!/bin/bash

# RAG Evaluation Framework Runner Script
# Modified to run in current conda environment

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "GEMINI_API_KEY environment variable not set."
    echo "Please set it using: export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Run the evaluation framework
echo "Running RAG evaluation framework..."
python main.py --csv sample_data.csv --llm "Gemini 1.5 Flash" --questions 2

echo "Evaluation complete."