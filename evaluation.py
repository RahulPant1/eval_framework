"""Evaluation module for RAG systems in the evaluation framework.

Contains functions for:
- Generating RAG responses
- Evaluating response quality
- Calculating metrics (BLEU, ROUGE, METEOR)
- Running evaluation harnesses
- Generating evaluation reports
"""

from typing import Dict, List, Any, Tuple
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.tokenize import word_tokenize
import nltk
from datetime import datetime

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Error downloading NLTK resources: {e}")

def generate_rag_response(question: str, db, dataset_id, llm_name: str, num_results: int = 3) -> str:
    """
    Retrieves relevant context chunks from vector store and generates an answer.
    
    Args:
        question: The question to answer
        db: MongoDB database object
        dataset_id: ID of the dataset to use
        llm_name: Name of the LLM to use
        num_results: Number of context chunks to retrieve
        
    Returns:
        Generated response text
    """
    from vector_store import VectorStore
    from llm_handlers import get_llm_handler
    import os
    
    try:
        # Get dataset chunks
        dataset = db.datasets.find_one({"_id": dataset_id})
        if not dataset:
            raise ValueError(f"Dataset with ID {dataset_id} not found")
            
        # Initialize vector store - chunks are already added during dataset loading
        vector_store = VectorStore()
        
        # Retrieve similar documents
        similar_docs = vector_store.get_similar_documents(question, top_k=num_results)
        if not similar_docs:
            logging.warning(f"No relevant documents found for question: {question}")
            return "I apologize, but I couldn't find relevant information to answer your question."
            
        context = " ".join([doc["text"] for doc in similar_docs])
        
        # Get LLM handler
        llm_handler = get_llm_handler(db, llm_name)
        
        # Generate answer
        answers = llm_handler.generate_answers([question], context)
        return answers[0] if answers else "Sorry, I encountered an error while generating the response."
    except Exception as e:
        logging.error(f"Error generating RAG response: {e}")
        return f"Error generating response: {str(e)}"

def evaluate_response(rag_response: str, expected_answer: str) -> Tuple[bool, str]:
    """
    Basic evaluation of whether response contains expected information.
    
    Args:
        rag_response: Generated RAG response
        expected_answer: Expected answer text
        
    Returns:
        Tuple of (success boolean, explanation)
    """
    # Simple check if key parts of expected answer are in the response
    expected_tokens = set(word_tokenize(expected_answer.lower()))
    response_tokens = set(word_tokenize(rag_response.lower()))
    
    # Calculate token overlap
    common_tokens = expected_tokens.intersection(response_tokens)
    overlap_ratio = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
    
    if overlap_ratio >= 0.7:  # 70% overlap threshold
        return True, f"Response contains {overlap_ratio:.2%} of expected information"
    else:
        return False, f"Response only contains {overlap_ratio:.2%} of expected information"

def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculates BLEU score between reference and candidate texts.
    
    Args:
        reference: Reference text
        candidate: Candidate text to evaluate
        
    Returns:
        BLEU score (0-1)
    """
    try:
        # Tokenize texts
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())
        
        # Calculate BLEU score with smoothing
        smoothie = SmoothingFunction().method1
        return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)
    except Exception as e:
        logging.error(f"Error calculating BLEU score: {e}")
        return 0.0

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculates ROUGE scores for precision, recall, and F1.
    
    Args:
        reference: Reference text
        candidate: Candidate text to evaluate
        
    Returns:
        Dictionary of ROUGE metrics
    """
    try:
        rouge = Rouge()
        scores = rouge.get_scores(candidate, reference)[0]
        
        # Extract and return the scores
        return {
            "rouge-1": {
                "precision": scores["rouge-1"]["p"],
                "recall": scores["rouge-1"]["r"],
                "f1": scores["rouge-1"]["f"]
            },
            "rouge-2": {
                "precision": scores["rouge-2"]["p"],
                "recall": scores["rouge-2"]["r"],
                "f1": scores["rouge-2"]["f"]
            },
            "rouge-l": {
                "precision": scores["rouge-l"]["p"],
                "recall": scores["rouge-l"]["r"],
                "f1": scores["rouge-l"]["f"]
            }
        }
    except Exception as e:
        logging.error(f"Error calculating ROUGE scores: {e}")
        return {
            "rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge-2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "rouge-l": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }

def calculate_meteor(reference: str, candidate: str) -> float:
    """
    Calculates METEOR score for semantic similarity.
    
    Args:
        reference: Reference text
        candidate: Candidate text to evaluate
        
    Returns:
        METEOR score (0-1)
    """
    try:
        # Tokenize texts
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        
        # Calculate METEOR score
        return nltk.translate.meteor_score.meteor_score([reference_tokens], candidate_tokens)
    except Exception as e:
        logging.error(f"Error calculating METEOR score: {e}")
        return 0.0

def run_evaluation_harness(db, test_suite_id, llm_name: str) -> Dict[str, Any]:
    """
    Main evaluation pipeline that runs tests for a given test suite.
    
    Args:
        db: MongoDB database object
        test_suite_id: ID of the test suite to run
        llm_name: Name of the LLM to use
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Get test suite
        test_suite = db.test_suites.find_one({"_id": test_suite_id})
        if not test_suite:
            raise ValueError(f"Test suite with ID {test_suite_id} not found")
            
        # Get dataset
        dataset_id = test_suite["dataset_id"]
        
        # Get synthetic data for this dataset
        synthetic_data = list(db.synthetic_data.find({"dataset_id": dataset_id}))
        
        results = []
        for data_point in synthetic_data:
            # Original question evaluation
            question = data_point["question"]
            expected_answer = data_point["answer"]
            
            # Generate RAG response
            rag_response = generate_rag_response(question, db, dataset_id, llm_name)
            
            # Evaluate response
            success, explanation = evaluate_response(rag_response, expected_answer)
            
            # Calculate metrics
            bleu_score = calculate_bleu(expected_answer, rag_response)
            rouge_scores = calculate_rouge(expected_answer, rag_response)
            meteor_score = calculate_meteor(expected_answer, rag_response)
            
            # Store result
            result = {
                "question": question,
                "expected_answer": expected_answer,
                "rag_response": rag_response,
                "success": success,
                "explanation": explanation,
                "metrics": {
                    "bleu": bleu_score,
                    "rouge": rouge_scores,
                    "meteor": meteor_score
                }
            }
            
            # If modified questions exist, evaluate them too
            # if "modified_questions" in data_point and data_point["modified_questions"]:
            #     modified_results = []
            #     for modified_question in data_point["modified_questions"]:
            #         # Generate RAG response for modified question
            #         mod_rag_response = generate_rag_response(modified_question, db, dataset_id, llm_name)
                    
            #         # Evaluate response
            #         mod_success, mod_explanation = evaluate_response(mod_rag_response, expected_answer)
                    
            #         # Calculate metrics
            #         mod_bleu = calculate_bleu(expected_answer, mod_rag_response)
            #         mod_rouge = calculate_rouge(expected_answer, mod_rag_response)
            #         mod_meteor = calculate_meteor(expected_answer, mod_rag_response)
                    
            #         # Store modified result
            #         modified_results.append({
            #             "modified_question": modified_question,
            #             "rag_response": mod_rag_response,
            #             "success": mod_success,
            #             "explanation": mod_explanation,
            #             "metrics": {
            #                 "bleu": mod_bleu,
            #                 "rouge": mod_rouge,
            #                 "meteor": mod_meteor
            #             }
            #         })
                
            #     result["modified_results"] = modified_results
            
            results.append(result)
        
        # Store evaluation run
        evaluation_run = {
            "test_suite_id": test_suite_id,
            "llm_name": llm_name,
            "timestamp": datetime.now(),
            "results": results
        }
        
        evaluation_run_id = db.evaluation_runs.insert_one(evaluation_run).inserted_id
        
        # Store individual test results in test_results collection
        for i, result in enumerate(results):
            test_result = {
                "evaluation_run_id": evaluation_run_id,
                "test_suite_id": test_suite_id,
                "question": result["question"],
                "expected_answer": result["expected_answer"],
                "rag_response": result["rag_response"],
                "success": result["success"],
                "explanation": result["explanation"],
                "timestamp": datetime.now()
            }
            db.test_results.insert_one(test_result)
            
            # Store metrics in metrics collection
            metrics_entry = {
                "evaluation_run_id": evaluation_run_id,
                "test_result_id": test_result["_id"],
                "metrics": result["metrics"],
                "timestamp": datetime.now()
            }
            db.metrics.insert_one(metrics_entry)
        
        # Store metrics definitions if they don't exist
        if db.metrics_definitions.count_documents({}) == 0:
            metrics_definitions = [
                {
                    "metric_name": "bleu",
                    "metric_description": "BLEU (Bilingual Evaluation Understudy) score for measuring text similarity",
                    "metric_type": "generation",
                    "metric_range": "0-1",
                    "higher_is_better": True
                },
                {
                    "metric_name": "rouge",
                    "metric_description": "ROUGE (Recall-Oriented Understudy for Gisting Evaluation) for measuring text similarity",
                    "metric_type": "generation",
                    "metric_range": "0-1",
                    "higher_is_better": True
                },
                {
                    "metric_name": "meteor",
                    "metric_description": "METEOR (Metric for Evaluation of Translation with Explicit ORdering) for semantic similarity",
                    "metric_type": "generation",
                    "metric_range": "0-1",
                    "higher_is_better": True
                }
            ]
            db.metrics_definitions.insert_many(metrics_definitions)
        
        return {
            "run_id": evaluation_run_id,
            "test_suite_id": test_suite_id,
            "llm_name": llm_name,
            "num_tests": len(results),
            "success_rate": sum(1 for r in results if r["success"]) / len(results) if results else 0,
            "average_metrics": {
                "bleu": sum(r["metrics"]["bleu"] for r in results) / len(results) if results else 0,
                "rouge-1-f1": sum(r["metrics"]["rouge"]["rouge-1"]["f1"] for r in results) / len(results) if results else 0,
                "meteor": sum(r["metrics"]["meteor"] for r in results) / len(results) if results else 0
            }
        }
    except Exception as e:
        logging.error(f"Error running evaluation harness: {e}")
        return {"error": str(e)}


def generate_metrics_report(db, evaluation_run_id, output_format="json", output_dir="reports"):
    """
    Generates a comprehensive metrics report from an evaluation run.
    
    Args:
        db: MongoDB database object
        evaluation_run_id: ID of the evaluation run to generate report for
        output_format: Format of the report (json, txt, html)
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
    from report_generator import generate_metrics_report as generate_report
    
    try:
        # Call the report generator function
        report_path = generate_report(db, evaluation_run_id, [output_format], output_dir)
        logging.info(f"Generated metrics report at: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Error generating metrics report: {e}")
        return None