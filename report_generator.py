"""Report generator module for RAG evaluation framework.

Contains functions for generating comprehensive reports from evaluation results.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import os

def generate_metrics_report(db, evaluation_run_id, output_formats=["html", "markdown"], output_dir="reports"):
    """
    Generates a comprehensive metrics report from an evaluation run.
    
    Args:
        db: MongoDB database object
        evaluation_run_id: ID of the evaluation run to generate report for
        output_formats: List of formats to generate (json, txt, html, markdown)
        output_dir: Directory to save the report
        
    Returns:
        Dictionary of paths to the generated report files
    """
    try:
        # Get evaluation run
        evaluation_run = db.evaluation_runs.find_one({"_id": evaluation_run_id})
        if not evaluation_run:
            raise ValueError(f"Evaluation run with ID {evaluation_run_id} not found")
            
        # Get test suite
        test_suite = db.test_suites.find_one({"_id": evaluation_run["test_suite_id"]})
        if not test_suite:
            raise ValueError(f"Test suite with ID {evaluation_run['test_suite_id']} not found")
            
        # Get dataset
        dataset = db.datasets.find_one({"_id": test_suite["dataset_id"]})
        if not dataset:
            raise ValueError(f"Dataset with ID {test_suite['dataset_id']} not found")
        
        # Create report structure
        report = {
            "report_id": str(evaluation_run_id),
            "timestamp": datetime.now().isoformat(),
            "evaluation_summary": {
                "test_suite_name": test_suite["test_suite_name"],
                "dataset_name": dataset["dataset_name"],
                "llm_name": evaluation_run["llm_name"],
                "num_tests": len(evaluation_run["results"]),
                "success_rate": sum(1 for r in evaluation_run["results"] if r["success"]) / len(evaluation_run["results"]) if evaluation_run["results"] else 0,
            },
            "aggregate_metrics": calculate_aggregate_metrics(evaluation_run["results"]),
            "per_question_metrics": []
        }
        
        # Add per-question metrics
        for result in evaluation_run["results"]:
            question_metrics = {
                "question": result["question"],
                "success": result["success"],
                "metrics": result["metrics"],
                "explanation": result["explanation"]
            }
            report["per_question_metrics"].append(question_metrics)
            
        # Add modified questions metrics if available
        modified_questions_exist = any("modified_results" in r for r in evaluation_run["results"])
        if modified_questions_exist:
            report["modified_questions_summary"] = calculate_modified_questions_summary(evaluation_run["results"])
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_evaluation_report_{timestamp}"
        
        output_paths = {}
        
        # If no formats specified, default to HTML and Markdown
        if not output_formats:
            output_formats = ["html", "markdown"]
            
        for output_format in output_formats:
            if output_format == "json":
                file_path = os.path.join(output_dir, f"{filename}.json")
                with open(file_path, "w") as f:
                    json.dump(report, f, indent=2)
                output_paths["json"] = file_path
                
            elif output_format == "txt":
                file_path = os.path.join(output_dir, f"{filename}.txt")
                with open(file_path, "w") as f:
                    write_text_report(f, report)
                output_paths["txt"] = file_path
                
            elif output_format == "html":
                file_path = os.path.join(output_dir, f"{filename}.html")
                with open(file_path, "w") as f:
                    write_html_report(f, report)
                output_paths["html"] = file_path
                
            elif output_format == "markdown":
                file_path = os.path.join(output_dir, f"{filename}.md")
                with open(file_path, "w") as f:
                    write_markdown_report(f, report)
                output_paths["markdown"] = file_path
                
            else:
                logging.warning(f"Unsupported output format: {output_format}")
            
        logging.info(f"Generated metrics reports: {', '.join(output_paths.values())}")
        return output_paths
    
    except Exception as e:
        logging.error(f"Error generating metrics report: {e}")
        return None

def calculate_aggregate_metrics(results):
    """
    Calculates aggregate metrics from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not results:
        return {}
        
    # Calculate average metrics
    avg_bleu = sum(r["metrics"]["bleu"] for r in results) / len(results)
    avg_rouge_1_precision = sum(r["metrics"]["rouge"]["rouge-1"]["precision"] for r in results) / len(results)
    avg_rouge_1_recall = sum(r["metrics"]["rouge"]["rouge-1"]["recall"] for r in results) / len(results)
    avg_rouge_1_f1 = sum(r["metrics"]["rouge"]["rouge-1"]["f1"] for r in results) / len(results)
    avg_rouge_2_f1 = sum(r["metrics"]["rouge"]["rouge-2"]["f1"] for r in results) / len(results)
    avg_rouge_l_f1 = sum(r["metrics"]["rouge"]["rouge-l"]["f1"] for r in results) / len(results)
    avg_meteor = sum(r["metrics"]["meteor"] for r in results) / len(results)
    
    # Calculate min/max metrics
    min_bleu = min(r["metrics"]["bleu"] for r in results)
    max_bleu = max(r["metrics"]["bleu"] for r in results)
    min_rouge_1_f1 = min(r["metrics"]["rouge"]["rouge-1"]["f1"] for r in results)
    max_rouge_1_f1 = max(r["metrics"]["rouge"]["rouge-1"]["f1"] for r in results)
    min_meteor = min(r["metrics"]["meteor"] for r in results)
    max_meteor = max(r["metrics"]["meteor"] for r in results)
    
    return {
        "bleu": {
            "average": avg_bleu,
            "min": min_bleu,
            "max": max_bleu
        },
        "rouge": {
            "rouge-1": {
                "precision": avg_rouge_1_precision,
                "recall": avg_rouge_1_recall,
                "f1": avg_rouge_1_f1
            },
            "rouge-2": {
                "f1": avg_rouge_2_f1
            },
            "rouge-l": {
                "f1": avg_rouge_l_f1
            }
        },
        "meteor": {
            "average": avg_meteor,
            "min": min_meteor,
            "max": max_meteor
        }
    }

def calculate_modified_questions_summary(results):
    """
    Calculates summary metrics for modified questions.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with modified questions summary
    """
    modified_results = []
    for result in results:
        if "modified_results" in result and result["modified_results"]:
            for mod_result in result["modified_results"]:
                modified_results.append({
                    "original_question": result["question"],
                    "modified_question": mod_result["modified_question"],
                    "success": mod_result["success"],
                    "metrics": mod_result["metrics"]
                })
    
    if not modified_results:
        return {}
        
    # Calculate success rate
    success_rate = sum(1 for r in modified_results if r["success"]) / len(modified_results)
    
    # Calculate average metrics
    avg_bleu = sum(r["metrics"]["bleu"] for r in modified_results) / len(modified_results)
    avg_rouge_1_f1 = sum(r["metrics"]["rouge"]["rouge-1"]["f1"] for r in modified_results) / len(modified_results)
    avg_meteor = sum(r["metrics"]["meteor"] for r in modified_results) / len(modified_results)
    
    return {
        "num_modified_questions": len(modified_results),
        "success_rate": success_rate,
        "average_metrics": {
            "bleu": avg_bleu,
            "rouge-1-f1": avg_rouge_1_f1,
            "meteor": avg_meteor
        }
    }

def write_text_report(file, report):
    """
    Writes report in text format.
    
    Args:
        file: File object to write to
        report: Report data
    """
    file.write(f"RAG Evaluation Report\n")
    file.write(f"Generated: {report['timestamp']}\n")
    file.write(f"Report ID: {report['report_id']}\n\n")
    
    # Write summary
    file.write(f"Evaluation Summary:\n")
    file.write(f"  Test Suite: {report['evaluation_summary']['test_suite_name']}\n")
    file.write(f"  Dataset: {report['evaluation_summary']['dataset_name']}\n")
    file.write(f"  LLM: {report['evaluation_summary']['llm_name']}\n")
    file.write(f"  Number of Tests: {report['evaluation_summary']['num_tests']}\n")
    file.write(f"  Success Rate: {report['evaluation_summary']['success_rate']:.2%}\n\n")
    
    # Write aggregate metrics
    file.write(f"Aggregate Metrics:\n")
    file.write(f"  BLEU: {report['aggregate_metrics']['bleu']['average']:.4f} (min: {report['aggregate_metrics']['bleu']['min']:.4f}, max: {report['aggregate_metrics']['bleu']['max']:.4f})\n")
    file.write(f"  ROUGE-1 F1: {report['aggregate_metrics']['rouge']['rouge-1']['f1']:.4f}\n")
    file.write(f"  ROUGE-2 F1: {report['aggregate_metrics']['rouge']['rouge-2']['f1']:.4f}\n")
    file.write(f"  ROUGE-L F1: {report['aggregate_metrics']['rouge']['rouge-l']['f1']:.4f}\n")
    file.write(f"  METEOR: {report['aggregate_metrics']['meteor']['average']:.4f} (min: {report['aggregate_metrics']['meteor']['min']:.4f}, max: {report['aggregate_metrics']['meteor']['max']:.4f})\n\n")
    
    # Write per-question metrics
    file.write(f"Per-Question Metrics:\n")
    for i, question in enumerate(report['per_question_metrics']):
        file.write(f"  Question {i+1}: {question['question']}\n")
        file.write(f"    Success: {'Yes' if question['success'] else 'No'}\n")
        file.write(f"    Explanation: {question['explanation']}\n")
        file.write(f"    BLEU: {question['metrics']['bleu']:.4f}\n")
        file.write(f"    ROUGE-1 F1: {question['metrics']['rouge']['rouge-1']['f1']:.4f}\n")
        file.write(f"    METEOR: {question['metrics']['meteor']:.4f}\n\n")
    
    # Write modified questions summary if available
    if "modified_questions_summary" in report:
        file.write(f"Modified Questions Summary:\n")
        file.write(f"  Number of Modified Questions: {report['modified_questions_summary']['num_modified_questions']}\n")
        file.write(f"  Success Rate: {report['modified_questions_summary']['success_rate']:.2%}\n")
        file.write(f"  Average BLEU: {report['modified_questions_summary']['average_metrics']['bleu']:.4f}\n")
        file.write(f"  Average ROUGE-1 F1: {report['modified_questions_summary']['average_metrics']['rouge-1-f1']:.4f}\n")
        file.write(f"  Average METEOR: {report['modified_questions_summary']['average_metrics']['meteor']:.4f}\n")

def write_html_report(file, report):
    """
    Writes report in HTML format.
    
    Args:
        file: File object to write to
        report: Report data
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metrics {{ margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Evaluation Report</h1>
            <p>Generated: {report['timestamp']}</p>
            <p>Report ID: {report['report_id']}</p>
            
            <div class="summary">
                <h2>Evaluation Summary</h2>
                <p><strong>Test Suite:</strong> {report['evaluation_summary']['test_suite_name']}</p>
                <p><strong>Dataset:</strong> {report['evaluation_summary']['dataset_name']}</p>
                <p><strong>LLM:</strong> {report['evaluation_summary']['llm_name']}</p>
                <p><strong>Number of Tests:</strong> {report['evaluation_summary']['num_tests']}</p>
                <p><strong>Success Rate:</strong> {report['evaluation_summary']['success_rate']:.2%}</p>
            </div>
            
            <div class="metrics">
                <h2>Aggregate Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Average</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                    <tr>
                        <td>BLEU</td>
                        <td>{report['aggregate_metrics']['bleu']['average']:.4f}</td>
                        <td>{report['aggregate_metrics']['bleu']['min']:.4f}</td>
                        <td>{report['aggregate_metrics']['bleu']['max']:.4f}</td>
                    </tr>
                    <tr>
                        <td>ROUGE-1 F1</td>
                        <td>{report['aggregate_metrics']['rouge']['rouge-1']['f1']:.4f}</td>
                        <td colspan="2">-</td>
                    </tr>
                    <tr>
                        <td>ROUGE-2 F1</td>
                        <td>{report['aggregate_metrics']['rouge']['rouge-2']['f1']:.4f}</td>
                        <td colspan="2">-</td>
                    </tr>
                    <tr>
                        <td>ROUGE-L F1</td>
                        <td>{report['aggregate_metrics']['rouge']['rouge-l']['f1']:.4f}</td>
                        <td colspan="2">-</td>
                    </tr>
                    <tr>
                        <td>METEOR</td>
                        <td>{report['aggregate_metrics']['meteor']['average']:.4f}</td>
                        <td>{report['aggregate_metrics']['meteor']['min']:.4f}</td>
                        <td>{report['aggregate_metrics']['meteor']['max']:.4f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="per-question">
                <h2>Per-Question Metrics</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Question</th>
                        <th>Success</th>
                        <th>BLEU</th>
                        <th>ROUGE-1 F1</th>
                        <th>METEOR</th>
                    </tr>
    """
    
    for i, question in enumerate(report['per_question_metrics']):
        success_class = "success" if question['success'] else "failure"
        success_text = "Yes" if question['success'] else "No"
        html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{question['question']}</td>
                        <td class="{success_class}">{success_text}</td>
                        <td>{question['metrics']['bleu']:.4f}</td>
                        <td>{question['metrics']['rouge']['rouge-1']['f1']:.4f}</td>
                        <td>{question['metrics']['meteor']:.4f}</td>
                    </tr>
        """
    
    html += """
                </table>
            </div>
    """
    
    # Add modified questions summary if available
    if "modified_questions_summary" in report:
        html += f"""
            <div class="modified-questions">
                <h2>Modified Questions Summary</h2>
                <p><strong>Number of Modified Questions:</strong> {report['modified_questions_summary']['num_modified_questions']}</p>
                <p><strong>Success Rate:</strong> {report['modified_questions_summary']['success_rate']:.2%}</p>
                <p><strong>Average BLEU:</strong> {report['modified_questions_summary']['average_metrics']['bleu']:.4f}</p>
                <p><strong>Average ROUGE-1 F1:</strong> {report['modified_questions_summary']['average_metrics']['rouge-1-f1']:.4f}</p>
                <p><strong>Average METEOR:</strong> {report['modified_questions_summary']['average_metrics']['meteor']:.4f}</p>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    file.write(html)


def write_markdown_report(file, report):
    """
    Writes report in Markdown format.
    
    Args:
        file: File object to write to
        report: Report data
    """
    file.write(f"# RAG Evaluation Report\n\n")
    file.write(f"Generated: {report['timestamp']}  \n")
    file.write(f"Report ID: {report['report_id']}  \n\n")
    
    # Write summary
    file.write(f"## Evaluation Summary\n\n")
    file.write(f"- **Test Suite:** {report['evaluation_summary']['test_suite_name']}  \n")
    file.write(f"- **Dataset:** {report['evaluation_summary']['dataset_name']}  \n")
    file.write(f"- **LLM:** {report['evaluation_summary']['llm_name']}  \n")
    file.write(f"- **Number of Tests:** {report['evaluation_summary']['num_tests']}  \n")
    file.write(f"- **Success Rate:** {report['evaluation_summary']['success_rate']:.2%}  \n\n")
    
    # Write aggregate metrics
    file.write(f"## Aggregate Metrics\n\n")
    file.write("| Metric | Average | Min | Max |\n")
    file.write("|--------|---------|-----|-----|\n")
    file.write(f"| BLEU | {report['aggregate_metrics']['bleu']['average']:.4f} | {report['aggregate_metrics']['bleu']['min']:.4f} | {report['aggregate_metrics']['bleu']['max']:.4f} |\n")
    file.write(f"| ROUGE-1 F1 | {report['aggregate_metrics']['rouge']['rouge-1']['f1']:.4f} | - | - |\n")
    file.write(f"| ROUGE-2 F1 | {report['aggregate_metrics']['rouge']['rouge-2']['f1']:.4f} | - | - |\n")
    file.write(f"| ROUGE-L F1 | {report['aggregate_metrics']['rouge']['rouge-l']['f1']:.4f} | - | - |\n")
    file.write(f"| METEOR | {report['aggregate_metrics']['meteor']['average']:.4f} | {report['aggregate_metrics']['meteor']['min']:.4f} | {report['aggregate_metrics']['meteor']['max']:.4f} |\n\n")
    
    # Write per-question metrics
    file.write(f"## Per-Question Metrics\n\n")
    
    for i, question in enumerate(report['per_question_metrics']):
        file.write(f"### Question {i+1}: {question['question']}\n\n")
        file.write(f"- **Success:** {'✅ Yes' if question['success'] else '❌ No'}  \n")
        file.write(f"- **Explanation:** {question['explanation']}  \n")
        file.write(f"- **Metrics:**  \n")
        file.write(f"  - BLEU: {question['metrics']['bleu']:.4f}  \n")
        file.write(f"  - ROUGE-1 F1: {question['metrics']['rouge']['rouge-1']['f1']:.4f}  \n")
        file.write(f"  - METEOR: {question['metrics']['meteor']:.4f}  \n\n")
    
    # Write modified questions summary if available
    if "modified_questions_summary" in report:
        file.write(f"## Modified Questions Summary\n\n")
        file.write(f"- **Number of Modified Questions:** {report['modified_questions_summary']['num_modified_questions']}  \n")
        file.write(f"- **Success Rate:** {report['modified_questions_summary']['success_rate']:.2%}  \n")
        file.write(f"- **Average Metrics:**  \n")
        file.write(f"  - BLEU: {report['modified_questions_summary']['average_metrics']['bleu']:.4f}  \n")
        file.write(f"  - ROUGE-1 F1: {report['modified_questions_summary']['average_metrics']['rouge-1-f1']:.4f}  \n")
        file.write(f"  - METEOR: {report['modified_questions_summary']['average_metrics']['meteor']:.4f}  \n")