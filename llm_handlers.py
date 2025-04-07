"""
LLM Handlers module for interacting with language models in RAG evaluation.

Contains functions for:
- Generating QA pairs from text chunks
- Generating modified questions for robustness testing
- Generating answers using RAG pipeline
- Parsing model outputs
"""

from typing import List, Tuple, Dict, Any
import logging
import google.generativeai as genai
import os
from utils import QAPairParser

# Configuration
MODEL_NAME = "gemini-1.5-flash"  # Default Gemini model

class BaseLLMHandler:
    """Base class for LLM handlers"""
    def generate_qa_pairs(self, prompt_template: str, context: str, num_questions: int) -> str:
        """Generate QA pairs from context"""
        raise NotImplementedError
        
    def generate_modified_question(self, question: str) -> List[str]:
        """Generate modified versions of a question"""
        raise NotImplementedError
        
    def generate_rag_response(self, question: str, context: str) -> str:
        """Generate a response based on the question and context"""
        raise NotImplementedError

class GeminiLLMHandler(BaseLLMHandler):
    """Handler for Gemini 1.5 Flash LLM"""
    def __init__(self):
        # Configure Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logging.warning("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL_NAME)
    
    def generate_qa_pairs(self, prompt_template: str, context: str, num_questions: int) -> str:
        """
        Generate QA pairs from context using Gemini 1.5 Flash.
        
        Args:
            prompt_template: Template for generating QA pairs
            context: Text context to generate QA pairs from
            num_questions: Number of QA pairs to generate
            
        Returns:
            String containing generated QA pairs
        """
        try:
            prompt = prompt_template.format(context=context, num_questions=num_questions)
            logging.info(f"Prompt sent to Gemini model: {prompt}")
            response = self.model.generate_content(prompt)
            logging.info(f"Received response from Gemini model: {response}")

            # Safely extract the generated text from response.candidates[0].content.parts
            if hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                text_parts = []
                for part in parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                    else:
                        logging.warning(f"Unexpected part format: {part}")
                final_text = '\n'.join(text_parts)
                logging.info(f"Final extracted text: {final_text}")
                return final_text

            logging.error("No candidates found in Gemini response")
            return "Error: No valid content generated."

        except Exception as e:
            logging.error(f"Error generating QA pairs with Gemini: {e}")
            return f"Error generating QA pairs: {str(e)}"

    
    def generate_modified_question(self, question: str) -> List[str]:
        """
        Generate modified versions of a question using Gemini 1.5 Flash.
        
        Args:
            question: Original question to modify
            
        Returns:
            List of modified questions
        """
        try:
            prompt = f"Generate a modified version of the following question, without changing its meaning. Provide 3 variations: 1) Paraphrasing, 2) Adding noise, 3) Changing sentence structure. Original Question: {question}"
            response = self.model.generate_content(prompt)
            return response.text.split("\n")
        except Exception as e:
            logging.error(f"Error generating modified questions with Gemini: {e}")
            return [f"Error generating modified question: {str(e)}"]
    
    def generate_answers(self, questions: List[str], context: str) -> List[str]:
        """
        Generate answers for a list of questions based on the provided context using Gemini 1.5 Flash.
        """
        try:
            logging.info("Calling generate_answers method")
            answers = []
            for question in questions:
                prompt = f"Answer the following question based on the context: Question: {question}\n\nContext: {context}"
                logging.info(f"Prompt sent to Gemini model: {prompt}")
                response = self.model.generate_content(prompt)
                logging.info(f"Received response from Gemini model: {response}")

                if hasattr(response, 'candidates') and response.candidates:
                    parts = response.candidates[0].content.parts
                    text_parts = []
                    for part in parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                        else:
                            logging.warning(f"Unexpected part format: {part}")
                    final_text = '\n'.join(text_parts)
                    answers.append(final_text)
                else:
                    logging.warning("No candidates in response")
                    answers.append("Error: No valid content generated.")
            
            return answers
        except Exception as e:
            logging.error(f"Error generating answers with Gemini: {e}")
            return [f"Error generating answer: {str(e)}"] * len(questions)

    
    def generate_rag_response(self, question: str, context: str) -> str:
        """
        Generate a response based on the question and context using Gemini 1.5 Flash.
        
        Args:
            question: Question to answer
            context: Context for answering the question
            
        Returns:
            Generated response
        """
        try:
            prompt = f"Answer the following question based on the context: Question: {question}\n\nContext: {context}"
            logging.info(f"Prompt sent to Gemini model: {prompt}")
            response = self.model.generate_content(prompt)
            logging.info(f"Received response from Gemini model: {response}")
            return response.text
        except Exception as e:
            logging.error(f"Error generating RAG response with Gemini: {e}")
            return f"Error generating response: {str(e)}"

class Llama2LLMHandler(BaseLLMHandler):
    """Handler for Llama 2 LLM"""
    def __init__(self, chunk_content_size: int = 100):
        self.chunk_content_size = chunk_content_size
    
    def generate_qa_pairs(self, prompt_template: str, context: str, num_questions: int) -> str:
        """
        Generate QA pairs from context using Llama 2.
        This is a simplified implementation with placeholder functionality.
        
        Args:
            prompt_template: Template for generating QA pairs
            context: Text context to generate QA pairs from
            num_questions: Number of QA pairs to generate
            
        Returns:
            String containing generated QA pairs
        """
        # Placeholder implementation
        qa_pairs = []
        for i in range(num_questions):
            qa_pairs.append(f"Question {i+1}: What is the main topic of this text?\nAnswer {i+1}: The main topic is about {context[:self.chunk_content_size]}...")
        
        return "\n\n".join(qa_pairs)
    
    def generate_modified_question(self, question: str) -> List[str]:
        """
        Generate modified versions of a question using Llama 2.
        This is a simplified implementation with placeholder functionality.
        
        Args:
            question: Original question to modify
            
        Returns:
            List of modified questions
        """
        return [
            f"Paraphrased: {question}", 
            f"With noise: {question} (with some additional terms)", 
            f"Restructured: Have you considered {question.lower()}?"
        ]
    
    def generate_rag_response(self, question: str, context: str) -> str:
        """
        Generate a response based on the question and context using Llama 2.
        This is a simplified implementation with placeholder functionality.
        
        Args:
            question: Question to answer
            context: Context for answering the question
            
        Returns:
            Generated response
        """
        # Placeholder implementation
        return f"Based on the provided context, the answer to '{question}' is related to {context[:self.chunk_content_size]}..."

def get_llm_handler(db, llm_name: str = "Gemini 1.5 Flash") -> BaseLLMHandler:
    """
    Factory function to get the appropriate LLM handler.
    
    Args:
        db: MongoDB database object
        llm_name: Name of the LLM to use
        
    Returns:
        LLM handler instance
    """
    handlers = {
        "Gemini 1.5 Flash": GeminiLLMHandler(),
        "Llama 2": Llama2LLMHandler()
    }
    
    # Check if the requested LLM exists in the database
    llm_exists = db.llms.find_one({"llm_name": llm_name})
    if not llm_exists or llm_name not in handlers:
        logging.warning(f"LLM '{llm_name}' not found or not supported. Falling back to Gemini 1.5 Flash.")
        return handlers["Gemini 1.5 Flash"]
    
    return handlers[llm_name]