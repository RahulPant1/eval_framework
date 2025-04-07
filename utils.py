"""
Utils module for shared utility functions in RAG evaluation framework.

Contains:
- QAPairParser class for parsing LLM outputs
- Data loading utilities
"""

import re
import pandas as pd
import logging

class QAPairParser:
    """Parses QA pairs from LLM outputs with multiple formats support"""
    
    @staticmethod
    def parse_qa_pairs(text, num_questions):
        """
        Parse QA pairs from text using a primary regex pattern and fallbacks.
        Handles multi-line questions and answers.
        Returns a list of tuples (question, answer).
        """
        qa_pairs = []
        
        # Log the raw text for debugging
        logging.debug(f"Raw text for QA parsing: {text}")

        # Primary Regex: Looks for "Question: ... Answer: ..." pattern, handling multi-line content.
        # Uses re.DOTALL so '.' matches newlines.
        # Uses lookahead (?=\n\s*Question:|\Z) to correctly delimit the end of an answer.
        # Modified to handle cases where questions or answers contain newlines or where there might be
        # extra whitespace between QA pairs.
        primary_pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\n\s*Question:|\Z)"
        
        try:
            matches = re.finditer(primary_pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(qa_pairs) >= num_questions:
                    break
                question = match.group(1).strip()
                answer = match.group(2).strip()
                if question and answer:
                    qa_pairs.append((question, answer))
        except Exception as e:
            logging.error(f"Error during primary QA parsing with regex: {e}")

        # Fallback: If primary regex yields insufficient pairs, try original block-based approach
        # (This part is kept as a fallback but might be less reliable)
        if len(qa_pairs) < num_questions:
            logging.warning("Primary regex parsing yielded fewer pairs than expected. Trying fallback methods.")
            # Split the text into potential QA blocks based on double newlines
            blocks = re.split(r'\n\s*\n', text)
            for block in blocks:
                 if len(qa_pairs) >= num_questions:
                    break
                 # Use the original _extract_qa_from_block for these blocks
                 # Note: This internal method might need refinement too if issues persist
                 question, answer = QAPairParser._extract_qa_from_block(block)
                 if question and answer and (question, answer) not in qa_pairs: # Avoid duplicates
                    qa_pairs.append((question, answer))

        # Ensure we return exactly num_questions pairs, padding if necessary
        while len(qa_pairs) < num_questions:
            logging.warning(f"Padding QA pair {len(qa_pairs)+1} due to parsing failure.")
            qa_pairs.append((f"Failed to parse question {len(qa_pairs)+1}", 
                             f"Failed to parse answer {len(qa_pairs)+1}"))
        
        logging.info(f"Successfully parsed {len(qa_pairs[:num_questions])} QA pairs.")
        return qa_pairs[:num_questions]  # Return only the requested number
    
    @staticmethod
    def _extract_qa_from_block(block):
        """Extract question and answer from a block of text using multiple patterns"""
        # Pattern 1: "Q: ... A: ..."
        qa_pattern1 = re.search(r'(?:Q(?:uestion)?[\:\.]?\s*)(.*?)(?:\s*A(?:nswer)?[\:\.]?\s*)(.*)$', block, re.DOTALL)
        if qa_pattern1:
            return qa_pattern1.group(1).strip(), qa_pattern1.group(2).strip()
        
        # Pattern 2: "Question: ... Answer: ..."
        qa_pattern2 = re.search(r'(?:Question[\:\.]?\s*)(.*?)(?:\s*Answer[\:\.]?\s*)(.*)$', block, re.DOTALL)
        if qa_pattern2:
            return qa_pattern2.group(1).strip(), qa_pattern2.group(2).strip()
        
        # Pattern 3: Numbered format with Q/A labels
        qa_pattern3 = re.search(r'(?:\d+[\.)\s]*)?(?:Q(?:uestion)?[\:\.]?\s*)(.*?)(?:\s*A(?:nswer)?[\:\.]?\s*)(.*)$', block, re.DOTALL)
        if qa_pattern3:
            return qa_pattern3.group(1).strip(), qa_pattern3.group(2).strip()
        
        # Add more patterns here as needed
        return None, None

def load_data_from_csv(csv_file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(csv_file_path)
        if 'chunk_content' not in df.columns:
            raise ValueError("CSV file must contain a column named 'chunk_content'")
        chunks = df['chunk_content'].tolist()
        logging.info(f"Loaded {len(chunks)} chunks from CSV file: {csv_file_path}")
        return chunks
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.error(f"Error loading data from CSV: {e}")
        raise