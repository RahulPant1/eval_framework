# # #!/usr/bin/env python3
#!/usr/bin/env python3
"""
Test script to verify Gemini API functionality for QA pair generation.
This script tests the ability to generate question-answer pairs from context
and properly parse them using the framework's QAPairParser.
"""

import os
import sys
import logging
import google.generativeai as genai
from utils import QAPairParser

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gemini_api(num_questions=3, model_name='gemini-1.5-flash'):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY=your_api_key")
        return False

    try:
        genai.configure(api_key=api_key)

        # Create the model instance
        model = genai.GenerativeModel(model_name)

        sample_context = (
            "The Python programming language was created by Guido van Rossum and released in 1991. "
            "Python is known for its readability and simplicity. It supports multiple programming paradigms "
            "including procedural, object-oriented, and functional programming."
        )

        prompt = (
            f"Generate {num_questions} question-answer pairs from the following text. "
            "Format each pair as 'Question: <question>\\nAnswer: <answer>'\n\n"
            f"Text: {sample_context}"
        )

        logger.info(f"Sending prompt to Gemini model {model_name}...")

        # Just pass a string prompt here
        response = model.generate_content(prompt)

        # Simplified response handling
        try:
            # For gemini-1.5 models, the response structure is different
            # Access the text directly from the response object
            response_text = response.text
            logger.info(f"Raw API response:\n{response_text}")
            
            # Parse the QA pairs using the framework's QAPairParser
            qa_pairs = QAPairParser.parse_qa_pairs(response_text, num_questions)
            
            # Display the parsed QA pairs in a structured format
            logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs:")
            for i, (question, answer) in enumerate(qa_pairs, 1):
                logger.info(f"\nPair {i}:")
                logger.info(f"Question: {question}")
                logger.info(f"Answer: {answer}")
            
            # Check if we got the expected number of QA pairs
            if len(qa_pairs) == num_questions:
                logger.info(f"✅ Successfully generated and parsed {num_questions} QA pairs")
                return True
            else:
                logger.warning(f"⚠️ Expected {num_questions} QA pairs but got {len(qa_pairs)}")
                return len(qa_pairs) > 0  # Return True if we got at least some QA pairs
            
        except AttributeError as e:
            # If response.text is not available, try to access content through candidates
            logger.warning(f"Could not access text directly: {e}")
            try:
                # Try to access through candidates if available
                if hasattr(response, 'candidates') and response.candidates:
                    response_text = response.candidates[0].content.parts[0].text
                    logger.info(f"Raw API response from candidates:\n{response_text}")
                    
                    # Parse the QA pairs using the framework's QAPairParser
                    qa_pairs = QAPairParser.parse_qa_pairs(response_text, num_questions)
                    
                    # Display the parsed QA pairs
                    logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs:")
                    for i, (question, answer) in enumerate(qa_pairs, 1):
                        logger.info(f"\nPair {i}:")
                        logger.info(f"Question: {question}")
                        logger.info(f"Answer: {answer}")
                    
                    return len(qa_pairs) > 0  # Return True if we got at least some QA pairs
                else:
                    logger.error("No candidates found in response")
                    return False
            except Exception as inner_e:
                logger.error(f"Error accessing response content: {inner_e}")
                return False

    except Exception as e:
        logger.error(f"Error testing Gemini API: {e}")
        print(f"\nError: {e}")
        return False

def test_with_custom_context(context, num_questions=3, model_name='gemini-1.5-flash'):
    """Test QA pair generation with a custom context"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY=your_api_key")
        return False

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt = (
            f"Generate {num_questions} question-answer pairs from the following text. "
            "Format each pair as 'Question: <question>\\nAnswer: <answer>'\n\n"
            f"Text: {context}"
        )

        logger.info(f"Sending prompt with custom context to Gemini model {model_name}...")
        response = model.generate_content(prompt)
        
        try:
            response_text = response.text
            logger.info(f"Raw API response:\n{response_text}")
            
            # Parse the QA pairs
            qa_pairs = QAPairParser.parse_qa_pairs(response_text, num_questions)
            
            # Display the parsed QA pairs
            logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs:")
            for i, (question, answer) in enumerate(qa_pairs, 1):
                logger.info(f"\nPair {i}:")
                logger.info(f"Question: {question}")
                logger.info(f"Answer: {answer}")
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            return []
            
    except Exception as e:
        logger.error(f"Error testing with custom context: {e}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Gemini API for QA pair generation')
    parser.add_argument('--context', type=str, help='Custom context to generate QA pairs from')
    parser.add_argument('--num_questions', type=int, default=3, help='Number of QA pairs to generate')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='Gemini model to use')
    
    args = parser.parse_args()
    
    if args.context:
        # Test with custom context
        qa_pairs = test_with_custom_context(args.context, args.num_questions, args.model)
        success = len(qa_pairs) == args.num_questions
    else:
        # Run the default test
        success = test_gemini_api(args.num_questions, args.model)
    
    if success:
        print("\n✅ Gemini API test for QA pair generation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Gemini API test for QA pair generation failed!")
        sys.exit(1)
