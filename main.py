import os
import PyPDF2
import openai
import json
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
import unicodedata
import re
import argparse
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exceptions for provider fallback
class RateLimitException(Exception):
    """Exception raised when a provider rate limit is encountered."""
    pass

class ProviderUnavailableException(Exception):
    """Exception raised when a provider is unavailable."""
    pass

class ApiKeyMissingException(Exception):
    """Exception raised when an API key is missing."""
    pass

class ProviderError(Exception):
    """Generic exception for provider errors."""
    pass

class PaperSummary(BaseModel):
    title: str
    authors: List[str]
    year: int
    research_question: str
    theoretical_framework: str
    methodology: str
    main_arguments: List[str]
    findings: str
    significance: str
    limitations: str
    future_research: str

def clean_text(text: str) -> str:
    """Clean and normalize text to handle special characters."""
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # Remove non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def load_providers_config(config_path: str = "providers_config.json") -> List[Dict[str, str]]:
    """Load the provider configuration from the JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get("providers", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading provider config: {str(e)}")
        return []

def check_api_key_present(api_key_env: str) -> bool:
    """
    Check if an API key is present in the environment variables.
    
    Args:
        api_key_env: The name of the environment variable to check
        
    Returns:
        bool: True if the API key is present and not empty, False otherwise
    """
    api_key = os.environ.get(api_key_env)
    return api_key is not None and api_key.strip() != ""

@retry(
    retry=retry_if_exception_type((RateLimitException, ApiKeyMissingException)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=1, max=60)
)
def call_provider_with_fallback(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    max_tokens: int = 3000,
    temperature: float = 0.7,
    custom_provider_order: List[str] = None,
    provider_config_path: str = "providers_config.json",
    json_mode: bool = False
) -> Dict[str, Any]:
    """
    Call AI providers with fallback if one fails.
    
    Args:
        prompt: The user prompt to send to the model
        system_message: System message for chat models
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation (0.0 to 1.0)
        custom_provider_order: Optional custom order of provider names to try
        provider_config_path: Path to the providers configuration JSON file
        json_mode: Whether to request response in JSON format
        
    Returns:
        Dict containing the response from the successful provider
    """
    try:
        # Import litellm here to avoid global import issues
        import litellm
        providers = load_providers_config(provider_config_path)
        
        if not providers:
            raise ValueError("No providers configured. Please check your configuration file.")
        
        # If custom order provided, reorder providers accordingly
        if custom_provider_order:
            # Create a mapping of provider names to their configurations
            provider_map = {p["name"]: p for p in providers}
            
            # Create a new ordered list based on custom order
            ordered_providers = []
            for provider_name in custom_provider_order:
                if provider_name in provider_map:
                    ordered_providers.append(provider_map[provider_name])
            
            # Add any remaining providers not in the custom order at the end
            for provider in providers:
                if provider["name"] not in custom_provider_order:
                    ordered_providers.append(provider)
                    
            providers = ordered_providers
        
        # Track errors for detailed reporting
        errors = {}
        available_providers = []
        unavailable_providers = []
        
        # First, check which providers have API keys available
        for provider in providers:
            provider_name = provider["name"]
            api_key_env = provider["api_key_env"]
            
            if check_api_key_present(api_key_env):
                available_providers.append(provider)
            else:
                unavailable_providers.append(provider_name)
                errors[provider_name] = f"API key not configured ({api_key_env})"
        
        if unavailable_providers:
            logger.info(f"Skipping providers with missing API keys: {', '.join(unavailable_providers)}")
            
        if not available_providers:
            missing_keys = [f"{p['name']} ({p['api_key_env']})" for p in providers]
            raise ApiKeyMissingException(
                f"No API keys found for any provider. Please set at least one of: {', '.join(missing_keys)}"
            )
        
        logger.info(f"Attempting to use providers in order: {[p['name'] for p in available_providers]}")
        
        # Now try each provider that has an API key
        for provider in available_providers:
            provider_name = provider["name"]
            model = provider["default_model"]
            api_key_env = provider["api_key_env"]
            api_key = os.environ.get(api_key_env)
            
            try:
                logger.info(f"Trying provider: {provider_name} with model: {model}")
                
                # Set the API key for the provider
                if provider_name == "openai":
                    litellm.openai_api_key = api_key
                elif provider_name == "anthropic":
                    litellm.anthropic_api_key = api_key
                elif provider_name == "gemini":
                    litellm.google_api_key = api_key
                elif provider_name == "openrouter":
                    litellm.openrouter_api_key = api_key
                elif provider_name == "deepseek":
                    litellm.deepseek_api_key = api_key
                elif provider_name == "groq":
                    litellm.groq_api_key = api_key
                elif provider_name == "mistral":
                    litellm.mistral_api_key = api_key
                else:
                    # For other providers, dynamically set the API key
                    setattr(litellm, f"{provider_name}_api_key", api_key)
                
                # Call the model
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                
                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                
                # Add JSON mode for providers that support it
                if json_mode:
                    # OpenAI and compatible providers
                    if provider_name in ["openai", "openrouter", "anthropic", "azure", "groq"]:
                        completion_kwargs["response_format"] = {"type": "json_object"}
                
                response = litellm.completion(**completion_kwargs)
                
                logger.info(f"Successfully received response from {provider_name}")
                return {
                    "content": response.choices[0].message.content,
                    "provider": provider_name,
                    "model": model
                }
                
            except Exception as e:
                error_msg = str(e)
                errors[provider_name] = error_msg
                logger.warning(f"Error with provider {provider_name}: {error_msg}")
                
                # Check for rate limit or availability errors to determine if we should retry or continue
                if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    logger.info(f"Rate limit hit with {provider_name}, trying next provider")
                    continue  # Try the next provider
                elif "unavailable" in error_msg.lower() or "service unavailable" in error_msg.lower():
                    logger.info(f"Service unavailable for {provider_name}, trying next provider")
                    continue  # Try the next provider
                else:
                    # For other errors, try to determine if we should continue or abort
                    if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                        # Authentication errors - try next provider
                        logger.info(f"Authentication error with {provider_name}, trying next provider")
                        continue
                    elif "bad request" in error_msg.lower() or "invalid request" in error_msg.lower():
                        # Request format errors - might be the same for all providers, but try next anyway
                        logger.info(f"Bad request error with {provider_name}, trying next provider")
                        continue
                    else:
                        # Unexpected error, but try next provider
                        logger.info(f"Unexpected error with {provider_name}, trying next provider")
                        continue
        
        # If we've tried all providers and none worked
        error_details = "\n".join([f"{k}: {v}" for k, v in errors.items()])
        raise ProviderError(f"All providers failed. Details:\n{error_details}")
        
    except ImportError:
        logger.error("litellm not installed. Install using: pip install litellm")
        raise ImportError("litellm not installed. Install using: pip install litellm")

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def analyze_pdf(text: str, filename: str, text_limit: int = 6000) -> PaperSummary:
    """Analyze the content of a PDF and generate a structured summary."""
    prompt = f"""Analyze the following academic paper and provide a detailed summary in JSON format:

    Filename: {filename}
    Text: {text[:text_limit]}  # Limit text to {text_limit} characters

    Provide the summary in a structured JSON format with the following fields:
    - title: string
    - authors: array of strings
    - year: integer
    - research_question: string
    - theoretical_framework: string
    - methodology: string
    - main_arguments: array of strings
    - findings: string
    - significance: string
    - limitations: string
    - future_research: string"""

    try:
        system_message = "You are a helpful assistant that provides comprehensive academic summaries in JSON format."
        response = call_provider_with_fallback(
            prompt=prompt,
            system_message=system_message,
            max_tokens=1000,
            temperature=0.7,
            json_mode=True
        )
        
        logger.info(f"Analysis of {filename} completed using {response['provider']} with model {response['model']}")
        
        # Parse the response content as JSON and create PaperSummary
        summary = PaperSummary.parse_raw(response["content"])
        return summary
    except Exception as e:
        logger.error(f"Error analyzing PDF {filename}: {str(e)}")
        raise

def process_pdf(file_path: str, text_limit: int = 6000) -> PaperSummary:
    """Process a single PDF file."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing: {filename}")
    text = extract_text_from_pdf(file_path)
    return analyze_pdf(text, filename, text_limit)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def synthesize_reviews(summaries: List[PaperSummary], word_limit: int = 2500) -> str:
    """Synthesize multiple paper summaries into a comprehensive literature review."""
    prompt = f"""Create a comprehensive literature review based on the following paper summaries. 
    Focus on synthesizing information, comparing and contrasting key arguments, methodologies, and significance of findings. 
    Highlight any contradictions, agreements, or trends between authors. 
    Discuss the evolution of ideas and methodologies in the field.
    Identify gaps in the current research and suggest future research directions.
    Keep the review under {word_limit} words.

    Summaries: {[summary.dict() for summary in summaries]}

    Structure the review as follows:
    1. Introduction
    2. Theoretical Frameworks
    3. Methodological Approaches
    4. Synthesis of Main Arguments and Findings
    5. Significance and Implications
    6. Gaps and Future Research Directions
    7. Conclusion"""

    try:
        system_message = "You are a helpful assistant that creates comprehensive, well-structured literature reviews."
        response = call_provider_with_fallback(
            prompt=prompt,
            system_message=system_message,
            max_tokens=3000,
            temperature=0.7
        )
        
        logger.info(f"Literature review synthesis completed using {response['provider']} with model {response['model']}")
        return response["content"]
    except Exception as e:
        logger.error(f"Error synthesizing literature review: {str(e)}")
        raise

def create_apa_citation(summary: PaperSummary) -> str:
    """Create an APA 7th edition style citation for a paper."""
    # Handle case where there are no authors
    if not summary.authors:
        return f"Unknown. ({summary.year}). {summary.title}."

    # Format authors: Last name, First initial. for all authors
    formatted_authors = []
    for author in summary.authors:
        parts = author.split()
        if len(parts) > 1:
            last_name = ' '.join(parts[-2:]) if parts[-2].lower() in ['van', 'von', 'de', 'du'] else parts[-1]
            initials = '. '.join(name[0].upper() + '.' for name in parts[:-1] if name.lower() not in ['van', 'von', 'de', 'du'])
            formatted_authors.append(f"{last_name}, {initials}")
        else:
            formatted_authors.append(author)
    
    # Join authors
    if len(formatted_authors) == 1:
        authors_string = formatted_authors[0]
    elif len(formatted_authors) == 2:
        authors_string = f"{formatted_authors[0]} & {formatted_authors[1]}"
    elif len(formatted_authors) > 2:
        authors_string = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    else:
        authors_string = "Unknown"
    
    # Capitalize only the first word and proper nouns in the title
    title_words = summary.title.split()
    title = ' '.join([word.capitalize() if i == 0 or word.lower() not in ['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of'] else word.lower() for i, word in enumerate(title_words)])
    
    # Remove any trailing period from the title
    if title.endswith('.'):
        title = title[:-1]
    
    return f"{authors_string} ({summary.year}). {title}."

def create_paper_list(summaries: List[PaperSummary]) -> str:
    """Create a formatted list of reviewed papers with APA citations."""
    paper_list = "## List of Reviewed Papers\n\n"
    for summary in summaries:
        try:
            citation = create_apa_citation(summary)
            paper_list += f"- {citation}\n"
        except Exception as e:
            logger.error(f"Error creating citation for paper: {summary.title}. Error: {str(e)}")
            paper_list += f"- Error in citation: {summary.title}\n"
    return paper_list

def find_pdf_folder():
    """Find the 'PDF' folder in the same directory as the script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, 'PDF')
    if os.path.isdir(pdf_folder):
        return pdf_folder
    else:
        raise FileNotFoundError("PDF folder not found in the script directory.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate literature reviews from PDF papers.')
    parser.add_argument('--individual-summary-length', type=int, default=6000,
                      help='Character limit for initial text analysis per paper (default: 6000)')
    parser.add_argument('--final-review-length', type=int, default=7000,
                      help='Word limit for the final literature review (default: 7000)')
    parser.add_argument('--custom-provider-order', type=str, nargs='+',
                      help='Custom order of providers to try (e.g., "gemini openai anthropic")')
    return parser.parse_args()

def main():
    try:
        # Load environment variables from .env file
        load_dotenv()
        
        args = parse_args()
        pdf_folder = find_pdf_folder()
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.error("No PDF files found in the PDF folder. Exiting.")
            return
        
        # Configure custom provider order if specified
        custom_provider_order = args.custom_provider_order if args.custom_provider_order else None
        if custom_provider_order:
            logger.info(f"Using custom provider order: {', '.join(custom_provider_order)}")
        
        summaries = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_pdf, os.path.join(pdf_folder, pdf), args.individual_summary_length) 
                      for pdf in pdf_files]
            for future in tqdm(as_completed(futures), total=len(pdf_files), desc="Analyzing PDFs"):
                try:
                    summary = future.result()
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}")

        if not summaries:
            logger.error("No papers were successfully processed. Exiting.")
            return

        logger.info("Synthesizing literature review...")
        literature_review = synthesize_reviews(summaries, args.final_review_length)
        
        paper_list = create_paper_list(summaries)
        
        output_filename = f'literature_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(output_filename, 'w') as f:
            f.write(literature_review)
            f.write("\n\n")
            f.write(paper_list)
        
        logger.info(f"Literature review completed and saved as {output_filename}")
    
    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
