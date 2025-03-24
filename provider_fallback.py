import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
def call_litellm_with_fallback(
    prompt: str,
    max_tokens: int = 3000,
    temperature: float = 0.7,
    custom_provider_order: List[str] = None,
    provider_config_path: str = "providers_config.json"
) -> Dict[str, Any]:
    """
    Call LiteLLM with multiple providers in order, falling back if one fails.
    
    Args:
        prompt: The user prompt to send to the model
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation (0.0 to 1.0)
        custom_provider_order: Optional custom order of provider names to try
        provider_config_path: Path to the providers configuration JSON file
        
    Returns:
        Dict containing the response from the successful provider
    """
    try:
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
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
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

def get_response(prompt: str, max_tokens: int = 3000, temperature: float = 0.7) -> str:
    """
    Simple wrapper function to get a response with fallback handling.
    
    Args:
        prompt: The text prompt to send
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation (0.0 to 1.0)
        
    Returns:
        The text response from the model
    """
    try:
        result = call_litellm_with_fallback(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return result["content"]
    except ApiKeyMissingException as e:
        logger.error(f"API key missing: {str(e)}")
        return f"Error: No API keys configured. Please add at least one API key to the .env file. ({str(e)})"
    except Exception as e:
        logger.error(f"Failed to get response from any provider: {str(e)}")
        return f"Error: Failed to get a response from any provider. Please try again later. ({str(e)})"

if __name__ == "__main__":
    # Example usage
    response = get_response("Explain how neural networks work in simple terms.")
    print(f"Response: {response}") 