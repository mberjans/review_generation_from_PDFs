#!/usr/bin/env python3
"""
API Key Tester for AI Literature Review Generator

This script tests the API keys for all configured providers to verify 
they are correctly set up and functioning.
"""

import os
import json
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_providers_config(config_path="providers_config.json"):
    """Load the provider configuration from the JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get("providers", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading provider config: {str(e)}")
        return []

def test_openai_key():
    """Test the OpenAI API key."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or key == "your_openai_key_here":
        logger.warning("OpenAI API key not found or is set to default value")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        models = client.models.list()
        logger.info("✅ OpenAI API key is valid")
        return True
    except Exception as e:
        logger.error(f"❌ OpenAI API key error: {str(e)}")
        return False

def test_anthropic_key():
    """Test the Anthropic API key."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or key == "your_anthropic_key_here":
        logger.warning("Anthropic API key not found or is set to default value")
        return False
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        models = client.models.list()
        logger.info("✅ Anthropic API key is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Anthropic API key error: {str(e)}")
        return False

def test_gemini_key():
    """Test the Google Gemini API key."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key == "your_gemini_key_here":
        logger.warning("Gemini API key not found or is set to default value")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        models = genai.list_models()
        logger.info("✅ Gemini API key is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini API key error: {str(e)}")
        return False

def test_mistral_key():
    """Test the Mistral API key."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key or key == "your_mistral_key_here":
        logger.warning("Mistral API key not found or is set to default value")
        return False
    
    try:
        import mistralai.client
        client = mistralai.client.MistralClient(api_key=key)
        models = client.list_models()
        logger.info("✅ Mistral API key is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Mistral API key error: {str(e)}")
        return False

def test_groq_key():
    """Test the Groq API key."""
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_groq_key_here":
        logger.warning("Groq API key not found or is set to default value")
        return False
    
    try:
        import groq
        client = groq.Groq(api_key=key)
        models = client.models.list()
        logger.info("✅ Groq API key is valid")
        return True
    except Exception as e:
        logger.error(f"❌ Groq API key error: {str(e)}")
        return False

def test_openrouter_key():
    """Test the OpenRouter API key."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key == "your_openrouter_key_here":
        logger.warning("OpenRouter API key not found or is set to default value")
        return False
    
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://ai-literature-review-generator.local", 
            "X-Title": "AI Literature Review Generator"
        }
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)
        if response.status_code == 200:
            logger.info("✅ OpenRouter API key is valid")
            return True
        else:
            logger.error(f"❌ OpenRouter API key error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ OpenRouter API key error: {str(e)}")
        return False

def test_deepseek_key():
    """Test the DeepSeek API key."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key or key == "your_deepseek_key_here":
        logger.warning("DeepSeek API key not found or is set to default value")
        return False
    
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        response = requests.get("https://api.deepseek.com/models", headers=headers)
        if response.status_code == 200:
            logger.info("✅ DeepSeek API key is valid")
            return True
        else:
            logger.error(f"❌ DeepSeek API key error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ DeepSeek API key error: {str(e)}")
        return False

def main():
    """Main function to test all API keys."""
    load_dotenv()
    
    # Dictionary mapping provider names to test functions
    test_functions = {
        "openai": test_openai_key,
        "anthropic": test_anthropic_key,
        "gemini": test_gemini_key,
        "mistral": test_mistral_key,
        "groq": test_groq_key,
        "openrouter": test_openrouter_key,
        "deepseek": test_deepseek_key
    }
    
    # Get the configured providers
    providers = load_providers_config()
    
    # Track valid providers
    valid_providers = []
    
    print("\n=== Testing API Keys ===\n")
    
    # Test each provider
    for provider in providers:
        provider_name = provider["name"]
        api_key_env = provider["api_key_env"]
        
        # Check if we have a test function for this provider
        if provider_name in test_functions:
            print(f"Testing {provider_name.upper()} API key...")
            if test_functions[provider_name]():
                valid_providers.append(provider_name)
            print()
        else:
            logger.warning(f"No test function for provider: {provider_name}")
    
    # Print summary
    print("\n=== API Key Test Summary ===")
    if valid_providers:
        print(f"Valid API keys found for: {', '.join(valid_providers)}")
    else:
        print("No valid API keys found. Please check your .env file.")
    
    print("\nRecommended provider order based on available keys:")
    if valid_providers:
        print(f"--custom-provider-order {' '.join(valid_providers)}")
    else:
        print("No valid providers to recommend.")

if __name__ == "__main__":
    main() 