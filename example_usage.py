import os
from dotenv import load_dotenv
from provider_fallback import get_response, call_litellm_with_fallback

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    """Example of using the provider fallback mechanism."""
    print("Example 1: Basic usage with default provider order")
    prompt = "Explain quantum computing in simple terms."
    response = get_response(prompt)
    print(f"\nResponse:\n{response}\n")
    
    print("-" * 80)
    
    print("Example 2: Custom provider order")
    prompt = "What are the major challenges in artificial intelligence today?"
    
    # Custom order: try OpenRouter first, then Gemini, then others
    custom_order = ["openrouter", "gemini"]
    
    full_response = call_litellm_with_fallback(
        prompt=prompt,
        max_tokens=2000,
        temperature=0.8,
        custom_provider_order=custom_order
    )
    
    print(f"\nProvider: {full_response['provider']}")
    print(f"Model: {full_response['model']}")
    print(f"Response:\n{full_response['content']}\n")

if __name__ == "__main__":
    main() 