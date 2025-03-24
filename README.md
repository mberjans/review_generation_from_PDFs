# AI Literature Review Generator with Provider Fallback

This tool automatically generates comprehensive literature reviews from academic PDF papers using AI. It processes multiple papers in parallel, creates structured summaries, and synthesizes them into a cohesive literature review.

## New Feature: Multi-Provider Fallback

The tool now supports multiple AI providers with automatic fallback capabilities. If one provider becomes unavailable or rate-limited, the system automatically tries the next available provider in the configured order.

### Supported Providers

The system supports the following providers (in default order):

1. Google Gemini
2. OpenRouter (with DeepSeek R1 models)
3. DeepSeek
4. Anthropic (Claude models)
5. Groq
6. Mistral AI
7. OpenAI

You can customize the provider order using command-line arguments.

## Features

- Support for multiple AI providers with automatic fallback
- Parallel processing of multiple PDF papers
- Configurable text analysis limits
- Customizable model parameters
- APA-style citation generation
- Structured output in Markdown format

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- API keys for at least one of the supported AI providers
- PDF files to analyze

## Installation

1. Clone the repository or download the source code:
```bash
git clone <repository-url>
cd AI-Literature-Review-Generator
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory with your API keys:
```bash
cp .env.template .env
```

5. Edit the `.env` file to add your API keys for the providers you want to use:
```env
# You only need API keys for the providers you want to use
# The system will automatically try providers in order, skipping any without API keys
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
MISTRAL_API_KEY=your_mistral_key_here
OPENAI_API_KEY=your_openai_key_here
```

6. Create a `PDF` folder in the project root directory and place your PDF papers there:
```bash
mkdir PDF
# Copy your PDF files into the PDF folder
```

## Usage

### Basic Usage

```bash
python main.py
```

This will use the default settings:
- 6000 characters per paper for analysis
- 7000 words for the final review
- Default provider order from providers_config.json

### Advanced Usage

1. Customize individual summary and final review lengths:
```bash
python main.py --individual-summary-length 10000 --final-review-length 5000
```

2. Specify a custom provider order:
```bash
python main.py --custom-provider-order openrouter gemini anthropic
```
This will try OpenRouter first, then Gemini, then Anthropic, and finally any remaining providers.

### All Available Command-line Arguments

```
--individual-summary-length INT  Character limit for initial text analysis per paper (default: 6000)
--final-review-length INT        Word limit for the final literature review (default: 7000)
--custom-provider-order STR [STR ...]  Custom order of providers to try (e.g., "gemini openai anthropic")
```

## Output

The script generates a Markdown file with the following naming convention:
```
literature_review_[timestamp].md
```

The output file contains:
1. Comprehensive literature review with structured sections
2. List of reviewed papers with APA-style citations

## Configuring Provider Order

The default provider order is configured in the `providers_config.json` file. You can edit this file to permanently change the default order or add new providers.

Example structure:
```json
{
  "providers": [
    {
      "name": "gemini",
      "default_model": "gemini-pro",
      "api_key_env": "GEMINI_API_KEY"
    },
    {
      "name": "openrouter",
      "default_model": "openrouter/deepseek/deepseek-r1-distill-llama-8b",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    ...
  ]
}
```

Each provider has:
- `name`: The provider identifier
- `default_model`: The model to use from this provider
- `api_key_env`: The environment variable name that stores the API key

## Troubleshooting

1. **Missing API Keys**
   - The system will automatically skip providers with missing API keys
   - Ensure you have at least one provider's API key configured in your .env file

2. **PDF Folder Not Found**
   - Ensure there's a folder named `PDF` in the project root directory
   - Ensure the folder contains PDF files

3. **Provider Errors**
   - If one provider fails, the system will automatically try the next one
   - Check the logs for detailed error information

## Limitations

- Text extraction is limited to the specified character limit per paper
- PDF files must contain extractable text
- API rate limits may affect processing speed
- Costs depend on API usage and selected models

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[Specify your license here]

# AI Provider Fallback Mechanism

This project implements a fallback mechanism for AI providers using the LiteLLM library. If one provider becomes unavailable (e.g., due to rate limiting), the system automatically tries the next provider in the configured order.

## Features

- Configurable provider order through a JSON file
- Automatic fallback if a provider is unavailable or rate-limited
- Automatic detection of missing API keys with intelligent provider selection
- Support for multiple AI models (Gemini, OpenRouter, DeepSeek, etc.)
- Detailed error logging
- Simple API for integration into existing projects

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your API keys:

```bash
cp .env.template .env
```

3. Edit the `.env` file to add your API keys for the providers you want to use.

> **Note:** You don't need to add API keys for all providers. The system will automatically check which API keys are available and use only those providers. It will skip any providers with missing API keys.

## Usage

### Basic Usage

```python
from provider_fallback import get_response

# Get a response with default settings (using providers in the order specified in providers_config.json)
response = get_response("Your prompt here")
print(response)
```

### Advanced Usage

```python
from provider_fallback import call_litellm_with_fallback

# Use a custom provider order
custom_order = ["openrouter", "gemini", "anthropic"]

result = call_litellm_with_fallback(
    prompt="Your prompt here",
    max_tokens=2000,
    temperature=0.7,
    custom_provider_order=custom_order
)

print(f"Provider: {result['provider']}")
print(f"Model: {result['model']}")
print(f"Response: {result['content']}")
```

### Running the Example

```bash
python example_usage.py
```

## Customizing Provider Order

The default provider order is specified in the `providers_config.json` file. You can modify this file to change the default order or add new providers.

Example structure:

```json
{
  "providers": [
    {
      "name": "gemini",
      "default_model": "gemini-pro",
      "api_key_env": "GEMINI_API_KEY"
    },
    {
      "name": "openrouter",
      "default_model": "openrouter/deepseek/deepseek-r1-distill-llama-8b",
      "api_key_env": "OPENROUTER_API_KEY"
    },
    ...
  ]
}
```

You can also override the provider order at runtime using the `custom_provider_order` parameter.

## Error Handling

The system will automatically try all configured providers before giving up. If all providers fail, an error message will be returned explaining the reason for each failure.

## Requirements

- Python 3.7+
- LiteLLM
- Tenacity
- Requests
- python-dotenv 