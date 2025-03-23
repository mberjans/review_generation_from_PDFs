# AI Literature Review Generator

This tool automatically generates comprehensive literature reviews from academic PDF papers using AI (OpenAI or Google Gemini). It processes multiple papers in parallel, creates structured summaries, and synthesizes them into a cohesive literature review.

## Features

- Support for both OpenAI and Google Gemini AI providers
- Parallel processing of multiple PDF papers
- Configurable text analysis limits
- Customizable model parameters
- Automatic fallback from OpenAI to Gemini if errors occur
- APA-style citation generation
- Structured output in Markdown format

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- API keys for OpenAI and/or Google Gemini
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
pip install openai google-generative-ai python-dotenv PyPDF2 pydantic tqdm tenacity
```

4. Create a `.env` file in the project root directory with your API keys and optional configurations:
```env
# Required (at least one of these)
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Optional OpenAI configurations
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=custom_url  # Optional, for OpenAI-compatible APIs
OPENAI_MAX_TOKENS=3000
OPENAI_TEMPERATURE=0.7

# Optional Gemini configurations
GEMINI_MODEL=gemini-pro
GEMINI_TEMPERATURE=0.7
GEMINI_MAX_TOKENS=3000
```

5. Create a `PDF` folder in the project root directory and place your PDF papers there:
```bash
mkdir PDF
# Copy your PDF files into the PDF folder
```

## Usage

### Basic Usage

1. Process papers using OpenAI (default):
```bash
python main.py
```

2. Process papers using Google Gemini:
```bash
python main.py --provider gemini
```

### Advanced Usage

1. Customize text analysis limit (characters per paper):
```bash
python main.py --text-limit 10000
```

2. Use specific OpenAI model with custom settings:
```bash
python main.py --provider openai \
    --openai-model gpt-4-turbo-preview \
    --openai-temperature 0.8 \
    --openai-max-tokens 4000
```

3. Use custom OpenAI-compatible API endpoint:
```bash
python main.py --provider openai --openai-base-url https://your-custom-endpoint.com/v1
```

4. Use Gemini with custom settings:
```bash
python main.py --provider gemini \
    --gemini-model gemini-pro \
    --gemini-temperature 0.9 \
    --gemini-max-tokens 4000
```

### All Available Command-line Arguments

```
--provider {openai,gemini}  The AI provider to use (default: openai)
--text-limit INT           Character limit for initial text analysis (default: 6000)

# OpenAI-specific arguments
--openai-model STR        OpenAI model to use (default: from env or gpt-4)
--openai-base-url STR     Custom OpenAI-compatible API base URL
--openai-temperature FLOAT OpenAI temperature (default: from env or 0.7)
--openai-max-tokens INT   OpenAI max tokens (default: from env or 3000)

# Gemini-specific arguments
--gemini-model STR        Gemini model to use (default: from env or gemini-pro)
--gemini-temperature FLOAT Gemini temperature (default: from env or 0.7)
--gemini-max-tokens INT   Gemini max output tokens (default: from env or 3000)
```

## Output

The script generates a Markdown file with the following naming convention:
```
literature_review_[provider]_[timestamp].md
```

The output file contains:
1. Literature review metadata (provider, model, settings)
2. Comprehensive literature review with structured sections
3. List of reviewed papers with APA-style citations

## Troubleshooting

1. **PDF Folder Not Found**
   - Ensure there's a folder named `PDF` in the project root directory
   - Ensure the folder contains PDF files

2. **API Key Errors**
   - Check that your `.env` file exists and contains valid API keys
   - For OpenAI: Verify your API key has sufficient credits
   - For Gemini: Ensure your API key has the necessary permissions

3. **Model-Specific Errors**
   - OpenAI: Check if the specified model is available for your API key
   - Gemini: Verify the model name is correct and available in your region

4. **PDF Processing Errors**
   - Ensure PDFs are not password-protected
   - Verify PDFs are readable and not corrupted
   - Check if PDFs contain extractable text (not scanned images)

## Limitations

- Text extraction is limited to the specified character limit per paper
- PDF files must contain extractable text
- API rate limits may affect processing speed
- Costs depend on API usage and selected models

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[Specify your license here] 