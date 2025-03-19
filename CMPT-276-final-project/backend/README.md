# Factify Backend

Factify is a real-time fact-checking tool that verifies claims by retrieving evidence from academic sources and using AI to analyze the credibility of statements.

## Features

- **Claim Decomposition**: Breaks complex claims into verifiable sub-claims
- **Evidence Retrieval**: Finds relevant academic papers from OpenAlex
- **Confidence Scoring**: Evaluates evidence and assigns confidence scores to claims
- **Citation Timeline**: Tracks how papers' relevance changes over time
- **Author Credibility**: Assesses academic reputation of sources
- **Social Context**: Provides explanations about the broader implications of claims

## Tech Stack

- **Backend Framework**: Flask
- **External APIs**:
  - OpenAlex API for academic paper retrieval
  - Google's Gemini API for AI-based analysis

## Setup Instructions

### Prerequisites

- Python 3.9+ installed
- Google API key for Gemini (or another LLM)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/factify.git
   cd factify
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```

5. Edit the `.env` file and add your API keys and configuration.

### Running the Application

Run the Flask application:
```
python app.py
```

The API will be available at `http://localhost:8080`.

For production deployment, use Gunicorn:
```
gunicorn app:app
```

## API Documentation

### Health Check

- **GET** `/health`
  - Returns status of the API

### Claim Verification

- **POST** `/api/verification/claim`
  - Verifies a claim using academic evidence
  - Request Body: `{ "claim": "Your claim to verify" }`

### Evidence Retrieval

- **GET** `/api/evidence/search`
  - Searches for academic papers
  - Query Parameters: 
    - `query`: Search term
    - `page` (optional): Page number, default: 1
    - `per_page` (optional): Results per page, default: 10

- **GET** `/api/evidence/work/{id}`
  - Retrieves details of a specific paper
  - Path Parameter: `id`: OpenAlex work ID

- **GET** `/api/evidence/work/{id}/citations`
  - Retrieves citation timeline for a paper
  - Path Parameter: `id`: OpenAlex work ID

### Author Credibility

- **GET** `/api/verification/author/{id}/credibility`
  - Assesses the credibility of an author
  - Path Parameter: `id`: OpenAlex author ID

## Environment Variables

- `SECRET_KEY`: Flask secret key
- `FLASK_DEBUG`: Set to 'True' for development
- `GOOGLE_API_KEY`: API key for Google's Gemini
- `OPENALEX_EMAIL`: Email to use with OpenAlex's polite pool
- `MAX_PAPERS_TO_RETRIEVE`: Maximum number of papers to retrieve per query
- `TOP_PAPERS_TO_EVALUATE`: Number of top papers to use for evaluation
- `CONFIDENCE_THRESHOLD`: Threshold for determining verdict

## License

This project is licensed under the MIT License - see the LICENSE file for details.