# Factify Backend

Factify is a real-time fact-checking tool that verifies claims by retrieving evidence from academic sources and using AI to analyze the credibility of statements.

## Features

- **Claim Decomposition**: Breaks complex claims into verifiable sub-claims
- **Evidence Retrieval**: Finds relevant academic papers from OpenAlex, CrossRef, and Semantic Scholar
- **Confidence Scoring**: Evaluates evidence and assigns confidence scores to claims
- **Citation Timeline**: Tracks how papers' relevance changes over time
- **Author Credibility**: Assesses academic reputation of sources

## Tech Stack

- **Backend Framework**: Flask
- **External APIs**:
  - OpenAlex API for academic paper retrieval
  - CrossRef API for additional academic sources
  - Semantic Scholar API for research papers
  - Google's Gemini API for AI-based analysis
- **Vector Database**: FAISS for semantic search
- **Embedding Model**: SentenceTransformer

## Complete Setup Instructions

### Prerequisites

- Python 3.9+ installed
- Node.js and npm for the frontend
- Google API key for Gemini

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/factify.git
   cd factify
   ```

2. Create a virtual environment and activate it:
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory:
   ```
   touch .env
   ```

5. Add the following configuration to your `.env` file:
   ```
# Flask configuration
SECRET_KEY=rushik-behal
FLASK_DEBUG=True
PORT=8080

# Google API configuration for Gemini
GOOGLE_API_KEY=AIzaSyBaRn2Bxx0gtKJGobs--Jw4zO5Kc5Avb5Q

# OpenAlex configuration
OPENALEX_EMAIL=rba137@sfu.ca

# Database configuration
DATABASE_URL=sqlite:///factify_rag.db

# Embedding and RAG configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_EVIDENCE_TO_RETRIEVE=800
MAX_EVIDENCE_TO_STORE=800
RAG_TOP_K=55

# Old config
MAX_PAPERS_TO_RETRIEVE=600
TOP_PAPERS_TO_EVALUATE=25
CONFIDENCE_THRESHOLD=0.75 
   ```

### Frontend Setup

1. Navigate to the project root directory and install frontend dependencies:
   ```
   cd ..  # Return to project root from backend directory
   npm install
   ```

### Running the Application

1. Start the backend server (from the backend directory):
   ```
   cd backend  # Skip if already in backend directory
   python app.py
   ```
   The backend API will be available at `http://localhost:8080`.

2. In a new terminal, start the frontend development server (from the project root):
   ```
   npm run dev
   ```
   The frontend will be available at `http://localhost:3000` (or another port if specified).

3. Open your browser and navigate to `http://localhost:3000` to use the application.

### Troubleshooting

- **Missing dependencies**: If you encounter missing module errors, try running `pip install -r requirements.txt` again.
- **CUDA/GPU issues**: The application uses CPU by default. No special configuration is needed for basic usage.
- **API rate limits**: The application implements rate limiting for external APIs. If you encounter 429 errors, wait a few moments and try again.
- **Tokenizers warnings**: You may see warnings about tokenizers parallelism. These can be safely ignored or fixed by setting the environment variable `TOKENIZERS_PARALLELISM=false`.

## API Documentation

### Health Check

- **GET** `/health`
  - Returns status of the API and its dependencies

### Claim Verification

- **POST** `/api/verify_claim`
  - Verifies a claim using RAG-based academic evidence analysis
  - Request Body: `{ "claim": "Your claim to verify" }`
  - Returns verification result, evidence, and confidence score

## Environment Variables

- `SECRET_KEY`: Flask secret key
- `FLASK_DEBUG`: Set to 'True' for development
- `GOOGLE_API_KEY`: API key for Google's Gemini
- `OPENALEX_EMAIL`: Email to use with OpenAlex's polite pool
- `DATABASE_URL`: Database connection string (default: SQLite)
- `EMBEDDING_MODEL`: SentenceTransformer model to use (default: all-MiniLM-L6-v2)
- `MAX_EVIDENCE_TO_RETRIEVE`: Maximum number of papers to retrieve per source
- `MAX_EVIDENCE_TO_STORE`: Maximum number of total papers to store per query
- `RAG_TOP_K`: Number of most relevant chunks to use for analysis

