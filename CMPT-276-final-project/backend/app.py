# app.py
import os
import concurrent.futures
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import datetime
import re  # Add this at the top with other imports
import socket  # For DNS resolution test
import numpy as np # Add numpy import back
import math # For log scaling
import xml.etree.ElementTree as ET # Add this for PubMed XML parsing


# Set Tokenizer Parallelism to avoid fork issues (can be 'true' or 'false')
# Setting to 'false' is often safer in web server environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Restrict FAISS threading to avoid resource issues
os.environ["OMP_NUM_THREADS"] = "4"  # Limit OpenMP threads used by FAISS

# --- New Imports ---
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Date, MetaData, Table, text # Import text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
# --- End New Imports ---
# --- New pgvector Import ---
from pgvector.sqlalchemy import Vector
# --- End pgvector Import ---
# --- Import for UniqueViolation ---
from psycopg2 import errors
# --- End Import ---

# Configure logging
logging.basicConfig(level=logging.INFO)
# Add this line to reduce SQLAlchemy logging noise on errors
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Database Config ---
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'factify-dev-key'),
    DEBUG=os.getenv('FLASK_DEBUG', 'False') == 'True',
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY'),
    OPENALEX_EMAIL=os.getenv('OPENALEX_EMAIL', 'rba137@sfu.ca'),
    # --- Database Config ---
    DATABASE_URL=os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/postgres'),
    EMBEDDING_MODEL_API = "models/text-embedding-004", # Using Gemini embedding API model
    # --- THIS VALUE WILL BE DYNAMICALLY UPDATED BASED ON SCHEMA DETECTION ---
    EMBEDDING_DIMENSION = 768, # Default dimension for text-embedding-004
    # Replace single limit with specific API limits
    OPENALEX_MAX_RESULTS=int(os.getenv('OPENALEX_MAX_RESULTS', '200')),
    CROSSREF_MAX_RESULTS=int(os.getenv('CROSSREF_MAX_RESULTS', '1000')),
    SEMANTIC_SCHOLAR_MAX_RESULTS=int(os.getenv('SEMANTIC_SCHOLAR_MAX_RESULTS', '100')),
    PUBMED_MAX_RESULTS=int(os.getenv('PUBMED_MAX_RESULTS', '200')), # Max results for PubMed ESearch
    # Keep these settings
    MAX_EVIDENCE_TO_STORE=int(os.getenv('MAX_EVIDENCE_TO_STORE', '100')), # Reduced default from 800 to 100
    RAG_TOP_K=int(os.getenv('RAG_TOP_K', '20')), # Number of chunks for RAG analysis
)

# --- Initialize Gemini API
gemini_api_key = app.config.get('GOOGLE_API_KEY')
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

    gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using 2.0 Flash as the model.
else:
    print("WARNING: GOOGLE_API_KEY not set. Gemini features will not work.")
    gemini_model = None

# --- New: Database Setup ---
Base = declarative_base()

# Log the database URL (without password)
database_url = app.config['DATABASE_URL']
logger.info(f"Using database connection: {database_url.split('@')[0].split(':')[0]}:****@{database_url.split('@')[1] if '@' in database_url else 'localhost'}")

# Test DNS resolution for database host
try:
    hostname = database_url.split('@')[1].split('/')[0].split(':')[0]
    logger.info(f"Testing DNS resolution for: {hostname}")
    resolved_ip = socket.gethostbyname(hostname)
    logger.info(f"Successfully resolved {hostname} to {resolved_ip}")
except socket.gaierror as e:
    logger.error(f"DNS resolution error for {hostname}: {e}")
    logger.error("Please check your network connection, DNS settings, or if the hostname is correct")
    # Provide helpful suggestions for Supabase issues
    if 'supabase.co' in database_url:
        logger.error("SUPABASE CONNECTION TROUBLESHOOTING:")
        logger.error("1. Check if your DATABASE_URL is correctly formatted.")
        logger.error("2. For direct connections, use: postgresql://postgres:password@db.YOUR-PROJECT-REF.supabase.co:5432/postgres")
        logger.error("3. For pooler connections, use: postgresql://postgres.YOUR-PROJECT-REF:password@aws-0-REGION.pooler.supabase.com:5432/postgres")
        logger.error("4. Ensure your project reference ID in the connection string is correct.")
        logger.error("5. Verify your Supabase database is active in the Supabase dashboard.")
    # Don't exit here, let SQLAlchemy handle the connection error

# Handle potential SSL requirement for PostgreSQL connection
if database_url.startswith('postgresql'):
    # Force SSL mode for Supabase connections
    if 'supabase.co' in database_url:
        logger.info("Supabase connection detected, setting SSL requirements")
        if '?' not in database_url:
            database_url += "?sslmode=require"
        elif 'sslmode=' not in database_url:
            database_url += "&sslmode=require"
        
try:
    # Create the engine with more detailed error handling
    logger.info(f"Creating database engine...")
    engine = create_engine(
        database_url, 
        pool_pre_ping=True,  # Add health checks for connections
        connect_args={
            # Longer timeout for slow connections
            'connect_timeout': 30
        } if database_url.startswith('postgresql') else {}
    )
    
    # Test the connection
    logger.info("Testing database connection...")
    with engine.connect() as connection:
        # Simple query to test connectivity
        connection.execute(text("SELECT 1"))
        logger.info("Database connection test successful!")
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to create database engine or test connection: {e}")
    raise

class Study(Base):
    __tablename__ = "studies"
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, index=True) # Link study to a specific claim request if needed
    doi = Column(String, unique=True, index=True)
    title = Column(Text)
    authors = Column(Text, nullable=True) # Store as JSON string or delimited
    pub_date = Column(String, nullable=True) # Store as string for flexibility
    abstract = Column(Text, nullable=True)
    source_api = Column(String) # 'crossref' or 'openalex'
    retrieved_at = Column(Date, default=datetime.date.today)
    citation_count = Column(Integer, nullable=True, default=0) # Add citation count column
    # --- Add pgvector column ---
    embedding = Column(Vector(app.config['EMBEDDING_DIMENSION']), nullable=True)
    # --- End Add pgvector column ---
    # relevance_score = Column(Float, nullable=True) # Add if calculated

# Create tables if they don't exist (better to use Alembic migrations)
logger.info("Creating database tables if they don't exist...")
try:
    # Make sure the vector extension is available before creating tables
    with engine.connect() as conn:
        try:
             logger.info("Checking if vector extension is enabled...")
             conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
             conn.commit()
             logger.info("Vector extension check complete.")
        except Exception as ext_e:
             logger.warning(f"Could not automatically enable pgvector extension: {ext_e}. Please ensure it is enabled manually in your Supabase dashboard.")

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully!")
except Exception as e:
    logger.error(f"Failed to create database tables: {e}")
    raise

# Modified to work with both SQLite and PostgreSQL
def add_column_if_not_exists():
    try:
        # Get a connection and detect database type
        # This function is now simplified as create_all handles column additions
        # if the model definition changes. We'll primarily use it for the index.
        # We assume PostgreSQL with pgvector. If SQLite is needed, this requires more complex logic.
        with engine.connect() as conn:
             dialect = engine.dialect.name
             logger.info(f"Database dialect detected: {dialect}")
             if dialect == 'postgresql':
                 logger.info("Checking/Adding columns and vector index for PostgreSQL...")
                 # Check/Add citation_count (SQLAlchemy's create_all should handle this, but belt-and-suspenders)
                 try:
                     conn.execute(text("ALTER TABLE studies ADD COLUMN IF NOT EXISTS citation_count INTEGER DEFAULT 0"))
                     conn.commit() # Commit after ALTER TABLE
                     logger.info("Checked/Added citation_count column.")
                 except Exception as e:
                     logger.warning(f"Could not add citation_count column (might already exist or other issue): {e}")
                     conn.rollback()

                 # Explicitly check and add the embedding column if it doesn't exist
                 # Check if the embedding column exists using information_schema
                 column_check_query = text("""
                     SELECT EXISTS (
                         SELECT 1
                         FROM information_schema.columns
                         WHERE table_schema = 'public' -- Adjust if using a different schema
                         AND table_name = 'studies'
                         AND column_name = 'embedding'
                     );
                 """)
                 embedding_column_exists = conn.execute(column_check_query).scalar()

                 if not embedding_column_exists:
                    logger.info("'embedding' column not found. Attempting to add it...")
                    try:
                        # Ensure vector extension is available
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        # Add the column with the correct dimension
                        conn.execute(text(f"ALTER TABLE studies ADD COLUMN embedding vector({app.config['EMBEDDING_DIMENSION']})"))
                        conn.commit()
                        logger.info(f"Successfully added 'embedding' column with dimension {app.config['EMBEDDING_DIMENSION']}.")
                    except Exception as add_col_e:
                        logger.error(f"Failed to add 'embedding' column: {add_col_e}")
                        conn.rollback()
                 else:
                     logger.info("'embedding' column already exists (checked via information_schema).")

                 # Create HNSW index for faster vector similarity search (Recommended for performance)
                 # You can adjust 'm' and 'ef_construction' based on recall/performance needs.
                 # Index types: https://github.com/pgvector/pgvector#indexing
                 # Using cosine distance (<=>) as it's common for sentence embeddings.
                 # Use L2 distance (<->) if your model/task suits it better.
                 # Use inner product (<#>) for max inner product search.
                 index_name = "idx_studies_embedding_cosine"
                 index_check_query = text(f"SELECT 1 FROM pg_indexes WHERE indexname = '{index_name}'")
                 index_exists = conn.execute(index_check_query).scalar()

                 if not index_exists:
                     logger.info(f"Creating HNSW index '{index_name}' on studies(embedding)... This may take time.")
                     try:
                         # Ensure vector extension is loaded in this session
                         conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                         # Create the index using cosine distance
                         # Adjust lists_or_m based on expected data size and performance needs
                         # For HNSW: m=16, ef_construction=64 are common starting points
                         conn.execute(text(f"""
                             CREATE INDEX {index_name} ON studies
                             USING hnsw (embedding vector_cosine_ops)
                             WITH (m = 16, ef_construction = 64)
                         """))
                         conn.commit() # Commit after CREATE INDEX
                         logger.info(f"Successfully created HNSW index '{index_name}'.")
                     except Exception as e:
                         logger.error(f"Failed to create HNSW index '{index_name}': {e}")
                         conn.rollback() # Rollback on error
                 else:
                      logger.info(f"HNSW index '{index_name}' already exists.")
             else:
                 logger.warning(f"Database dialect '{dialect}' detected. Automatic schema migration for vector column/index is only implemented for PostgreSQL.")

        logger.info("Database schema and index check complete.")
    except Exception as e:
        logger.error(f"Error checking or updating database schema/index: {e}")

# Perform the schema check/update
add_column_if_not_exists()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# --- End Database Setup ---

# --- New: Vector Store Service using pgvector ---
class VectorStoreService:
    def __init__(self, db_session):
        self.db = db_session

    def get_embedding_for_text(self, text, task_type="retrieval_document"):
        """Generates embedding for text using Gemini API and adapts to database dimension if needed."""
        if not gemini_api_key:
             logger.error("GOOGLE_API_KEY not set. Cannot generate embeddings.")
             raise ValueError("Gemini API key not configured.")
        if not text:
            logger.warning("Attempted to embed empty text.")
            return None
            
        # Truncate text if too long to avoid API limits (36000 bytes limit observed from error)
        # A safe limit for English text is around 20000 characters
        MAX_CHARS = 20000
        if len(text) > MAX_CHARS:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {MAX_CHARS} chars.")
            text = text[:MAX_CHARS]
            
        try:
            # Use the appropriate task_type for the embedding model
            # 'retrieval_document' for study abstracts/evidence
            # 'retrieval_query' for the user's claim
            result = genai.embed_content(
                model=app.config['EMBEDDING_MODEL_API'],
                content=text,
                task_type=task_type # Specify task type
            )
            
            embedding = result['embedding']  # API returns a list of floats
            
            # Get the expected dimension from app config (set during database detection)
            expected_dimension = app.config['EMBEDDING_DIMENSION']
            actual_dimension = len(embedding)
            
            # If dimensions don't match, adapt the embedding to match the database
            if actual_dimension != expected_dimension:
                logger.info(f"Adapting embedding from dimension {actual_dimension} to {expected_dimension}")
                
                if actual_dimension > expected_dimension:
                    # Truncate: Take only the first expected_dimension elements
                    adapted_embedding = embedding[:expected_dimension]
                else:
                    # Pad: Extend with zeros to reach expected_dimension
                    adapted_embedding = embedding + [0.0] * (expected_dimension - actual_dimension)
                
                return adapted_embedding
            
            return embedding  # Return the original embedding if dimensions match
            
        except Exception as e:
            logger.error(f"Error generating embedding using Gemini API: {e}")
            # Handle specific API errors (rate limits, etc.) if needed
            raise # Re-raise the exception to be handled by the caller

    def find_relevant_studies(self, claim_text, top_k):
        """
        Finds top_k relevant studies using vector similarity search.
        """
        if not gemini_api_key:
             logger.error("Gemini API key not available. Cannot perform vector search.")
             return []

        try:
            # 1. Embed the claim text using the API
            logger.info("Generating embedding for claim using Gemini API...")
            claim_embedding = self.get_embedding_for_text(claim_text, task_type="retrieval_query")
            if not claim_embedding:
                 logger.error("Failed to generate embedding for the claim.")
                 return []

            claim_embedding_np = np.array(claim_embedding).astype('float32')

            # 2. Perform vector similarity search
            logger.info(f"Searching database for top {top_k} studies...")
            relevant_studies = (
                self.db.query(Study)
                .filter(Study.embedding != None)
                .order_by(Study.embedding.cosine_distance(claim_embedding_np)) # Use the numpy array
                .limit(top_k)
                .all()
            )

            if not relevant_studies:
                logger.warning("Vector search returned no relevant studies.")
                return []

            logger.info(f"Retrieved {len(relevant_studies)} relevant studies from DB vector search.")
            return relevant_studies

        except Exception as e:
            logger.error(f"Error during vector search in database: {e}")
            return []

# --- End Vector Store Service ---

def clean_abstract(text):
    """Clean JATS XML tags and other formatting from abstract text."""
    if not text:
        return ""
    
    # Remove JATS XML tags
    text = re.sub(r'</?jats:[^>]+>', '', text)
    # Remove any remaining XML-like tags
    text = re.sub(r'</?[^>]+>', '', text)
    # Remove 'Abstract:' or 'Abstract' at the start
    text = re.sub(r'^(?:Abstract:?\s*)', '', text, flags=re.IGNORECASE)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Modified OpenAlex Service ---
class OpenAlexService:
    BASE_URL = "https://api.openalex.org"

    def __init__(self, email=None):
        self.email = email or app.config.get('OPENALEX_EMAIL')
        # Ensure email is provided for User-Agent politeness
        if not self.email:
            logger.warning("OPENALEX_EMAIL not set, using default. Please set it for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 20 # Increased timeout for larger requests

    def _reconstruct_abstract_from_inverted_index(self, abstract_inverted_index):
        # (Keep the existing implementation - unchanged)
        if not abstract_inverted_index:
            return ""
        max_position = 0
        for positions in abstract_inverted_index.values():
            if positions and max(positions) > max_position:
                max_position = max(positions)
        words = [""] * (max_position + 1)
        for word, positions in abstract_inverted_index.items():
            for position in positions:
                words[position] = word
        return " ".join(words)

    def search_works_by_keyword(self, keywords, per_page=10):
        """Search works using keyword search."""
        # Join keywords for search query if it's a list
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)

        # Use the configured max results per source. Be mindful of potential API limits.
        # OpenAlex official limit seems to be 200. Higher values might cause errors.
        per_page = min(per_page, 200) # Cap at 200 based on OpenAlex docs, was 100

        # Re-add filter for abstracts and citation count (lowered threshold)
        openalex_filter = 'has_abstract:true,cited_by_count:>10' # Lowered threshold from 50 to 10

        params = {
            'search': search_query,
            'per-page': per_page,
            'filter': openalex_filter, # Use the defined filter
            'select': 'id,doi,title,authorships,publication_date,abstract_inverted_index,primary_location,cited_by_count' # Add cited_by_count
        }
        logger.info(f"Querying OpenAlex: {search_query} with per_page={per_page}, filter='{openalex_filter}'")
        try:
            response = requests.get(
                f"{self.BASE_URL}/works",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("OpenAlex rate limit hit, sleeping for 2 seconds.")
                time.sleep(2)
                return self.search_works_by_keyword(keywords, per_page) # Retry
                
            # Handle forbidden errors
            if response.status_code == 403:
                logger.warning("OpenAlex returned 403 Forbidden. Trying with smaller batch size.")
                if per_page > 25:
                    # Retry with a much smaller batch size
                    return self.search_works_by_keyword(keywords, 25)
                else:
                    # If we're already using a small batch size, it's some other issue
                    logger.error(f"OpenAlex 403 Forbidden error even with small batch size: {response.text}")
                    return None

            response.raise_for_status() # Raise HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode OpenAlex JSON response: {e}")
             return None

    def process_results(self, results_json):
        """Processes OpenAlex JSON results into a standardized format."""
        processed = []
        if not results_json or 'results' not in results_json:
            return processed

        for paper in results_json.get('results', []):
            # --- Check for Retraction --- 
            # OpenAlex might use `is_retracted` or status fields, adjust based on observed data
            # Assuming a field like `is_retracted` or `publication_status` exists
            if paper.get('is_retracted', False) or paper.get('publication_status', '').lower() == 'retracted':
                 doi = paper.get('doi', 'Unknown DOI')
                 logger.warning(f"Skipping retracted OpenAlex article: DOI {doi}")
                 continue # Skip this article
            # --- End Check for Retraction ---
            
            # Extract abstract
            abstract = ""
            if paper.get('abstract_inverted_index'):
                try:
                    abstract = self._reconstruct_abstract_from_inverted_index(paper.get('abstract_inverted_index'))
                except Exception as e:
                    logger.warning(f"Error reconstructing abstract for OpenAlex ID {paper.get('id')}: {e}")

            # Skip if abstract is empty or too short
            if not abstract or len(abstract) < 50:
                continue

            # Extract authors
            authors = ", ".join([a.get('author', {}).get('display_name', '') for a in paper.get('authorships', []) if a.get('author')])
            
            # Extract citation count
            citation_count = paper.get('cited_by_count', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": paper.get('doi'),
                "title": paper.get('title', 'Untitled'),
                "authors": authors,
                "pub_date": paper.get('publication_date'),
                "abstract": abstract,
                "source_api": "openalex",
                "citation_count": citation_count
            })
        return processed
# --- End OpenAlex Service ---

# --- New: CrossRef Service ---
class CrossRefService:
    BASE_URL = "https://api.crossref.org"

    def __init__(self, email=None):
        self.email = email or app.config.get('OPENALEX_EMAIL') # Reuse email for politeness
        if not self.email:
            logger.warning("Email not set for CrossRef, using default. Set OPENALEX_EMAIL for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 30 # Increased timeout for potentially larger/slower requests

    def search_works_by_keyword(self, keywords, rows=10):
        """Search CrossRef works using keyword query."""
        # Join keywords for search query if it's a list
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)

        # Use the configured max results per source. CrossRef limit is 1000 but often unstable.
        # Capping at a safer limit like 200 might be wise, adjust if needed.
        rows = min(rows, 200) # Cap at 200, was 100. Can be increased further but test stability.

        params = {
            'query': search_query,
            'rows': rows,
            'select': 'DOI,title,author,abstract,published-print,published-online,created,is-referenced-by-count'  # Added citation count field
        }
        logger.info(f"Querying CrossRef using 'query' param: {search_query} with rows={rows} (Abstracts will be filtered post-retrieval)")
        try:
            response = requests.get(
                f"{self.BASE_URL}/works",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
            
            # Handle rate limiting or errors
            if response.status_code == 429:
                logger.warning("CrossRef rate limit hit, sleeping for 2 seconds.")
                time.sleep(2)
                return self.search_works_by_keyword(keywords, rows)
                
            # Handle other error responses
            if response.status_code >= 400:
                logger.warning(f"CrossRef returned error {response.status_code}. Trying with smaller batch size.")
                if rows > 25:
                    # Retry with a smaller batch size
                    return self.search_works_by_keyword(keywords, 25)
                else:
                    logger.error(f"CrossRef error {response.status_code} even with small batch size: {response.text}")
                    return None
                    
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CrossRef API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode CrossRef JSON response: {e}")
             return None

    def process_results(self, results_json):
        """Processes CrossRef JSON results into a standardized format."""
        processed = []
        if not results_json or 'message' not in results_json or 'items' not in results_json['message']:
            return processed

        for item in results_json['message'].get('items', []):
            # --- Filter for abstract existence AFTER retrieval ---
            abstract = item.get('abstract', '').strip()
            title_list = item.get('title', [])
            title = ". ".join(title_list) if title_list else 'Untitled'
            
            # --- NEW: Check for Retraction Keyword --- 
            # Look for explicit retraction markers in title or abstract
            if title.lower().strip().startswith('[retracted]') or \
               abstract.lower().strip().startswith('[retracted]'):
                doi = item.get('DOI', 'Unknown DOI')
                logger.warning(f"Skipping likely retracted CrossRef article based on keyword: DOI {doi}")
                continue # Skip this item
            # --- End Retraction Keyword Check ---
            
            # Clean JATS XML tags from abstract
            abstract = re.sub(r'</?jats:[^>]+>', '', abstract)
            abstract = re.sub(r'</?[^>]+>', '', abstract)
            
            # Skip if abstract is empty or too short after cleaning
            if not abstract or len(abstract) < 50:
                continue

            # Extract authors
            authors_list = []
            if item.get('author'):
                authors_list = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item['author']]
            authors = ", ".join(filter(None, authors_list)) # Join non-empty names

             # Extract publication date (can be complex)
            pub_date_parts = item.get('published-print', {}).get('date-parts', [[]])[0] or \
                             item.get('published-online', {}).get('date-parts', [[]])[0] or \
                             item.get('created', {}).get('date-parts', [[]])[0]
            pub_date = "-".join(map(str, pub_date_parts)) if pub_date_parts else None

            # Extract citation count
            citation_count = item.get('is-referenced-by-count', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": item.get('DOI'),
                "title": title,
                "authors": authors,
                "pub_date": pub_date,
                "abstract": abstract, # Use cleaned abstract
                "source_api": "crossref",
                "citation_count": citation_count
            })
        return processed
# --- End CrossRef Service ---

# --- New: Semantic Scholar Service ---
class SemanticScholarService:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self):
        # Semantic Scholar doesn't strictly require an API key for basic search,

        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{app.config.get("OPENALEX_EMAIL", "rba137@sfu.ca")})'} # Reuse email for politeness
        self.timeout = 30 # Increased timeout

    def search_works_by_keyword(self, keywords, limit=10):
        """Search Semantic Scholar works using keyword query with pagination."""
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        # Use configured limit, but respect the API's 100-per-request limit
        max_per_page = 100
        total_limit = limit # User requested total limit
        retrieved_count = 0
        current_offset = 0
        all_results_data = []
        total_reported_by_api = None # Store the total number of results reported by the API

        # Define the fields we want to retrieve - using correct field names
        fields = 'externalIds,title,authors,year,abstract,citationCount,publicationDate,journal'

        logger.info(f"Querying Semantic Scholar: '{search_query}' with total limit={total_limit} (using pagination)")

        while retrieved_count < total_limit:
            # Determine how many to request in this batch
            request_limit = min(max_per_page, total_limit - retrieved_count)
            if request_limit <= 0:
                 break # Should not happen, but safety check

            params = {
                'query': search_query,
                'limit': request_limit,
                'fields': fields,
                'offset': current_offset
            }
            logger.info(f"  - Requesting batch: limit={request_limit}, offset={current_offset}")

            try:
                response = requests.get(
                    f"{self.BASE_URL}/paper/search",
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout
                )

                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    logger.warning("Semantic Scholar rate limit hit, sleeping for 5 seconds.") # Longer sleep
                    time.sleep(5)
                    continue # Retry the same request after waiting

                # Handle other potential errors
                if response.status_code >= 400:
                    logger.error(f"Semantic Scholar API request failed with status {response.status_code} at offset {current_offset}: {response.text}")
                    # Decide whether to stop or continue trying next pages
                    break # Stop pagination on error

                response.raise_for_status() # Raise HTTP errors for other codes (e.g., 5xx)
                page_data = response.json()

                # Check if 'data' exists and is a list
                if 'data' not in page_data or not isinstance(page_data['data'], list):
                    logger.warning(f"Semantic Scholar response missing 'data' list or is not a list at offset {current_offset}.")
                    break # Stop if data format is unexpected

                # Store the total reported by the API on the first request
                if total_reported_by_api is None:
                    total_reported_by_api = page_data.get('total', 0)
                    logger.info(f"  - Semantic Scholar API reports {total_reported_by_api} total potential results for the query.")

                current_results = page_data['data']
                all_results_data.extend(current_results)
                num_in_page = len(current_results)
                retrieved_count += num_in_page
                current_offset += num_in_page

                logger.info(f"  - Retrieved {num_in_page} results in this batch. Total retrieved: {retrieved_count}")

                # Stop if the API returns fewer results than requested (means we reached the end)
                # Or if we have retrieved all reported results
                if num_in_page < request_limit or (total_reported_by_api is not None and retrieved_count >= total_reported_by_api):
                    logger.info(f"  - Reached end of Semantic Scholar results (requested {request_limit}, got {num_in_page} or reached total {total_reported_by_api}).")
                    break

            except requests.exceptions.Timeout:
                logger.error(f"Semantic Scholar API request timed out at offset {current_offset}. Stopping pagination.")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"Semantic Scholar API request failed at offset {current_offset}: {e}. Stopping pagination.")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode Semantic Scholar JSON response at offset {current_offset}: {e}. Stopping pagination.")
                break
            except Exception as e:
                logger.error(f"Unexpected error during Semantic Scholar pagination at offset {current_offset}: {e}")
                break # Stop on unexpected errors

        logger.info(f"Finished Semantic Scholar pagination. Total results collected: {len(all_results_data)}")
        # Return the aggregated results in the expected format for process_results
        return {'data': all_results_data, 'total': len(all_results_data)} # Mimic single page structure but with all data

    def process_results(self, results_json):
        """Processes Semantic Scholar JSON results into a standardized format."""
        processed = []
        if not results_json or 'data' not in results_json:
            # Also check for 'total' and log if 0 results were found
            if results_json and results_json.get('total', 0) == 0:
                logger.info("Semantic Scholar returned 0 results for the query.")
            else:
                logger.warning(f"Invalid or empty Semantic Scholar results received: {results_json}")
            return processed

        for item in results_json.get('data', []):
            # --- Check for Retraction --- 
            # Semantic Scholar might indicate retractions in `publicationTypes` or a dedicated field
            # Checking for 'Retraction' in publicationTypes as a common indicator
            publication_types = item.get('publicationTypes', [])
            is_retracted = False
            if isinstance(publication_types, list):
                is_retracted = any(pt.lower() == 'retraction' for pt in publication_types)
                
            if is_retracted:
                doi = item.get('externalIds', {}).get('DOI', 'Unknown DOI')
                logger.warning(f"Skipping retracted Semantic Scholar article: DOI {doi}")
                continue # Skip this article
            # --- End Check for Retraction ---
            
            # Skip if abstract is missing or too short
            abstract = item.get('abstract', '')
            if not abstract or len(abstract) < 50:
                continue

            # Extract DOI from externalIds
            external_ids = item.get('externalIds', {})
            doi = external_ids.get('DOI')

            # Extract authors - handle the new author structure
            authors = []
            for author in item.get('authors', []):
                if isinstance(author, dict) and author.get('name'):
                    authors.append(author['name'])
            authors_str = ", ".join(authors)

            # Extract publication date (prefer 'publicationDate', fallback to 'year')
            pub_date = item.get('publicationDate') or str(item.get('year')) if item.get('year') else None

            # Extract citation count
            citation_count = item.get('citationCount', 0)

            # Accept all studies with abstracts, regardless of citation count
            processed.append({
                "doi": doi,
                "title": item.get('title', 'Untitled'),
                "authors": authors_str,
                "pub_date": pub_date,
                "abstract": abstract,
                "source_api": "semantic_scholar",
                "citation_count": citation_count
            })
        return processed
# --- End Semantic Scholar Service ---

# --- New: PubMed Service ---
class PubMedService:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email=None):
        self.email = email or app.config.get('OPENALEX_EMAIL') # Reuse email for politeness
        if not self.email:
            logger.warning("Email not set for PubMed, using default. Set OPENALEX_EMAIL for polite API usage.")
            self.email = 'rba137@sfu.ca' # Default if not set
        self.headers = {'User-Agent': f'Factify/1.0 (mailto:{self.email})'}
        self.timeout = 30 # Increased timeout
        # Rate limiting: NCBI allows 3 requests/second without API key, 10/second with key.
        # We'll add a small delay between requests.
        self.request_delay = 0.5 # Increased delay to 0.5 seconds (was 0.4)

    def _fetch_article_details(self, pmids):
        """Fetches detailed article information for a list of PMIDs with retry logic."""
        if not pmids:
            return []

        pmid_str = ",".join(pmids)
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid_str,
            'retmode': 'xml',
            'rettype': 'abstract',
            'email': self.email
        }
        logger.info(f"Fetching PubMed details for {len(pmids)} PMIDs...")

        max_retries = 2
        retry_delay = 1.5 # Seconds to wait before retrying after a 429

        for attempt in range(max_retries + 1):
            try:
                # Ensure delay *before* each attempt
                if attempt > 0:
                    logger.warning(f"Rate limit hit (429). Retrying PubMed EFetch in {retry_delay}s... (Attempt {attempt}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                     # Still apply base delay before the first attempt of a fetch
                     time.sleep(self.request_delay)

                response = requests.get(fetch_url, params=params, headers=self.headers, timeout=self.timeout)

                if response.status_code == 429:
                    # If it's the last attempt, raise the error, otherwise loop will handle delay/retry
                    if attempt == max_retries:
                        response.raise_for_status() # Raise the 429 error
                    else:
                        continue # Go to the next attempt after delay

                response.raise_for_status() # Raise other HTTP errors immediately
                return response.text # Return XML content on success

            except requests.exceptions.RequestException as e:
                # If it's the last attempt or not a 429 error, log and return None
                if attempt == max_retries or response.status_code != 429:
                     logger.error(f"PubMed EFetch request failed after {attempt+1} attempt(s): {e}")
                     return None
                 # Otherwise, the loop will retry for 429

        return None # Should not be reached if loop logic is correct, but as fallback

    def _parse_pubmed_xml(self, xml_content):
        """Parses the XML from EFetch into a standardized list of dictionaries."""
        processed = []
        if not xml_content:
            return processed

        try:
            root = ET.fromstring(xml_content)
            for article in root.findall('.//PubmedArticle'):
                medline_citation = article.find('.//MedlineCitation')
                if medline_citation is None:
                    continue

                # --- Check for Retraction --- 
                pub_status_elem = medline_citation.find('Article/PublicationStatus')
                if pub_status_elem is not None and pub_status_elem.text == 'Retracted Publication':
                    pmid_elem = medline_citation.find('PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else 'Unknown PMID'
                    logger.warning(f"Skipping retracted PubMed article: PMID {pmid}")
                    continue # Skip this article
                # --- End Check for Retraction ---

                pmid_elem = medline_citation.find('PMID')
                pmid = pmid_elem.text if pmid_elem is not None else None

                article_elem = medline_citation.find('Article')
                if article_elem is None:
                    continue

                title_elem = article_elem.find('ArticleTitle')
                title = title_elem.text if title_elem is not None else 'Untitled'

                abstract_elem = article_elem.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                 # Skip if abstract is missing or too short
                if not abstract or len(abstract) < 50:
                    continue
                    
                # Clean the abstract
                abstract = clean_abstract(abstract)
                
                # Extract authors
                authors_list = []
                author_list_elem = article_elem.find('AuthorList')
                if author_list_elem is not None:
                    for author in author_list_elem.findall('Author'):
                        last_name_elem = author.find('LastName')
                        fore_name_elem = author.find('ForeName')
                        initials_elem = author.find('Initials')
                        name_parts = []
                        if fore_name_elem is not None and fore_name_elem.text:
                             name_parts.append(fore_name_elem.text)
                        elif initials_elem is not None and initials_elem.text:
                             name_parts.append(initials_elem.text + '.') # Add dot for initials
                        if last_name_elem is not None and last_name_elem.text:
                             name_parts.append(last_name_elem.text)
                        if name_parts:
                            authors_list.append(" ".join(name_parts))
                authors = ", ".join(authors_list)

                # Extract publication date (simplified - prefers journal issue pub date)
                pub_date = None
                journal_issue = article_elem.find('.//Journal/JournalIssue')
                if journal_issue is not None:
                    pub_date_elem = journal_issue.find('PubDate')
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        month_elem = pub_date_elem.find('Month')
                        day_elem = pub_date_elem.find('Day')
                        date_parts = []
                        if year_elem is not None and year_elem.text:
                            date_parts.append(year_elem.text)
                            if month_elem is not None and month_elem.text:
                                date_parts.append(month_elem.text.zfill(2)) # Pad month
                                if day_elem is not None and day_elem.text:
                                    date_parts.append(day_elem.text.zfill(2)) # Pad day
                        pub_date = "-".join(date_parts)

                # Extract DOI if available
                doi = None
                doi_elem = article_elem.find(".//ELocationID[@EIdType='doi']") or article_elem.find(".//ArticleId[@IdType='doi']")
                if doi_elem is not None:
                    doi = doi_elem.text
                    
                # Citation count is not directly available via EFetch search results.
                # Set to 0 as a placeholder.
                citation_count = 0

                processed.append({
                    "doi": doi,
                    "title": title,
                    "authors": authors,
                    "pub_date": pub_date,
                    "abstract": abstract,
                    "source_api": "pubmed",
                    "citation_count": citation_count,
                    "pmid": pmid # Include PMID for potential future use
                })
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during PubMed XML processing: {e}")

        return processed

    def search_works_by_keyword(self, keywords, retmax=10):
        """Search PubMed works using keyword query via ESearch, then fetch details via EFetch."""
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        # Use configured max results, but be mindful of API limits (ESearch can be slow)
        retmax = min(retmax, 500) # Keep a reasonable cap on initial search

        esearch_url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': search_query,
            'retmax': retmax,
            'retmode': 'json', # Get PMIDs as JSON
            'sort': 'relevance', # Sort by relevance
            'email': self.email
        }
        logger.info(f"Querying PubMed ESearch: '{search_query}' with retmax={retmax}")
        try:
            time.sleep(self.request_delay) # Adhere to rate limits
            response = requests.get(esearch_url, params=params, headers=self.headers, timeout=self.timeout)
            response.raise_for_status() # Raise HTTP errors
            esearch_results = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed ESearch request failed: {e}")
            return [] # Return empty list on search failure
        except json.JSONDecodeError as e:
             logger.error(f"Failed to decode PubMed ESearch JSON response: {e}")
             return []

        if 'esearchresult' not in esearch_results or 'idlist' not in esearch_results['esearchresult']:
            logger.info("PubMed ESearch returned no results or unexpected format.")
            return []

        pmids = esearch_results['esearchresult']['idlist']
        if not pmids:
            logger.info("PubMed ESearch returned no PMIDs.")
            return []

        logger.info(f"PubMed ESearch found {len(pmids)} potential PMIDs.")

        # Fetch details in batches to avoid overly large EFetch requests
        batch_size = 100 # Fetch details for 100 PMIDs at a time
        all_processed_results = []
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            logger.info(f"Fetching details for PubMed batch {i//batch_size + 1} ({len(batch_pmids)} PMIDs)..." )
            xml_content = self._fetch_article_details(batch_pmids)
            if xml_content:
                batch_results = self._parse_pubmed_xml(xml_content)
                all_processed_results.extend(batch_results)
                logger.info(f"Processed {len(batch_results)} articles from batch.")
            else:
                 logger.warning(f"Failed to fetch details for PubMed batch {i//batch_size + 1}.")
                 
            # Add a small delay between fetch batches as well
            if i + batch_size < len(pmids):
                time.sleep(self.request_delay / 2)
                
        logger.info(f"Total processed PubMed articles after fetching details: {len(all_processed_results)}")
        return all_processed_results

    # process_results is integrated into search_works_by_keyword via _parse_pubmed_xml
# --- End PubMed Service ---

# --- Modified Gemini Service ---
class GeminiService:
    def __init__(self, model=None):
        self.model = model

    def preprocess_claim(self, claim):
        """Extracts keywords and category from the claim using Gemini."""
        if not self.model:
            logger.error("Gemini model not available for preprocessing.")
            # Fallback: Use simple keyword extraction (e.g., based on nouns/verbs) or return empty
            return {"keywords": [], "category": "unknown"}

        prompt = f"""
        Analyze the following claim to help find relevant academic research.
        Claim: "{claim}"

        1.  **Extract Key Terms:** Identify the 5-7 most important nouns, noun phrases, or technical terms central to the claim's core assertion. These terms should be suitable for searching academic databases like OpenAlex and CrossRef.
        2.  **Categorize Claim:** Classify the claim into ONE primary category from the following list:
            *   Health & Medicine
            *   Biology & Life Sciences
            *   Physical Sciences (Physics, Chemistry, Astronomy)
            *   Earth & Environmental Sciences
            *   Technology & Engineering
            *   Social Sciences (Psychology, Sociology, Economics, Politics)
            *   Humanities (History, Arts, Literature)
            *   Mathematics & Computer Science
            *   General / Other

        Return the results ONLY as a JSON object with keys "keywords" (a list of strings) and "category" (a single string). Do not include any explanations or surrounding text.

        Example:
        Claim: "Regular exercise reduces the risk of cardiovascular disease."
        {{
            "keywords": ["regular exercise", "cardiovascular disease", "risk reduction", "heart health", "physical activity"],
            "category": "Health & Medicine"
        }}

        Now, analyze the claim provided above.
        """

        try:
            logger.info(f"Sending preprocessing request to Gemini for claim: '{claim}'")
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                # Validate expected keys
                if "keywords" in result and "category" in result and isinstance(result["keywords"], list):
                     logger.info(f"Successfully preprocessed claim. Keywords: {result['keywords']}, Category: {result['category']}")
                     return result
                else:
                    raise ValueError("Missing or invalid keys in JSON response.")
            except (ValueError, IndexError) as e:
                 logger.error(f"Failed to parse JSON from Gemini preprocessing response: {e}. Response: {response_text}")
                 # Fallback or re-attempt logic could go here
                 return {"keywords": [], "category": "unknown", "error": "LLM parsing failed"}


        except Exception as e:
            logger.error(f"Error during Gemini claim preprocessing: {e}")
            return {"keywords": [], "category": "unknown", "error": str(e)}


    def analyze_with_rag(self, claim, evidence_chunks):
        """Analyzes the claim against retrieved evidence chunks using Gemini RAG."""
        if not self.model:
            logger.error("Gemini model not available for RAG analysis.")
            return {
                "verdict": "Inconclusive",
                "reasoning": "Analysis could not be performed (LLM unavailable).",
                "confidence": 0.0
            }

        if not evidence_chunks:
            logger.warning("No evidence chunks provided for RAG analysis.")
            return {
                "verdict": "Inconclusive",
                "reasoning": "No relevant evidence found to analyze the claim.",
                "confidence": 0.0
            }

        # Format evidence for the prompt
        formatted_evidence = "\n\n".join([f"Evidence Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(evidence_chunks)])

        prompt = f"""
        You are a meticulous fact-checking analyst. Your task is to evaluate the following claim based *only* on the provided evidence chunks extracted from academic abstracts.

        Claim: "{claim}"

        Evidence Chunks:
        ---
        {formatted_evidence}
        ---

        Instructions:
        1.  Carefully read the claim and each evidence chunk.
        2.  **Distinguish Core Subject vs. External Factors:** When evaluating, clearly differentiate between evidence directly addressing the properties or effects of the claim's *core subject* versus evidence related to *external factors* (e.g. specific populations studied, interactions, study limitations mentioned in the evidence).
        3.  Determine how accurate the claim is based *primarily* on the scientific evidence related to the core subject, while acknowledging the influence of external factors.
        4.  Create TWO different summaries:
            a. First, provide a DETAILED SCIENTIFIC summary (3-5 sentences) that references specific evidence chunks using the `[EVIDENCE_CHUNK:NUMBERS]` format (e.g., `[EVIDENCE_CHUNK:5,12,18]`). **Crucially, if applicable, explicitly mention the distinction identified in step 2.** For instance, state if the evidence supports/refutes the core subject itself, but external factors introduce caveats (or vice-versa).
            b. Second, provide a SIMPLIFIED summary (2-3 sentences) in plain language. **This summary should also reflect the core subject vs. external factor distinction clearly** but without technical jargon or specific chunk references. Explain the main conclusion about the core subject and mention any important caveats from external factors.
        5.  Assign an ACCURACY SCORE between 0.0 (completely inaccurate based on core subject evidence) and 1.0 (completely accurate based on core subject evidence). This score should reflect how well the claim about the *core subject* is supported, considering the nuances. Explain *how* external factors modify the interpretation in the detailed reasoning.
        6.  If you still want to provide a categorical verdict, include it as "Supported", "Partially Supported", "Refuted", or "Inconclusive".

        Return ONLY a JSON object with the keys "verdict", "detailed_reasoning", "simplified_reasoning", and "accuracy_score". Do not include any other text, markdown formatting, or explanations outside the JSON structure.

        Example Output:
        {{
            "verdict": "Partially Supported",
            "detailed_reasoning": "Evidence suggests whey protein itself can aid muscle synthesis [EVIDENCE_CHUNK:3,7]. However, significant concerns are raised regarding supplement contamination [EVIDENCE_CHUNK:2,18,50] and potential adverse interactions [EVIDENCE_CHUNK:62], which are external factors not inherent to pure whey protein. Benefits may also be population-specific [EVIDENCE_CHUNK:11,17]. While whey shows promise, the risks associated with commercial supplements complicate a simple 'good'/'bad' assessment.",
            "simplified_reasoning": "Research indicates whey protein itself may have benefits for muscle growth. However, be cautious as whey supplements can sometimes be contaminated or interact negatively with other substances. Professional guidance is recommended before use.",
            "accuracy_score": 0.65
        }}

        Now, analyze the claim and evidence provided above.
        """

        try:
            logger.info(f"Sending RAG analysis request to Gemini for claim: '{claim}'")
            response = self.model.generate_content(prompt)
            response_text = response.text

             # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                 # Validate expected keys
                if "verdict" in result and ("detailed_reasoning" in result or "reasoning" in result) and ("accuracy_score" in result or "confidence" in result):
                    # Log accuracy score if available
                    accuracy_message = ""
                    if "accuracy_score" in result:
                        accuracy_message = f"Accuracy Score: {result['accuracy_score']}"
                    elif "confidence" in result:
                        # For backward compatibility
                        result["accuracy_score"] = result["confidence"]
                        accuracy_message = f"Confidence: {result['confidence']}"
                    
                    logger.info(f"Successfully analyzed claim. {accuracy_message}")
                    
                    # Ensure backward compatibility
                    if "reasoning" in result and "detailed_reasoning" not in result:
                        result["detailed_reasoning"] = result["reasoning"]
                    if "simplified_reasoning" not in result:
                        result["simplified_reasoning"] = result.get("reasoning", "Analysis complete but summary unavailable.")
                    return result
                else:
                    raise ValueError("Missing or invalid keys in JSON response.")
            except (ValueError, IndexError) as e:
                 logger.error(f"Failed to parse JSON from Gemini RAG response: {e}. Response: {response_text}")
                 # Fallback or re-attempt logic could go here
                 return {
                     "verdict": "Error",
                     "reasoning": "LLM analysis failed to produce valid JSON output.",
                     "detailed_reasoning": "LLM analysis failed to produce valid JSON output.",
                     "simplified_reasoning": "Analysis failed. Please try again.",
                     "accuracy_score": 0.0
                 }

        except Exception as e:
            logger.error(f"Error during Gemini RAG analysis: {e}")
            return {
                "verdict": "Error",
                "reasoning": f"An unexpected error occurred during analysis: {str(e)}",
                "detailed_reasoning": f"An unexpected error occurred during analysis: {str(e)}",
                "simplified_reasoning": "An unexpected error occurred during analysis.",
                "accuracy_score": 0.0
            }

# --- End Gemini Service ---


# --- New RAG Verification Service ---
class RAGVerificationService:
    def __init__(self, gemini_service, openalex_service, crossref_service, semantic_scholar_service, pubmed_service, db_session):
        self.gemini = gemini_service
        self.openalex = openalex_service
        self.crossref = crossref_service
        self.semantic_scholar = semantic_scholar_service
        self.pubmed = pubmed_service
        self.db = db_session
        self.vector_store = VectorStoreService(db_session) # Initialize vector store service

    def _embed_studies_batch(self, studies_to_embed):
        """Helper to embed a batch of Study objects using Gemini API."""
        if not studies_to_embed:
            return 0

        embedded_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process studies in smaller batches to avoid overwhelming the API
        batch_size = 20
        total_batches = (len(studies_to_embed) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(studies_to_embed)} studies for embedding in {total_batches} batches of {batch_size}...")
        
        for i in range(0, len(studies_to_embed), batch_size):
            batch = studies_to_embed[i:i+batch_size]
            logger.debug(f"Processing batch {(i//batch_size)+1}/{total_batches} ({len(batch)} studies)...")
            
            for study in batch:
                try:
                    if not study.abstract:
                        skipped_count += 1
                        continue
                        
                    embedding = self.vector_store.get_embedding_for_text(
                        study.abstract,
                        task_type="retrieval_document" # Use correct task type
                    )
                    
                    if embedding:
                        study.embedding = embedding # Assign the list directly
                        embedded_count += 1
                    else:
                        logger.warning(f"Failed to get embedding for study ID {study.id}, skipping.")
                        skipped_count += 1
                except Exception as embed_err:
                    error_count += 1
                    logger.error(f"Error embedding study ID {study.id}: {embed_err}. Skipping.")
                    # Continue with the next study rather than failing the whole batch
            
            # Add small delay between batches to avoid rate limiting
            if i + batch_size < len(studies_to_embed):
                time.sleep(0.5)  # 500ms delay between batches
                
        logger.info(f"Embedding complete: {embedded_count} successful, {skipped_count} skipped, {error_count} errors.")
        return embedded_count


    def process_claim_request(self, claim):
        """Orchestrates the RAG workflow with immediate API embedding."""
        start_time = time.time()
        logger.info(f"Starting RAG verification for claim: '{claim}'")

        # 1. Preprocess Claim
        preprocessing_result = self.gemini.preprocess_claim(claim)
        keywords = preprocessing_result.get("keywords", [])
        keyword_string = " ".join(keywords)
        category = preprocessing_result.get("category", "unknown")

        if not keywords:
            logger.warning("No keywords extracted, cannot retrieve evidence.")
            return {"error": "Could not extract keywords from claim.", "status": "failed"}

        logger.info(f"Extracted Keywords: {keywords}, Category: {category}")

        # 2. Retrieve Evidence - Concurrently
        openalex_limit = app.config['OPENALEX_MAX_RESULTS']
        crossref_limit = app.config['CROSSREF_MAX_RESULTS']
        semantic_scholar_limit = app.config['SEMANTIC_SCHOLAR_MAX_RESULTS']
        pubmed_limit = app.config['PUBMED_MAX_RESULTS'] # Get PubMed limit

        all_studies_data = []
        openalex_studies = []
        crossref_studies = []
        semantic_scholar_studies = []
        pubmed_studies = [] # Initialize list for PubMed studies

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: # Increased workers to 4
            future_openalex = executor.submit(self.openalex.search_works_by_keyword, keyword_string, per_page=openalex_limit)
            future_crossref = executor.submit(self.crossref.search_works_by_keyword, keyword_string, rows=crossref_limit)
            future_semantic = executor.submit(self.semantic_scholar.search_works_by_keyword, keyword_string, limit=semantic_scholar_limit)
            future_pubmed = executor.submit(self.pubmed.search_works_by_keyword, keyword_string, retmax=pubmed_limit) # Submit PubMed task

            try:
                openalex_data = future_openalex.result()
                if openalex_data:
                    openalex_studies = self.openalex.process_results(openalex_data)
                logger.info(f"Retrieved {len(openalex_studies)} studies from OpenAlex (limit: {openalex_limit}).")
            except Exception as e:
                logger.error(f"Error retrieving data from OpenAlex: {e}")

            try:
                crossref_data = future_crossref.result()
                if crossref_data:
                    crossref_studies = self.crossref.process_results(crossref_data)
                logger.info(f"Retrieved {len(crossref_studies)} studies from CrossRef (limit: {crossref_limit}).")
            except Exception as e:
                logger.error(f"Error retrieving data from CrossRef: {e}")

            try:
                semantic_scholar_data = future_semantic.result()
                if semantic_scholar_data:
                    semantic_scholar_studies = self.semantic_scholar.process_results(semantic_scholar_data)
                logger.info(f"Retrieved {len(semantic_scholar_studies)} studies from Semantic Scholar (limit: {semantic_scholar_limit}).")
            except Exception as e:
                logger.error(f"Error retrieving data from Semantic Scholar: {e}")

            # Process PubMed results
            try:
                pubmed_data = future_pubmed.result()
                if pubmed_data: # PubMed service returns the processed list directly
                    pubmed_studies = pubmed_data
                logger.info(f"Retrieved {len(pubmed_studies)} studies from PubMed (limit: {pubmed_limit}).")
            except Exception as e:
                logger.error(f"Error retrieving data from PubMed: {e}")

        all_studies_data = openalex_studies + crossref_studies + semantic_scholar_studies + pubmed_studies # Add PubMed results
        logger.info(f"Total studies retrieved before deduplication: {len(all_studies_data)}")

        # 3. Filter & Deduplicate
        seen_dois = set()
        seen_titles = {}
        unique_studies_data = []

        # Process studies with DOIs first
        for study_data in all_studies_data:
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
            doi = study_data.get('doi')
            if doi:
                doi_norm = doi.lower()
                if doi_norm not in seen_dois:
                    unique_studies_data.append(study_data)
                    seen_dois.add(doi_norm)
                    title = study_data.get('title')
                    if title:
                        title_lower = title.lower().strip()
                        seen_titles[title_lower] = doi_norm

        # Process studies without DOIs or with different casing
        for study_data in all_studies_data:
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
            doi = study_data.get('doi')
            doi_norm = doi.lower() if doi else None

            if doi_norm and doi_norm in seen_dois:
                continue

            if not doi_norm:
                title = study_data.get('title')
                if not title:
                    continue
                title_lower = title.lower().strip()
                if title_lower in seen_titles:
                    continue

                unique_studies_data.append(study_data)
                seen_titles[title_lower] = None

        # Limit total studies to process
        studies_to_process_data = unique_studies_data[:app.config['MAX_EVIDENCE_TO_STORE']]
        logger.info(f"Processing up to {len(studies_to_process_data)} unique studies with abstracts for this request.")

        if not studies_to_process_data:
            logger.warning("No usable evidence found after filtering.")
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "detailed_reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "simplified_reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "accuracy_score": 0.0,
                "evidence": [],
                "keywords_used": preprocessing_result.get("keywords", []),
                "category": preprocessing_result.get("category", "unknown"),
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

        # 4. Store Evidence & Embed NEW Studies Immediately
        stored_studies = []
        studies_requiring_embedding = []
        success_count = 0
        error_count = 0
        
        try:
            # Find existing studies
            existing_dois_in_db_map = {}
            study_dois_to_check = [s.get('doi').lower() for s in studies_to_process_data if s.get('doi')]
            
            if study_dois_to_check:
                existing_studies_in_db = self.db.query(Study).filter(Study.doi.in_(study_dois_to_check)).all()
                existing_dois_in_db_map = {study.doi.lower(): study for study in existing_studies_in_db}
                logger.info(f"Found {len(existing_dois_in_db_map)} studies already in database.")

            # Process in smaller batches for better transaction handling
            batch_size = 20
            for i in range(0, len(studies_to_process_data), batch_size):
                batch = studies_to_process_data[i:i+batch_size]
                batch_objects = []
                batch_to_embed = []
                
                logger.debug(f"Processing batch {i//batch_size + 1} ({len(batch)} studies)...")
                
                for study_data in batch:
                    try:
                        doi = study_data.get('doi')
                        doi_norm = doi.lower() if doi else None
                        existing_study = None
                        
                        if doi_norm and doi_norm in existing_dois_in_db_map:
                            existing_study = existing_dois_in_db_map[doi_norm]
                            
                        if existing_study:
                            stored_studies.append(existing_study)
                            # Optional update for existing study if needed
                        else:
                            # Create new study
                            study_obj = Study(
                                doi=doi,
                                title=study_data.get('title'),
                                authors=study_data.get('authors'),
                                pub_date=study_data.get('pub_date'),
                                abstract=study_data.get('abstract'),
                                source_api=study_data.get('source_api'),
                                citation_count=study_data.get('citation_count', 0),
                                embedding=None
                            )
                            
                            stored_studies.append(study_obj)
                            batch_objects.append(study_obj)
                            
                            if study_obj.abstract:
                                batch_to_embed.append(study_obj)
                                
                            if doi_norm:
                                existing_dois_in_db_map[doi_norm] = study_obj
                                
                        success_count += 1
                    except Exception as e:
                        logger.error(f"Error processing study {study_data.get('doi', 'unknown')}: {e}")
                        error_count += 1
                        continue
                
                # Add batch to session
                if batch_objects:
                    try:
                        self.db.add_all(batch_objects)
                        self.db.flush()
                        logger.debug(f"Added {len(batch_objects)} new studies to session.")
                    except SQLAlchemyError as e: # Catch SQLAlchemy errors specifically
                        self.db.rollback() # Rollback is crucial
                        # Check if the underlying error is a UniqueViolation
                        if isinstance(e.orig, errors.UniqueViolation):
                            # Log a concise warning for duplicates, which are expected
                            logger.warning(f"Batch insert/flush failed due to duplicate DOI(s). Rolled back batch.")
                        else:
                            # Log the first line for other database errors
                            error_summary = str(e).split('\n')[0]
                            logger.error(f"Database error during batch add/flush: {error_summary}")
                        continue # Continue to the next batch
                    except Exception as e: # Catch other unexpected errors
                        self.db.rollback()
                        logger.error(f"Unexpected error adding batch to session: {e}")
                        continue
                
                # Embed studies in this batch
                if batch_to_embed:
                    try:
                        num_embedded = self._embed_studies_batch(batch_to_embed)
                        logger.debug(f"Embedded {num_embedded} studies in this batch.")
                    except Exception as e:
                        logger.error(f"Error during batch embedding: {e}")
                        # Continue with unembedded studies
                
                # Commit this batch
                try:
                    self.db.commit()
                    logger.debug(f"Committed batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error committing batch: {e}")
                    self.db.rollback()
                    
                    # Try to commit individual studies if batch fails
                    for obj in batch_objects:
                        try:
                            self.db.add(obj)
                            self.db.commit()
                        except Exception as inner_e:
                            # Extract just the first line of the error message for conciseness
                            error_summary = str(inner_e).split('\n')[0]
                            logger.error(f"Error committing individual study {obj.doi or 'ID:'+str(obj.id)}: {error_summary}")
                            self.db.rollback()
            
            logger.info(f"Finished processing all studies. Success: {success_count}, Errors: {error_count}")
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during evidence storage/embedding phase: {e}")
            return {"error": "Database error storing/embedding evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during evidence storage/embedding phase: {e}")
            return {"error": f"Unexpected error: {str(e)}", "status": "failed"}

        # 5. Rank Studies
        logger.info(f"Ranking studies for RAG analysis...")
        
        # Get claim embedding
        try:
            claim_embedding = self.vector_store.get_embedding_for_text(claim, task_type="retrieval_query")
            if not claim_embedding:
                raise ValueError("Claim embedding failed.")
            claim_embedding_np = np.array(claim_embedding).astype('float32')
        except Exception as e:
            logger.error(f"Error generating claim embedding: {e}")
            return {"error": "Failed to generate claim embedding.", "status": "failed"}

        # Rank studies
        ranked_studies_with_scores = []
        for study in stored_studies:
            relevance_score = 0.0
            
            if study.embedding is not None:
                try:
                    study_embedding_np = np.array(study.embedding).astype('float32')
                    
                    # Check if dimensions match - resize if needed
                    if len(study_embedding_np) != len(claim_embedding_np):
                        logger.warning(f"Dimension mismatch: study {len(study_embedding_np)}, claim {len(claim_embedding_np)}")
                        # Skip this study for ranking
                        continue
                    
                    dot_product = np.dot(claim_embedding_np, study_embedding_np)
                    norm_claim = np.linalg.norm(claim_embedding_np)
                    norm_study = np.linalg.norm(study_embedding_np)
                    
                    if norm_claim > 0 and norm_study > 0:
                        similarity = dot_product / (norm_claim * norm_study)
                        relevance_score = max(0.0, min(1.0, similarity))
                except Exception as e:
                    logger.warning(f"Error calculating relevance for study {study.doi or study.id}: {e}")
                    relevance_score = 0.0
            else:
                relevance_score = 0.0

            # Use log scale for citation counts to prevent very high counts from dominating
            credibility_score = math.log10(study.citation_count + 1)
            
            # Weights for relevance vs. credibility
            relevance_weight = 0.7 if study.embedding is not None else 0.1
            credibility_weight = 1.0 - relevance_weight
            
            combined_score = (relevance_weight * relevance_score) + (credibility_weight * credibility_score)
            ranked_studies_with_scores.append((study, combined_score))

        # Sort and select top K
        ranked_studies_with_scores.sort(key=lambda item: item[1], reverse=True)
        top_k = app.config['RAG_TOP_K']
        top_ranked_studies = [study for study, score in ranked_studies_with_scores[:top_k]]
        relevant_chunks = [study.abstract for study in top_ranked_studies if study.abstract]

        logger.info(f"Selected top {len(top_ranked_studies)} ranked studies for LLM analysis.")
        
        if not relevant_chunks:
             logger.warning("No abstracts available from top ranked studies.")
             return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "detailed_reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "simplified_reasoning": "No relevant academic studies with abstracts could be retrieved for analysis.",
                "accuracy_score": 0.0,
                "evidence": [],
                "keywords_used": preprocessing_result.get("keywords", []),
                "category": preprocessing_result.get("category", "unknown"),
                "processing_time_seconds": round(time.time() - start_time, 2)
             }

        # 6. Analyze with LLM
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 7. Format and Return Output
        evidence_details = []
        for idx, study in enumerate(top_ranked_studies):
            if study.abstract:
                evidence_details.append({
                    "id": idx + 1,
                    "title": study.title,
                    "abstract": study.abstract,
                    "authors": study.authors,
                    "doi": study.doi,
                    "pub_date": study.pub_date,
                    "source_api": study.source_api,
                    "citation_count": study.citation_count or 0
                })

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."),
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "accuracy_score": analysis_result.get("accuracy_score", 0.0),
            "evidence": evidence_details,
            "keywords_used": preprocessing_result.get("keywords", []),
            "category": preprocessing_result.get("category", "unknown"),
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

        logger.info(f"RAG verification completed for claim: '{claim}'. Accuracy Score: {final_response['accuracy_score']}")
        return final_response

# --- End Modified RAG Verification Service ---

# Initialize services
openalex_service = OpenAlexService()
crossref_service = CrossRefService()
semantic_scholar_service = SemanticScholarService()
gemini_service = GeminiService(gemini_model)
pubmed_service = PubMedService() # Initialize PubMed service

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = "disconnected"
    try:
        connection = engine.connect()
        connection.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    # --- REMOVE Redis Check ---
    # redis_status = "disconnected" ...
    # --- End REMOVE Redis Check ---

    gemini_status = "ok" if gemini_model else "unavailable"
    # Change embedding status check - API based now
    embedding_status = "ok" if gemini_api_key else "api_key_missing"
    vector_db_status = db_status # Still tied to main DB

    return jsonify({
        "status": "ok",
        "service": "Factify RAG API",
        "version": "2.2.0", # Update version
        "dependencies": {
            "database": db_status,
            "gemini_model": gemini_status,
            "embedding_api": embedding_status, # Changed from embedding_model
            "vector_storage": vector_db_status,
            # REMOVE "redis_queue": redis_status,
            "openalex_api": "ok",
            "crossref_api": "ok",
            "semantic_scholar_api": "ok", # Assuming ok
            "pubmed_api": "ok" # Assuming ok
        }
    })

# --- Updated Claim Verification Endpoint (Logic inside RAG service is changed) ---
@app.route('/api/verify_claim', methods=['POST'])
def verify_claim_rag():
    """Verifies a claim using RAG with immediate API embedding."""
    try:
        # Parse request JSON data
        data = request.get_json()
        
        if not data or 'claim' not in data:
            return jsonify({
                "error": "Missing 'claim' in request body",
                "status": "failed"
            }), 400
            
        claim = data['claim']
        
        if not claim or not isinstance(claim, str) or len(claim.strip()) < 5:
            return jsonify({
                "error": "Claim must be a string with at least 5 characters",
                "status": "failed"
            }), 400
            
        # Get database session
        db = next(get_db())
        
        try:
            # Initialize RAG service
            rag_service = RAGVerificationService(
                gemini_service,
                openalex_service,
                crossref_service,
                semantic_scholar_service,
                pubmed_service, # Pass PubMed service instance
                db
            )
            
            # Process the claim
            result = rag_service.process_claim_request(claim)
            
            # Check if the result has an error key
            if result and "error" in result:
                # Still return error directly, not nested
                return jsonify(result), 500
                
            # Return the result, WRAPPED in a 'result' key for frontend compatibility
            return jsonify({"result": result})
            
        except Exception as e:
            logger.error(f"Error in RAG verification: {e}")
            return jsonify({
                "error": f"Internal server error during claim verification: {str(e)}",
                "status": "failed"
            }), 500
        finally:
            # --- Add expunge_all() here ---
            try:
                logger.debug("Expunging objects from DB session before closing.")
                db.expunge_all() # Detach all objects from the session
                logger.debug("Expunged objects successfully.")
            except Exception as expunge_e:
                logger.error(f"Error expunging session objects: {expunge_e}")
            # --- End Add ---
            logger.debug("Closing DB session.")
            db.close()
            logger.debug("DB session closed.")
            
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "error": f"Failed to process request: {str(e)}",
            "status": "failed"
        }), 400

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))