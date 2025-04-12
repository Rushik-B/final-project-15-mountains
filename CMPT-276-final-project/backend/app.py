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
from sentence_transformers import SentenceTransformer
# --- End New Imports ---
# --- New pgvector Import ---
from pgvector.sqlalchemy import Vector
# --- End pgvector Import ---
# --- Import for UniqueViolation ---
from psycopg2 import errors
# --- End Import ---

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'factify-dev-key'),
    DEBUG=os.getenv('FLASK_DEBUG', 'False') == 'True',
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY'),
    OPENALEX_EMAIL=os.getenv('OPENALEX_EMAIL', 'rba137@sfu.ca'),
    # --- Database Config ---
    DATABASE_URL=os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/postgres'),
    EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    # Replace single limit with specific API limits
    OPENALEX_MAX_RESULTS=int(os.getenv('OPENALEX_MAX_RESULTS', '200')),
    CROSSREF_MAX_RESULTS=int(os.getenv('CROSSREF_MAX_RESULTS', '1000')),
    SEMANTIC_SCHOLAR_MAX_RESULTS=int(os.getenv('SEMANTIC_SCHOLAR_MAX_RESULTS', '100')),
    # Keep these settings
    MAX_EVIDENCE_TO_STORE=int(os.getenv('MAX_EVIDENCE_TO_STORE', '800')), # Increased default from 400 to 800 total
    RAG_TOP_K=int(os.getenv('RAG_TOP_K', '20')), # Number of chunks for RAG analysis
)

# --- Initialize Embedding Model ---
# Load the embedding model once at application start
# This needs to happen before the Study model definition which uses embedding_dimension.
embedding_model_name = app.config['EMBEDDING_MODEL']
embedding_model = None
embedding_dimension = 384 # Default dimension for MiniLM-L6-v2, fallback if model load fails
try:
    logger.info(f"Loading sentence transformer model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    # Determine dimension dynamically
    dummy_embedding = embedding_model.encode(["test"])
    embedding_dimension = dummy_embedding.shape[1]
    logger.info(f"Successfully loaded embedding model with dimension {embedding_dimension}")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize Sentence Transformer model: {e}")
    logger.warning(f"Proceeding with default embedding dimension {embedding_dimension}. Vector search may be inaccurate or fail if this doesn't match the model.")
    # Keep embedding_model = None, checks later in the code will handle this.

# --- End Initialize Embedding Model ---

# Initialize Gemini API
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
    # The dimension must match the embedding model output
    embedding = Column(Vector(embedding_dimension), nullable=True)
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
                    logger.info("'embedding' column not found via information_schema. Attempting to add it...")
                    try:
                        # Ensure vector extension is available
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                        # Add the column with the correct dimension
                        conn.execute(text(f"ALTER TABLE studies ADD COLUMN embedding vector({embedding_dimension})"))
                        conn.commit() # Commit after ALTER TABLE
                        logger.info(f"Successfully added 'embedding' column with dimension {embedding_dimension}.")
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
        # Embedding model is now handled globally or passed in

    def add_embeddings_to_studies(self, studies_to_update):
        """
        Generates embeddings for studies that don't have them and updates the DB.
        Assumes embedding_model is available in the scope or passed.
        """
        if not embedding_model:
            logger.error("Embedding model not available. Cannot generate embeddings.")
            return
        if not studies_to_update:
            logger.info("No studies provided to embed.")
            return

        studies_needing_embedding = [s for s in studies_to_update if s.abstract and s.embedding is None]

        if not studies_needing_embedding:
            logger.info("No studies need embedding in this batch.")
            return

        logger.info(f"Generating embeddings for {len(studies_needing_embedding)} studies...")

        batch_size = 100 # Can be larger as we are just encoding, not storing large index
        embedded_count = 0
        try:
            for i in range(0, len(studies_needing_embedding), batch_size):
                batch = studies_needing_embedding[i:i+batch_size]
                abstracts = [s.abstract for s in batch]

                try:
                    embeddings = embedding_model.encode(abstracts, show_progress_bar=False)
                    embeddings_np = np.array(embeddings).astype('float32')

                    # Update the embedding field for each study in the batch
                    for study, embedding_vec in zip(batch, embeddings_np):
                        study.embedding = embedding_vec # Assign the numpy array directly
                        self.db.add(study) # Add to session to mark for update

                    embedded_count += len(batch)
                    logger.info(f"Embedded batch {i//batch_size + 1}. Total embedded: {embedded_count}")

                except Exception as e:
                    logger.error(f"Error embedding batch: {e}")
                    # Decide how to handle batch errors: skip batch, stop, etc.
                    continue # Continue with the next batch

            # Commit all the updates at the end
            self.db.commit()
            logger.info(f"Successfully generated and stored embeddings for {embedded_count} studies.")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during embedding update commit: {e}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during embedding generation/storage: {e}")

    def find_relevant_studies(self, claim_text, top_k):
        """
        Finds top_k relevant studies from the database using vector similarity search.
        Returns the full Study objects.
        """
        if not embedding_model:
            logger.error("Embedding model not available. Cannot perform vector search.")
            return []

        try:
            # 1. Embed the claim text
            claim_embedding = embedding_model.encode([claim_text])
            claim_embedding_np = np.array(claim_embedding).astype('float32')

            # 2. Perform vector similarity search in PostgreSQL
            # Options for distance:
            # - Study.embedding.cosine_distance(claim_embedding_np[0]) -> Lower is better (0=identical, 2=opposite)
            # - Study.embedding.l2_distance(claim_embedding_np[0]) -> Lower is better (Euclidean)
            # - Study.embedding.max_inner_product(claim_embedding_np[0]) -> Higher is better
            # We'll use cosine distance and order ascending (most similar first)
            logger.info(f"Searching database for top {top_k} studies using vector similarity (cosine distance)...")
            # Use the first element of claim_embedding_np since encode returns a list of embeddings
            relevant_studies = (
                self.db.query(Study)
                .filter(Study.embedding != None) # Ensure embedding exists
                .order_by(Study.embedding.cosine_distance(claim_embedding_np[0]))
                .limit(top_k)
                .all()
            )

            if not relevant_studies:
                logger.warning("Vector search returned no relevant studies.")
                return []

            logger.info(f"Retrieved {len(relevant_studies)} relevant studies from database vector search.")
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
                "title": ". ".join(item.get('title', ['Untitled'])),
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
        # but recommends one for higher rate limits. We'll proceed without one for now.
        # If rate limits become an issue, an API key can be added.
        # https://www.semanticscholar.org/product/api#Authentication
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
        2.  Determine how accurate the claim is based on the available scientific evidence.
        3.  Create TWO different summaries:
            a. First, provide a DETAILED SCIENTIFIC summary (3-5 sentences) that references specific evidence chunks. **Crucially, when referencing evidence chunk numbers (e.g., chunk 5, or chunks 5, 12, and 18), you MUST wrap the reference in the format `[EVIDENCE_CHUNK:NUMBERS]` where NUMBERS is a comma-separated list of the chunk numbers. Example: `... findings from [EVIDENCE_CHUNK:5,12,18] indicate ...` or `... as shown in [EVIDENCE_CHUNK:3] ...`. Do NOT use any other format for referencing chunks.**
            b. Second, provide a SIMPLIFIED summary (2-3 sentences) in plain language that explains your assessment to a general audience without technical jargon or references to specific evidence chunks.
        4.  Assign an ACCURACY SCORE between 0.0 (completely inaccurate) and 1.0 (completely accurate) to the claim. This score should reflect how well the claim is supported by the scientific evidence provided.
        5.  If you still want to provide a categorical verdict, include it as "Supported", "Partially Supported", "Refuted", or "Inconclusive".

        Return ONLY a JSON object with the keys "verdict", "detailed_reasoning", "simplified_reasoning", and "accuracy_score". Do not include any other text, markdown formatting, or explanations outside the JSON structure.

        Example Output:
        {{
            "verdict": "Partially Supported",
            "detailed_reasoning": "Evidence [EVIDENCE_CHUNK:3,7,12] suggests potential health benefits under specific conditions, while [EVIDENCE_CHUNK:5,9,15] indicate possible limitations. Methodological considerations noted in [EVIDENCE_CHUNK:2,8,14] and the need for more controlled trials are emphasized. The research provides moderate support for the claim, but with notable limitations ([EVIDENCE_CHUNK:4,11]).",
            "simplified_reasoning": "The research provides some support for this claim, but with important limitations. Some studies show potential benefits, while others highlight concerns. Scientists agree that the claim is partially accurate but more research is needed.",
            "accuracy_score": 0.6
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
    # Restore db_session and vector_store_service
    def __init__(self, gemini_service, openalex_service, crossref_service, semantic_scholar_service, db_session, vector_store_service):
        self.gemini = gemini_service
        self.openalex = openalex_service
        self.crossref = crossref_service
        self.semantic_scholar = semantic_scholar_service
        self.db = db_session # RESTORED
        self.vector_store = vector_store_service # RESTORED

    def process_claim_request(self, claim):
        """Orchestrates the entire RAG workflow for a claim."""
        start_time = time.time()
        logger.info(f"Starting RAG verification for claim: '{claim}'")

        # 1. Preprocess Claim (Keywords + Category)
        preprocessing_result = self.gemini.preprocess_claim(claim)
        keywords = preprocessing_result.get("keywords", [])
        keyword_string = " ".join(keywords)
        category = preprocessing_result.get("category", "unknown")

        if not keywords:
            logger.warning("No keywords extracted, cannot retrieve evidence.")
            return {"error": "Could not extract keywords from claim.", "status": "failed"}

        logger.info(f"Extracted Keywords: {keywords}, Category: {category}")

        # 2. Retrieve Evidence - Concurrently
        # Use specific limits for each API source
        openalex_limit = app.config['OPENALEX_MAX_RESULTS']
        crossref_limit = app.config['CROSSREF_MAX_RESULTS']
        semantic_scholar_limit = app.config['SEMANTIC_SCHOLAR_MAX_RESULTS']
        
        all_studies_data = []
        openalex_studies = []
        crossref_studies = []
        semantic_scholar_studies = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_openalex = executor.submit(self.openalex.search_works_by_keyword, keyword_string, per_page=openalex_limit)
            future_crossref = executor.submit(self.crossref.search_works_by_keyword, keyword_string, rows=crossref_limit)
            future_semantic = executor.submit(self.semantic_scholar.search_works_by_keyword, keyword_string, limit=semantic_scholar_limit)

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

        all_studies_data = openalex_studies + crossref_studies + semantic_scholar_studies
        logger.info(f"Total studies retrieved before deduplication: {len(all_studies_data)}")

        # --- Enhanced Filtering & Deduplication --- Restore original logic
        seen_dois = set()
        seen_titles = {} # For deduplicating studies without DOIs
        unique_studies_data = []

        # First pass: process studies with DOIs
        for study_data in all_studies_data:
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
            doi = study_data.get('doi')
            if doi:
                if doi not in seen_dois:
                    unique_studies_data.append(study_data)
                    seen_dois.add(doi)
                    title = study_data.get('title')
                    if title:
                        title_lower = title.lower()
                        seen_titles[title_lower] = True

        # Second pass: consider studies without DOIs
        for study_data in all_studies_data:
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
            doi = study_data.get('doi')
            if not doi:
                title = study_data.get('title')
                if not title:
                    continue
                title_lower = title.lower()
                if title_lower in seen_titles:
                    continue
                duplicate_found = False
                for existing_title in seen_titles:
                     if existing_title and (existing_title in title_lower or title_lower in existing_title):
                        duplicate_found = True
                        break
                if not duplicate_found:
                    unique_studies_data.append(study_data)
                    seen_titles[title_lower] = True

        # Limit total studies to *process* for this request (config was MAX_EVIDENCE_TO_STORE)
        studies_to_process_data = unique_studies_data[:app.config['MAX_EVIDENCE_TO_STORE']]
        logger.info(f"Processing up to {len(studies_to_process_data)} unique studies with abstracts for this request.")

        if not studies_to_process_data:
            logger.warning("No usable evidence found after filtering.")
            # Continue to RAG analysis, which will handle no evidence
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies with abstracts could be retrieved.",
                "detailed_reasoning": "No relevant academic studies with abstracts could be retrieved.", # Added for consistency
                "simplified_reasoning": "No relevant academic studies with abstracts could be retrieved.", # Added for consistency
                "accuracy_score": 0.0, # Added for consistency
                "evidence": [],
                "keywords_used": keywords,
                "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

        # 3. Store Evidence in DB - optimized for larger study pools (Restore this section)
        stored_studies = []
        batch_size = 50  # Process in batches to avoid memory issues with very large pools

        try:
            # Create a map of existing DOIs to avoid redundant queries
            existing_dois_map = {}
            if studies_to_process_data:
                study_dois = [s.get('doi') for s in studies_to_process_data if s.get('doi')]
                if study_dois:
                    existing_studies_in_db = self.db.query(Study).filter(Study.doi.in_(study_dois)).all()
                    existing_dois_map = {study.doi: study for study in existing_studies_in_db}
                    logger.info(f"Found {len(existing_dois_map)} studies from this batch already in database via DOI lookup.")

            # Process studies in batches
            for i in range(0, len(studies_to_process_data), batch_size):
                batch = studies_to_process_data[i:i+batch_size]
                batch_objects = []

                for study_data in batch:
                    doi = study_data.get('doi')

                    # Skip if DOI exists and we already have it in the database
                    if doi and doi in existing_dois_map:
                        existing_study = existing_dois_map[doi]
                        # Update citation count if it changed significantly
                        new_citation_count = study_data.get('citation_count', 0)
                        if new_citation_count > existing_study.citation_count:
                             if (new_citation_count - existing_study.citation_count) / (existing_study.citation_count + 1e-6) > 0.1:
                                existing_study.citation_count = new_citation_count
                                self.db.add(existing_study)
                                logger.info(f"Updating citation count for existing study DOI: {doi} to {new_citation_count}")
                        stored_studies.append(existing_study) # Add the existing study object
                        continue # Move to next study in batch
                    # Double-check DOI doesn't exist (in case it was added by concurrent process)
                    elif doi:
                        existing_study = self.db.query(Study).filter(Study.doi == doi).first()
                        if existing_study:
                            # Update citation count if it changed
                            new_citation_count = study_data.get('citation_count', 0)
                            if existing_study.citation_count != new_citation_count:
                                existing_study.citation_count = new_citation_count
                                self.db.add(existing_study)
                            stored_studies.append(existing_study)
                            # Add to our cache for future lookups
                            existing_dois_map[doi] = existing_study
                            logger.debug(f"Found study with DOI: {doi} that wasn't in cache")
                            continue

                    # Create new study object
                    try:
                        study_obj = Study(
                            doi=study_data.get('doi'),
                            title=study_data.get('title'),
                            authors=study_data.get('authors'),
                            pub_date=study_data.get('pub_date'),
                            abstract=study_data.get('abstract'),
                            source_api=study_data.get('source_api'),
                            citation_count=study_data.get('citation_count', 0),
                            embedding=None # Initialize embedding as None
                        )
                        self.db.add(study_obj)
                        batch_objects.append(study_obj)
                    except SQLAlchemyError as e:
                        # Log the error but continue with other studies
                        logger.warning(f"Error adding study with DOI {doi}: {e}")
                        continue

                # Commit batch and refresh objects to get IDs
                try:
                    self.db.commit()
                    for study_obj in batch_objects:
                        self.db.refresh(study_obj)
                        stored_studies.append(study_obj)

                    logger.info(f"Committed batch. Added {len(batch_objects)} new studies to DB session.")
                except SQLAlchemyError as e:
                    # Rollback on batch error but continue with next batch
                    self.db.rollback()
                    logger.error(f"Error committing batch: {e}")

                    # If we hit duplicate DOIs, try processing studies one by one
                    if isinstance(e.orig, errors.UniqueViolation):
                        logger.info("Attempting to process studies one by one to handle duplicates")
                        for study_data in batch:
                            try:
                                # Check if DOI exists in database (might have been added by a concurrent request)
                                doi = study_data.get('doi')
                                if doi:
                                    existing = self.db.query(Study).filter(Study.doi == doi).first()
                                    if existing:
                                        stored_studies.append(existing)
                                        continue

                                # Create and add the study
                                study_obj = Study(
                                    doi=study_data.get('doi'),
                                    title=study_data.get('title'),
                                    authors=study_data.get('authors'),
                                    pub_date=study_data.get('pub_date'),
                                    abstract=study_data.get('abstract'),
                                    source_api=study_data.get('source_api'),
                                    citation_count=study_data.get('citation_count', 0),
                                    embedding=None
                                )
                                self.db.add(study_obj)
                                self.db.commit()
                                self.db.refresh(study_obj)
                                stored_studies.append(study_obj)
                            except SQLAlchemyError as individual_error:
                                # If the *individual* insert also fails due to duplicate, just log and continue
                                if isinstance(individual_error.orig, errors.UniqueViolation):
                                    logger.warning(f"Skipping individual study due to concurrent duplicate insert: {study_data.get('doi')}")
                                else:
                                    logger.warning(f"Error adding individual study: {individual_error}")
                                self.db.rollback()
                    else:
                        # For other SQLAlchemy errors during batch commit, we might want to stop or handle differently
                        logger.error(f"Non-duplicate error during batch commit: {e}")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error storing evidence: {e}")
            return {"error": "Database error storing evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error storing evidence: {e}")
            return {"error": "Unexpected error storing evidence.", "status": "failed"}

        logger.info(f"Finished processing and storing/updating {len(stored_studies)} studies in the database for this request.")

        # --- Vector Store Operations --- Restore this section
        # 4. Generate and Store Embeddings for studies that don't have them yet
        if stored_studies:
            self.vector_store.add_embeddings_to_studies(stored_studies)
        else:
            logger.info("No studies were stored or updated, skipping embedding generation.")

        # 5. Rank the studies fetched *for this request* based on combined score (Restore this section)
        logger.info(f"Ranking {len(stored_studies)} studies based on relevance and credibility...")
        try: # Add try-except block for claim embedding
             claim_embedding = embedding_model.encode([claim])
             claim_embedding_np = np.array(claim_embedding[0]).astype('float32')
        except Exception as e:
             logger.error(f"Error generating claim embedding: {e}")
             # Handle error appropriately, maybe return error status
             return {"error": "Failed to generate claim embedding for ranking.", "status": "failed"}

        ranked_studies = []
        for study in stored_studies:
            # Calculate cosine similarity (1 - cosine_distance)
            # pgvector cosine_distance is 1 - similarity, so lower is better.
            # Similarity = 1 - distance. Higher similarity is better.
            # Using numpy for calculation here as pgvector functions are in DB query
            # Ensure embedding exists before calculating relevance
            if study.embedding is None:
                 logger.warning(f"Study DOI {study.doi} missing embedding, cannot calculate relevance score.")
                 relevance_score = 0.0 # Assign a default score or skip? Assigning 0 for now.
            else:
                 try:
                     study_embedding_np = np.array(study.embedding).astype('float32')
                     # Cosine similarity formula: dot(A, B) / (norm(A) * norm(B))
                     dot_product = np.dot(claim_embedding_np, study_embedding_np)
                     norm_claim = np.linalg.norm(claim_embedding_np)
                     norm_study = np.linalg.norm(study_embedding_np)
                     if norm_claim == 0 or norm_study == 0:
                         relevance_score = 0.0 # Avoid division by zero
                     else:
                          # Clamp score between 0 and 1 (or -1 and 1 depending on embeddings, often near 0-1)
                         relevance_score = max(0.0, min(1.0, dot_product / (norm_claim * norm_study)))
                 except Exception as e:
                     logger.error(f"Error calculating relevance score for study {study.doi}: {e}")
                     relevance_score = 0.0 # Assign default on error


            # Calculate credibility score (log-scaled citation count)
            # Add 1 to avoid log(0). Adjust scale factor as needed.
            credibility_score = math.log10(study.citation_count + 1)

            # Combine scores (example: weighted average - adjust weights as needed)
            relevance_weight = 0.6
            credibility_weight = 0.4
            combined_score = (relevance_weight * relevance_score) + (credibility_weight * credibility_score)

            ranked_studies.append((study, combined_score))

        # Sort studies by combined score in descending order
        ranked_studies.sort(key=lambda item: item[1], reverse=True)

        # Select top K studies for RAG analysis
        top_k = app.config['RAG_TOP_K']
        top_ranked_studies = [study for study, score in ranked_studies[:top_k]]
        relevant_chunks = [study.abstract for study in top_ranked_studies if study.abstract]

        # Corrected log message to use the actual top_k value
        logger.info(f"Selected top {len(top_ranked_studies)} ranked studies (using RAG_TOP_K={top_k}) for LLM analysis.")

        if not relevant_chunks:
            logger.warning("No abstracts available from top ranked studies.")
            # Handle case where top studies have no abstract (unlikely given filters, but possible)
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "Top ranked studies had no abstracts available for analysis.",
                "detailed_reasoning": "Top ranked studies had no abstracts available for analysis.", # Added for consistency
                "simplified_reasoning": "Top ranked studies had no abstracts available for analysis.", # Added for consistency
                "accuracy_score": 0.0, # Added for consistency
                "evidence": [],
                "keywords_used": keywords,
                "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

        # 6. Analyze with LLM (Gemini)
        logger.info(f"Analyzing claim with {len(relevant_chunks)} most relevant abstracts via LLM")
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 7. Format and Return Output (Restore this section)
        # Retrieve details of the *actually used* evidence chunks for the response
        evidence_details = []
        if top_ranked_studies:
             # Filter based on citation count threshold AFTER retrieval
             # We already ranked by citation count, but can apply a minimum threshold if desired
             # For now, we'll just use the top K studies directly as they are already ranked
             used_studies_filtered = top_ranked_studies # Use the studies selected by ranking
             for study in used_studies_filtered:
                 evidence_details.append({
                    "title": study.title,
                    "link": f"https://doi.org/{study.doi}" if study.doi else None,
                    "doi": study.doi,
                    "abstract": clean_abstract(study.abstract),
                    "pub_date": study.pub_date,
                    "source_api": study.source_api,
                    "citation_count": study.citation_count
                })
             logger.info(f"Formatted evidence details for {len(used_studies_filtered)} top ranked studies.")

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."),
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "accuracy_score": analysis_result.get("accuracy_score", analysis_result.get("confidence", 0.0)),
            "evidence": evidence_details, # Provide details of evidence used in RAG
            "keywords_used": keywords,
            "category": category,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

        logger.info(f"RAG verification completed for claim: '{claim}'. Accuracy Score: {final_response['accuracy_score']}")
        return final_response


# --- End RAG Verification Service ---

# Initialize services
openalex_service = OpenAlexService()
crossref_service = CrossRefService()
semantic_scholar_service = SemanticScholarService()
gemini_service = GeminiService(gemini_model)

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    db_status = "disconnected"
    try:
        # Try connecting to the database
        connection = engine.connect()
        connection.close()
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    gemini_status = "ok" if gemini_model else "unavailable"
    embedding_status = "ok" if embedding_model else "unavailable"
    vector_db_status = db_status # Tied to the main DB status now
    # Add Semantic Scholar status (can be simple 'ok' as no explicit connection needed for basic search)
    semantic_scholar_status = "ok"


    return jsonify({
        "status": "ok",
        "service": "Factify RAG API",
        "version": "2.1.0", # Updated version
        "dependencies": {
            "database": db_status,
            "gemini_model": gemini_status,
            "embedding_model": embedding_status,
            "vector_storage": vector_db_status, # Renamed from vector_index
            "openalex_api": "ok", # Assuming ok if service initialized
            "crossref_api": "ok", # Assuming ok if service initialized
            "semantic_scholar_api": semantic_scholar_status # Add status
        }
    })

# --- Updated Claim Verification Endpoint ---
@app.route('/api/verify_claim', methods=['POST']) # Changed endpoint slightly
def verify_claim_rag():
    """
    Verifies a claim using the RAG workflow:
    1. Preprocesses claim (keywords, category) via LLM.
    2. Retrieves evidence from OpenAlex & CrossRef.
    3. Stores/updates evidence in PostgreSQL DB.
    4. Generates & stores embeddings in PostgreSQL (pgvector) if missing.
    5. Retrieves relevant studies via pgvector similarity search.
    6. Analyzes claim + chunks via LLM for verdict.
    """
    data = request.get_json()

    if not data or 'claim' not in data or not data['claim'].strip():
        return jsonify({"error": "Missing or empty 'claim' in request body"}), 400

    claim = data['claim']

    # Get DB session and initialize services that depend on it (Restore this section)
    db = next(get_db()) # Get session from generator
    try:
        # Initialize VectorStoreService with the current session
        vector_store = VectorStoreService(db)
        # Initialize RAG service (embedding model is now global)
        rag_service = RAGVerificationService(
            gemini_service,
            openalex_service,
            crossref_service,
            semantic_scholar_service,
            db, # RESTORED
            vector_store # RESTORED
        )

        result = rag_service.process_claim_request(claim)

        if result.get("status") == "failed":
             # Handle specific errors if needed, otherwise return generic server error
             return jsonify({"status": "error", "message": result.get("error", "Processing failed")}), 500

        response = {
            "status": "success",
            "result": result
        }
        return jsonify(response)

    except SQLAlchemyError as db_exc:
         logger.exception(f"Database error during claim verification: {db_exc}") # Log full traceback
         db.rollback() # Rollback on error
         return jsonify({"status": "error", "error": "Database operation failed."}), 500

    except Exception as e:
        logger.exception(f"Unhandled exception during claim verification: {e}") # Log full traceback
        return jsonify({
            "status": "error",
            "error": "An internal server error occurred.",
            "detail": str(e) # Optionally include detail in debug mode
        }), 500
    finally:
         db.close() # Ensure session is closed


# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    # Use debug=True only for development, ensure it's False in production
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))