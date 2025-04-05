# app.py
import os
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

# --- New Imports ---
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Date, MetaData, Table, text # Import text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc  # Import garbage collector for memory management
# --- End New Imports ---

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
    # --- New Config ---
    DATABASE_URL=os.getenv('DATABASE_URL', 'sqlite:///factify_rag.db'), # Use SQLite by default
    EMBEDDING_MODEL=os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    MAX_EVIDENCE_TO_RETRIEVE=int(os.getenv('MAX_EVIDENCE_TO_RETRIEVE', '50')), # Reduced from 200 to 50 per source
    MAX_EVIDENCE_TO_STORE=int(os.getenv('MAX_EVIDENCE_TO_STORE', '100')), # Reduced from 400 to 100 total
    RAG_TOP_K=int(os.getenv('RAG_TOP_K', '10')), # Reduced from 20 to 10 for memory efficiency
    EMBEDDING_BATCH_SIZE=int(os.getenv('EMBEDDING_BATCH_SIZE', '5')), # Smaller batch size for embeddings
    LOW_MEMORY_MODE=os.getenv('LOW_MEMORY_MODE', 'True') == 'True'  # Enable low memory mode by default
)

# Initialize Gemini API
gemini_api_key = app.config.get('GOOGLE_API_KEY')
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        # Don't initialize the model here, do it lazily when needed
        gemini_model = None
        logger.info("Gemini API configured successfully, model will be loaded when needed")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        gemini_model = None
else:
    logger.warning("GOOGLE_API_KEY not set. Gemini features will not work.")
    gemini_model = None

# --- New: Database Setup ---
Base = declarative_base()
engine = create_engine(app.config['DATABASE_URL'])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    # relevance_score = Column(Float, nullable=True) # Add if calculated

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Add citation_count column if it doesn't exist
def add_column_if_not_exists():
    try:
        # Get a connection
        with engine.connect() as conn:
            # Check if the column exists
            result = conn.execute(text("PRAGMA table_info(studies)"))
            columns = [row[1] for row in result]
            if 'citation_count' not in columns:
                logger.info("Adding citation_count column to studies table")
                conn.execute(text("ALTER TABLE studies ADD COLUMN citation_count INTEGER DEFAULT 0"))
                conn.commit() # Commit the change
            logger.info("Database schema check complete")
    except Exception as e:
        logger.error(f"Error checking or updating database schema: {e}")

# Perform the schema check/update
add_column_if_not_exists()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# --- End Database Setup ---

# --- New: Vector Store Setup (In-Memory FAISS) ---
embedding_model_name = app.config['EMBEDDING_MODEL']
try:
    # Initially set to None - will be loaded on demand
    embedding_model = None
    embedding_dimension = 384  # Default dimension for all-MiniLM-L6-v2
    # FAISS index will be initialized when needed
    index = None
    index_to_study_id_map = {} # Map FAISS index to DB study ID
    current_index_pos = 0
    logger.info(f"Configured for deferred loading of embedding model: {embedding_model_name}")
except Exception as e:
    logger.error(f"Failed to initialize vector store setup: {e}")
    embedding_model = None
    index = None
    embedding_dimension = 0

class VectorStoreService:
    def __init__(self, db_session, faiss_index, index_map):
        self.db = db_session
        self.index = faiss_index
        self.index_map = index_map
        self.current_pos = max(index_map.keys()) + 1 if index_map else 0
        # Track the current session's embeddings for query-specific search
        self.current_session_indices = []
        self.current_session_study_ids = []
        # Add a synchronization lock to prevent concurrent batch processing
        import threading
        self.embedding_lock = threading.Lock()
        # Lazy-loaded embedding model
        self.embedding_model = None

    def _ensure_embedding_model(self):
        """Lazily load the embedding model only when needed"""
        global embedding_model, index, embedding_dimension
        
        if embedding_model is None:
            logger.info(f"Loading embedding model {embedding_model_name}")
            embedding_model = SentenceTransformer(embedding_model_name)
            
            # Create FAISS index if it doesn't exist
            if index is None:
                # Determine embedding dimension dynamically
                dummy_embedding = embedding_model.encode(["test"])
                embedding_dimension = dummy_embedding.shape[1]
                index = faiss.IndexFlatL2(embedding_dimension)
                self.index = index
                logger.info(f"Initialized FAISS index with dimension {embedding_dimension}")
                
            # Force garbage collection after model loading
            gc.collect()
            
        return embedding_model

    def embed_and_store(self, studies):
        """Embeds abstracts and adds them to the FAISS index in batches."""
        # Ensure model is loaded
        self._ensure_embedding_model()
        
        if embedding_model is None or self.index is None:
            logger.error("Embedding model or FAISS index not available.")
            return

        # Filter studies with abstracts
        studies_with_abstracts = [study for study in studies if study.abstract]
        
        if not studies_with_abstracts:
            logger.warning("No abstracts found in studies to embed.")
            return
            
        total_studies = len(studies_with_abstracts)
        logger.info(f"Preparing to embed {total_studies} abstracts...")
        
        # Process in much smaller batches to avoid memory issues
        batch_size = app.config['EMBEDDING_BATCH_SIZE']  # Use much smaller batches (e.g., 5)
        total_embedded = 0
        
        # Use lock to prevent concurrent access to FAISS index
        with self.embedding_lock:
            for i in range(0, total_studies, batch_size):
                batch = studies_with_abstracts[i:i+batch_size]
                batch_size_actual = len(batch)
                
                study_ids = []
                chunks_to_embed = []
                
                for study in batch:
                    study_ids.append(study.id)
                    # Truncate very long abstracts to avoid memory issues
                    abstract = study.abstract[:5000] if study.abstract and len(study.abstract) > 5000 else study.abstract
                    chunks_to_embed.append(abstract)
                
                try:
                    logger.info(f"Embedding batch of {batch_size_actual} abstracts (batch {i//batch_size + 1}/{(total_studies-1)//batch_size + 1})...")
                    embeddings = embedding_model.encode(chunks_to_embed, show_progress_bar=False)
                    embeddings_np = np.array(embeddings).astype('float32')

                    # Add embeddings to FAISS index and update map
                    start_index = self.current_pos
                    self.index.add(embeddings_np)
                    
                    for j, study_id in enumerate(study_ids):
                        self.index_map[start_index + j] = study_id
                        # Track indices and study IDs for the current session
                        self.current_session_indices.append(start_index + j)
                        self.current_session_study_ids.append(study_id)
                    
                    self.current_pos += batch_size_actual
                    total_embedded += batch_size_actual
                    
                    logger.info(f"Successfully embedded batch. Total embedded: {total_embedded}/{total_studies}")
                    
                    # Clear memory after each batch
                    del embeddings, embeddings_np, chunks_to_embed
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error embedding batch: {e}")
                    # Continue with the next batch rather than failing completely
            
        logger.info(f"Completed embedding process. Total FAISS index size: {self.index.ntotal}")
        logger.info(f"Current session has {len(self.current_session_indices)} embeddings")
        gc.collect()  # Final cleanup

    def retrieve_relevant_chunks(self, claim_text, top_k):
        """Retrieves top_k relevant study abstracts based on claim similarity."""
        if not embedding_model or not self.index or self.index.ntotal == 0:
            logger.error("Embedding model, FAISS index not available, or index is empty.")
            return []

        try:
            claim_embedding = embedding_model.encode([claim_text])
            claim_embedding_np = np.array(claim_embedding).astype('float32')

            logger.info(f"Searching FAISS index (size {self.index.ntotal}) for top {top_k} chunks.")
            distances, indices = self.index.search(claim_embedding_np, top_k)

            if not indices.size:
                logger.warning("FAISS search returned no relevant indices.")
                return []

            # Get the corresponding study IDs from the map
            relevant_study_ids = [self.index_map[i] for i in indices[0] if i in self.index_map]

            if not relevant_study_ids:
                logger.warning("No matching study IDs found for FAISS indices.")
                return []

            # Retrieve abstracts from the database
            relevant_studies = self.db.query(Study).filter(Study.id.in_(relevant_study_ids)).all()
            # Return abstracts in the order of relevance found by FAISS
            ordered_abstracts = []
            faiss_ordered_ids = [self.index_map[i] for i in indices[0] if i in self.index_map]
            study_dict = {s.id: s.abstract for s in relevant_studies if s.abstract}
            for study_id in faiss_ordered_ids:
                 if study_id in study_dict:
                     ordered_abstracts.append(study_dict[study_id])


            logger.info(f"Retrieved {len(ordered_abstracts)} relevant abstracts.")
            return ordered_abstracts

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
            
    def retrieve_query_specific_chunks(self, claim_text, top_k):
        """Retrieves top_k relevant study abstracts only from those added in the current session."""
        # Ensure model is loaded
        self._ensure_embedding_model()
        
        if embedding_model is None or self.index is None:
            logger.error("Embedding model or FAISS index not available.")
            return []
            
        if not self.current_session_indices:
            logger.warning("No studies were embedded in the current session.")
            return []
            
        try:
            # Get the actual number of embeddings in the current session
            session_size = len(self.current_session_indices)
            logger.info(f"Creating temporary index with {session_size} embeddings specifically for this claim")
            
            # Create a temporary FAISS index with same dimension but only containing this session's embeddings
            dimension = self.index.d  # Get dimension from main index
            temp_index = faiss.IndexFlatL2(dimension)
            
            # Extract embeddings from main index through database and re-embed
            # This is more reliable than trying to extract from FAISS directly
            study_ids = self.current_session_study_ids
            
            # Get abstracts from the database for these studies
            abstracts_to_embed = []
            id_to_position = {}  # Maps study_id to position in temp index
            
            if study_ids:
                # Process in batches to avoid large IN clauses
                batch_size = app.config['EMBEDDING_BATCH_SIZE']
                for i in range(0, len(study_ids), batch_size):
                    batch_ids = study_ids[i:i+batch_size]
                    studies = self.db.query(Study).filter(Study.id.in_(batch_ids)).all()
                    
                    for study in studies:
                        if study.abstract:
                            # Truncate very long abstracts
                            abstract = study.abstract[:5000] if len(study.abstract) > 5000 else study.abstract
                            abstracts_to_embed.append(abstract)
                            id_to_position[study.id] = len(abstracts_to_embed) - 1
                    
                    # Clear memory after each batch
                    gc.collect()
            
            # If we have abstracts to embed, create temporary embeddings
            if abstracts_to_embed:
                logger.info(f"Re-embedding {len(abstracts_to_embed)} abstracts for query-specific search")
                # Use smaller batches for re-embedding to avoid memory issues
                batch_size = app.config['EMBEDDING_BATCH_SIZE']
                
                for i in range(0, len(abstracts_to_embed), batch_size):
                    batch = abstracts_to_embed[i:i+batch_size]
                    
                    # Embed and immediately add to the temp index (don't store in memory)
                    batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
                    batch_embeddings_np = np.array(batch_embeddings).astype('float32')
                    temp_index.add(batch_embeddings_np)
                    
                    # Clean up after each batch
                    del batch_embeddings, batch_embeddings_np
                    gc.collect()
                
                # Now search only this temporary index with our claim
                claim_embedding = embedding_model.encode([claim_text])
                claim_embedding_np = np.array(claim_embedding).astype('float32')
                
                # Search the temporary index
                temp_top_k = min(top_k, temp_index.ntotal)
                distances, indices = temp_index.search(claim_embedding_np, temp_top_k)
                
                logger.info(f"Performed semantic search on claim-specific pool of {temp_index.ntotal} studies")
                
                # Map back to study IDs
                relevant_study_ids = []
                for idx in indices[0]:
                    # Find which study_id corresponds to this position
                    for study_id, pos in id_to_position.items():
                        if pos == idx:
                            relevant_study_ids.append(study_id)
                            break
                
                # Retrieve abstracts from the database in batches
                ordered_abstracts = []
                
                if relevant_study_ids:
                    # Process in smaller batches to reduce memory usage
                    for i in range(0, len(relevant_study_ids), batch_size):
                        batch_ids = relevant_study_ids[i:i+batch_size]
                        relevant_studies = self.db.query(Study).filter(Study.id.in_(batch_ids)).all()
                        
                        # Create mapping for this batch
                        batch_dict = {s.id: s.abstract for s in relevant_studies if s.abstract}
                        
                        # Add to results in order of relevance
                        for study_id in batch_ids:
                            if study_id in batch_dict:
                                ordered_abstracts.append(batch_dict[study_id])
                        
                        # Clean up batch memory
                        del batch_dict
                        gc.collect()
                    
                    logger.info(f"Retrieved {len(ordered_abstracts)} most relevant abstracts from query-specific pool")
                    
                    # Clean up
                    del temp_index, distances, indices, claim_embedding, claim_embedding_np
                    gc.collect()
                    
                    return ordered_abstracts
            
            logger.warning("Could not find relevant abstracts in claim-specific pool")
            return []
                
        except Exception as e:
            logger.error(f"Error in query-specific vector search: {e}")
            return []
# --- End Vector Store Setup ---

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
        
        # OpenAlex limits to 200 results per page, enforcing a safe limit
        per_page = min(per_page, 100)  # Cap at 100 to avoid 403 errors
        
        params = {
            'search': search_query,
            'per-page': per_page,
            'filter': 'has_abstract:true,cited_by_count:>50', # Ensure abstracts and >50 citations
            'select': 'id,doi,title,authorships,publication_date,abstract_inverted_index,primary_location,cited_by_count' # Add cited_by_count
        }
        logger.info(f"Querying OpenAlex: {search_query} with per_page={per_page}")
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

            # Extract authors
            authors = ", ".join([a.get('author', {}).get('display_name', '') for a in paper.get('authorships', []) if a.get('author')])
            
            # Extract citation count
            citation_count = paper.get('cited_by_count', 0)

            # Only include if citation count > 5
            if citation_count > 50:
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
        self.timeout = 20 # Increased timeout for larger requests

    def search_works_by_keyword(self, keywords, rows=10):
        """Search CrossRef works using keyword query."""
        # Join keywords for search query if it's a list
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        
        # Cap rows at 100 for stability and to avoid server-side rejections
        rows = min(rows, 100)
        
        params = {
            'query.bibliographic': search_query,
            'rows': rows,
            'filter': 'has-abstract:true', # Try to filter for abstracts
            'select': 'DOI,title,author,abstract,published-print,published-online,created,is-referenced-by-count'  # Added citation count field
        }
        logger.info(f"Querying CrossRef: {search_query} with rows={rows}")
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

            # Only include if citation count > 5
            if citation_count > 50:
                processed.append({
                    "doi": item.get('DOI'),
                    "title": ". ".join(item.get('title', ['Untitled'])),
                    "authors": authors,
                    "pub_date": pub_date,
                    "abstract": item.get('abstract', '').strip().lstrip('<jats:p>').rstrip('</jats:p>'), # Basic cleaning
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
        self.timeout = 20

    def search_works_by_keyword(self, keywords, limit=10):
        """Search Semantic Scholar works using keyword query."""
        search_query = keywords if isinstance(keywords, str) else " ".join(keywords)
        limit = min(limit, 100) # API max limit is 100

        # Define the fields we want to retrieve - using correct field names
        fields = 'externalIds,title,authors,year,abstract,citationCount,publicationDate,journal'

        params = {
            'query': search_query,
            'limit': limit,
            'fields': fields,
            'offset': 0 # Start from the beginning
        }
        logger.info(f"Querying Semantic Scholar: {search_query} with limit={limit}")
        try:
            response = requests.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )

            # Handle rate limiting (HTTP 429)
            if response.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit, sleeping for 2 seconds.")
                time.sleep(2)
                # Consider adding retry logic or reducing frequency if this persists
                return None # For now, just return None on rate limit

            # Handle other potential errors
            if response.status_code >= 400:
                logger.error(f"Semantic Scholar API request failed with status {response.status_code}: {response.text}")
                return None

            response.raise_for_status() # Raise HTTP errors for other codes (e.g., 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode Semantic Scholar JSON response: {e}")
            return None

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
            abstract = item.get('abstract')
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

            # Only include if citation count > 5 (consistent with other sources)
            if citation_count > 50:
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
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None

    def _ensure_model_loaded(self):
        """Lazily initialize the Gemini model when needed"""
        global gemini_model
        
        if self.model is None:
            if gemini_model is not None:
                self.model = gemini_model
            elif self.api_key:
                try:
                    logger.info("Initializing Gemini model")
                    genai.configure(api_key=self.api_key)
                    self.model = genai.GenerativeModel('gemini-2.0-flash')
                    gemini_model = self.model  # Store for future use
                    logger.info("Successfully initialized Gemini model")
                    # Force garbage collection
                    gc.collect()
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini model: {e}")
                    self.model = None
            else:
                logger.error("Cannot initialize Gemini model: No API key provided")
        
        return self.model

    def preprocess_claim(self, claim):
        """Extracts keywords and category from the claim using Gemini."""
        # Ensure model is loaded
        model = self._ensure_model_loaded()
        
        if not model:
            logger.error("Gemini model not available for preprocessing.")
            # Fallback: Use simple keyword extraction (e.g., based on nouns/verbs) or return empty
            return {"keywords": [], "category": "unknown"}

        # Use a shorter prompt with fewer keywords for lower memory usage
        prompt = f"""
        Analyze the following claim to help find relevant academic research.
        Claim: "{claim}"

        1.  **Extract Key Terms:** Identify 3-4 most important keywords central to the claim.
        2.  **Categorize Claim:** Classify into ONE primary category:
            *   Health & Medicine
            *   Biology & Life Sciences
            *   Physical Sciences
            *   Earth & Environmental Sciences
            *   Technology & Engineering
            *   Social Sciences
            *   Humanities
            *   Mathematics & Computer Science
            *   General / Other

        Return ONLY a JSON object with keys "keywords" (list of strings) and "category" (single string).
        """

        try:
            logger.info(f"Sending preprocessing request to Gemini for claim: '{claim}'")
            response = model.generate_content(prompt)
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
        # Ensure model is loaded
        model = self._ensure_model_loaded()
        
        if not model:
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

        # Use a smaller subset of chunks if there are too many
        if len(evidence_chunks) > app.config['RAG_TOP_K']:
            evidence_chunks = evidence_chunks[:app.config['RAG_TOP_K']]
            logger.info(f"Limiting evidence chunks for RAG analysis to {app.config['RAG_TOP_K']}")

        # Format evidence for the prompt
        formatted_evidence = "\n\n".join([f"Evidence Chunk {i+1}:\n{chunk[:1000]}" for i, chunk in enumerate(evidence_chunks)])

        prompt = f"""
        You are a fact-checking analyst evaluating the following claim based only on the provided evidence chunks.

        Claim: "{claim}"

        Evidence Chunks:
        ---
        {formatted_evidence}
        ---

        Instructions:
        1. Carefully read the claim and evidence.
        2. Determine claim accuracy based on evidence.
        3. Create two summaries:
           a. Detailed scientific summary (3 sentences) referencing evidence chunks in format [EVIDENCE_CHUNK:N].
           b. Simplified summary (2 sentences) for general audience.
        4. Assign accuracy score between 0.0 (inaccurate) and 1.0 (accurate).
        5. Add verdict: "Supported", "Partially Supported", "Refuted", or "Inconclusive".

        Return ONLY a JSON object with keys "verdict", "detailed_reasoning", "simplified_reasoning", and "accuracy_score".
        """

        try:
            logger.info(f"Sending RAG analysis request to Gemini for claim: '{claim}'")
            response = model.generate_content(prompt)
            response_text = response.text

            # Robust JSON extraction
            try:
                json_start = response_text.index('{')
                json_end = response_text.rindex('}') + 1
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                # Validate expected keys
                if "verdict" in result and "detailed_reasoning" in result and "accuracy_score" in result:
                    logger.info(f"Successfully analyzed claim. Accuracy Score: {result['accuracy_score']}")
                    return result
                else:
                    raise ValueError("Missing or invalid keys in JSON response.")
            except (ValueError, IndexError) as e:
                logger.error(f"Failed to parse JSON from Gemini RAG response: {e}. Response: {response_text}")
                return {
                    "verdict": "Error",
                    "reasoning": "LLM analysis failed to produce valid JSON output.",
                    "detailed_reasoning": "Analysis failed to produce valid JSON output.",
                    "simplified_reasoning": "Analysis failed. Please try again.",
                    "accuracy_score": 0.0
                }

        except Exception as e:
            logger.error(f"Error during Gemini RAG analysis: {e}")
            return {
                "verdict": "Error",
                "reasoning": f"An error occurred during analysis: {str(e)}",
                "detailed_reasoning": f"An error occurred during analysis: {str(e)}",
                "simplified_reasoning": "An error occurred during analysis.",
                "accuracy_score": 0.0
            }

# --- End Gemini Service ---


# --- New RAG Verification Service ---
class RAGVerificationService:
    def __init__(self, gemini_service, openalex_service, crossref_service, semantic_scholar_service, db_session, vector_store_service):
        self.gemini = gemini_service
        self.openalex = openalex_service
        self.crossref = crossref_service
        self.semantic_scholar = semantic_scholar_service
        self.db = db_session
        self.vector_store = vector_store_service

    def process_claim_request(self, claim):
        """Orchestrates the entire RAG workflow for a claim."""
        start_time = time.time()
        logger.info(f"Starting RAG verification for claim: '{claim}'")

        # 1. Preprocess Claim (Keywords + Category)
        preprocessing_result = self.gemini.preprocess_claim(claim)
        keywords = preprocessing_result.get("keywords", [])
        category = preprocessing_result.get("category", "unknown")
        if not keywords:
            logger.warning("No keywords extracted, cannot retrieve evidence.")
            return {"error": "Could not extract keywords from claim.", "status": "failed"}

        logger.info(f"Extracted Keywords: {keywords}, Category: {category}")

        # 2. Retrieve Evidence - process each source sequentially to save memory
        max_results_per_source = app.config['MAX_EVIDENCE_TO_RETRIEVE']
        all_studies_data = []
        
        # Process OpenAlex
        logger.info("Retrieving studies from OpenAlex...")
        openalex_data = self.openalex.search_works_by_keyword(keywords, per_page=max_results_per_source)
        if openalex_data:
            openalex_studies = self.openalex.process_results(openalex_data)
            all_studies_data.extend(openalex_studies)
            # Release memory
            del openalex_data, openalex_studies
            gc.collect()
        
        # Process CrossRef
        logger.info("Retrieving studies from CrossRef...")
        crossref_data = self.crossref.search_works_by_keyword(keywords, rows=max_results_per_source)
        if crossref_data:
            crossref_studies = self.crossref.process_results(crossref_data)
            all_studies_data.extend(crossref_studies)
            # Release memory
            del crossref_data, crossref_studies
            gc.collect()
        
        # Process Semantic Scholar
        logger.info("Retrieving studies from Semantic Scholar...")
        semantic_scholar_data = self.semantic_scholar.search_works_by_keyword(keywords, limit=max_results_per_source)
        if semantic_scholar_data:
            semantic_scholar_studies = self.semantic_scholar.process_results(semantic_scholar_data)
            all_studies_data.extend(semantic_scholar_studies)
            # Release memory
            del semantic_scholar_data, semantic_scholar_studies
            gc.collect()

        logger.info(f"Retrieved a total of {len(all_studies_data)} studies from all sources.")

        # 3. Filter and deduplicate studies
        # Process in smaller batches to avoid memory pressure
        seen_dois = set()
        seen_titles = {}
        unique_studies_data = []
        
        # First filter by abstract presence and length
        filtered_studies = []
        for study in all_studies_data:
            if study.get('abstract') and len(study.get('abstract', '')) >= 50:
                filtered_studies.append(study)
        
        # Release memory
        del all_studies_data
        gc.collect()
        
        # Get batch size from config
        batch_size = app.config['EMBEDDING_BATCH_SIZE']
        
        # Process studies with DOIs first
        for study in filtered_studies:
            doi = study.get('doi')
            if doi and doi not in seen_dois:
                unique_studies_data.append(study)
                seen_dois.add(doi)
                
                # Also track title to avoid duplicate non-DOI studies
                title = study.get('title')
                if title:
                    title_lower = title.lower()
                    seen_titles[title_lower] = True
                    
            # Force garbage collection every 10 studies
            if len(unique_studies_data) % 10 == 0:
                gc.collect()
        
        # Then process non-DOI studies
        for study in filtered_studies:
            doi = study.get('doi')
            if not doi:
                title = study.get('title')
                if not title:
                    continue
                    
                title_lower = title.lower()
                if title_lower in seen_titles:
                    continue
                
                # Simple similarity check with only the first 100 titles
                duplicate_found = False
                for existing_title in list(seen_titles.keys())[:100]:
                    if existing_title and (existing_title in title_lower or title_lower in existing_title):
                        duplicate_found = True
                        break
                        
                if not duplicate_found:
                    unique_studies_data.append(study)
                    seen_titles[title_lower] = True
            
            # Force garbage collection every 10 studies
            if len(unique_studies_data) % 10 == 0:
                gc.collect()
        
        # Release memory
        del filtered_studies, seen_titles
        gc.collect()

        # Reset vector store's session tracking for this new query
        self.vector_store.current_session_indices = []
        self.vector_store.current_session_study_ids = []

        # Limit total studies to store
        unique_studies_data = unique_studies_data[:app.config['MAX_EVIDENCE_TO_STORE']]
        logger.info(f"Filtered down to {len(unique_studies_data)} unique studies with abstracts (after deduplication).")

        if not unique_studies_data:
            logger.warning("No usable evidence found after filtering.")
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "No relevant academic studies with abstracts could be retrieved.",
                "evidence": [],
                "processing_time_seconds": time.time() - start_time
            }

        # 4. Store Evidence in DB - optimized for memory usage
        stored_studies = []
        batch_size = min(10, app.config['EMBEDDING_BATCH_SIZE'] * 2)  # Small batches to avoid memory issues
        
        try:
            # Create a map of existing DOIs to avoid redundant queries
            existing_dois = {}
            
            # Process DOIs in batches
            study_dois = [s.get('doi') for s in unique_studies_data if s.get('doi')]
            if study_dois:
                for i in range(0, len(study_dois), batch_size):
                    batch_dois = study_dois[i:i+batch_size]
                    existing_studies = self.db.query(Study.doi, Study.id).filter(Study.doi.in_(batch_dois)).all()
                    for study in existing_studies:
                        existing_dois[study.doi] = study.id
                    # Free memory
                    gc.collect()
                logger.info(f"Found {len(existing_dois)} already existing studies in database")
            
            # Process studies in batches
            for i in range(0, len(unique_studies_data), batch_size):
                batch = unique_studies_data[i:i+batch_size]
                batch_objects = []
                
                for study_data in batch:
                    doi = study_data.get('doi')
                    
                    # Skip if DOI exists and we already have it in the database
                    if doi and doi in existing_dois:
                        # Fetch the existing study
                        existing_study = self.db.query(Study).filter(Study.id == existing_dois[doi]).first()
                        if existing_study:
                            # Update citation count if it changed
                            new_citation_count = study_data.get('citation_count', 0)
                            if existing_study.citation_count != new_citation_count:
                                existing_study.citation_count = new_citation_count
                                self.db.add(existing_study)
                            stored_studies.append(existing_study)
                            logger.debug(f"Using existing study with DOI: {doi}")
                            continue
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
                            existing_dois[doi] = existing_study.id
                            logger.debug(f"Found study with DOI: {doi} that wasn't in cache")
                            continue
                    
                    # Create new study object
                    try:
                        # Truncate very long abstracts to prevent memory issues
                        abstract = study_data.get('abstract', '')
                        if len(abstract) > 10000:  # Limit abstract size
                            abstract = abstract[:10000]
                            
                        study_obj = Study(
                            doi=study_data.get('doi'),
                            title=study_data.get('title'),
                            authors=study_data.get('authors'),
                            pub_date=study_data.get('pub_date'),
                            abstract=abstract,
                            source_api=study_data.get('source_api'),
                            citation_count=study_data.get('citation_count', 0)
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
                    
                    logger.info(f"Stored batch of {len(batch_objects)} new studies. Total stored: {len(stored_studies)}")
                except SQLAlchemyError as e:
                    # Rollback on batch error but continue with next batch
                    self.db.rollback()
                    logger.error(f"Error committing batch: {e}")
                    
                    # If we hit duplicate DOIs, try processing studies one by one
                    if "UNIQUE constraint failed" in str(e):
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
                                abstract = study_data.get('abstract', '')
                                if len(abstract) > 10000:  # Limit abstract size
                                    abstract = abstract[:10000]
                                    
                                study_obj = Study(
                                    doi=study_data.get('doi'),
                                    title=study_data.get('title'),
                                    authors=study_data.get('authors'),
                                    pub_date=study_data.get('pub_date'),
                                    abstract=abstract,
                                    source_api=study_data.get('source_api'),
                                    citation_count=study_data.get('citation_count', 0)
                                )
                                self.db.add(study_obj)
                                self.db.commit()
                                self.db.refresh(study_obj)
                                stored_studies.append(study_obj)
                            except SQLAlchemyError as individual_error:
                                self.db.rollback()
                                logger.warning(f"Error adding individual study: {individual_error}")
                
                # Clear memory after each batch
                gc.collect()
            
            # Release memory now that we've stored everything
            del unique_studies_data, existing_dois
            gc.collect()
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error storing evidence: {e}")
            return {"error": "Database error storing evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error storing evidence: {e}")
            return {"error": "Unexpected error storing evidence.", "status": "failed"}

        # 5. Embed and Store in Vector DB
        logger.info(f"Embedding all {len(stored_studies)} studies in the vector database")
        self.vector_store.embed_and_store(stored_studies) # Pass the SQLAlchemy objects

        # 6. Retrieve Relevant Chunks via vector search on CURRENT CLAIM'S studies only
        top_k = app.config['RAG_TOP_K']
        logger.info(f"Performing vector search on this claim's specific pool of {len(self.vector_store.current_session_indices)} studies")
        relevant_chunks = self.vector_store.retrieve_query_specific_chunks(claim, top_k=top_k)
        
        # In low memory mode, don't perform fallback
        if not relevant_chunks and not app.config['LOW_MEMORY_MODE']:
            logger.warning("Query-specific search returned no results, trying fallback")
            # Check if we have a reasonable number of global studies to search
            global_index_size = self.vector_store.index.ntotal if self.vector_store.index else 0
            if global_index_size > top_k:
                logger.info(f"Falling back to global search across all {global_index_size} studies in database")
                relevant_chunks = self.vector_store.retrieve_relevant_chunks(claim, top_k=top_k)
        
        # Clear stored studies from memory once we have the relevant chunks
        del stored_studies
        gc.collect()

        if not relevant_chunks:
            logger.warning("Could not retrieve any relevant chunks for analysis.")
            return {
                "claim": claim,
                "verdict": "Inconclusive",
                "reasoning": "Could not find relevant academic evidence to analyze this claim.",
                "evidence": [],
                "keywords_used": keywords,
                "category": category,
                "processing_time_seconds": round(time.time() - start_time, 2)
            }

        # 7. Analyze with LLM (Gemini)
        logger.info(f"Analyzing claim with {len(relevant_chunks)} most relevant abstracts via LLM")
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 8. Format and Return Output - Get evidence details in batches
        evidence_details = []
        
        if relevant_chunks:
            # Process abstracts in smaller batches to avoid large IN clauses
            abstract_to_study = {}
            batch_size = min(5, app.config['EMBEDDING_BATCH_SIZE'])
            
            for i in range(0, len(relevant_chunks), batch_size):
                batch_abstracts = relevant_chunks[i:i+batch_size]
                # Query for each batch
                batch_studies = self.db.query(Study).filter(Study.abstract.in_(batch_abstracts)).all()
                
                # Create mapping for this batch, filtering by citation count
                for study in batch_studies:
                    if study.citation_count > 5:
                        abstract_to_study[study.abstract] = study
                
                # Clean up batch memory
                gc.collect()
            
            # Get details in the order the chunks were retrieved
            for chunk in relevant_chunks:
                study = abstract_to_study.get(chunk)
                if study:
                    evidence_details.append({
                        "title": study.title,
                        "link": f"https://doi.org/{study.doi}" if study.doi else None,
                        "doi": study.doi,
                        "abstract": clean_abstract(study.abstract),
                        "pub_date": study.pub_date,
                        "source_api": study.source_api,
                        "citation_count": study.citation_count
                    })
            
            logger.info(f"Filtered evidence to {len(evidence_details)} studies with more than 5 citations")
            
            # Clean up memory
            del abstract_to_study, relevant_chunks
            gc.collect()

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."),
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "accuracy_score": analysis_result.get("accuracy_score", analysis_result.get("confidence", 0.0)),
            "evidence": evidence_details,
            "keywords_used": keywords,
            "category": category,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

        logger.info(f"RAG verification completed for claim: '{claim}'. Accuracy Score: {final_response['accuracy_score']}")
        
        # Final memory cleanup
        del analysis_result
        gc.collect()
        
        return final_response

# --- End RAG Verification Service ---


# Initialize services
openalex_service = OpenAlexService()
crossref_service = CrossRefService()
semantic_scholar_service = SemanticScholarService()
gemini_service = GeminiService(gemini_api_key)



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

    # Also report memory usage if possible
    memory_info = {}
    try:
        import psutil
        process = psutil.Process()
        memory_info = {
            "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "memory_percent": round(process.memory_percent(), 2)
        }
    except ImportError:
        memory_info = {"error": "psutil not available"}

    # Check model status with lazy loading
    gemini_status = "configured" if gemini_api_key else "unavailable"
    embedding_status = "configured" if embedding_model_name else "unavailable"
    
    return jsonify({
        "status": "ok",
        "service": "Factify RAG API",
        "version": "2.2.0", # Updated version
        "dependencies": {
            "database": db_status,
            "gemini_api": gemini_status,
            "embedding_model": embedding_status,
            "vector_store": "configured"
        },
        "memory": memory_info,
        "config": {
            "low_memory_mode": app.config.get('LOW_MEMORY_MODE'),
            "max_evidence_retrieve": app.config.get('MAX_EVIDENCE_TO_RETRIEVE'),
            "max_evidence_store": app.config.get('MAX_EVIDENCE_TO_STORE'),
            "rag_top_k": app.config.get('RAG_TOP_K'),
            "embedding_batch_size": app.config.get('EMBEDDING_BATCH_SIZE')
        }
    })

# --- Updated Claim Verification Endpoint ---
@app.route('/api/verify_claim', methods=['POST'])
def verify_claim_rag():
    """Verifies a claim using the RAG workflow with memory optimization."""
    # Start tracking request processing time
    start_time = time.time()
    
    # Force garbage collection at start of request
    gc.collect()
    
    data = request.get_json()

    if not data or 'claim' not in data or not data['claim'].strip():
        return jsonify({"error": "Missing or empty 'claim' in request body"}), 400

    claim = data['claim']
    # Truncate very long claims
    if len(claim) > 1000:
        claim = claim[:1000]
        logger.warning("Claim was truncated as it exceeded 1000 characters")

    # Get DB session and initialize services that depend on it
    db = next(get_db()) # Get session from generator
    try:
        # Pass the current db session and the global FAISS index/map
        vector_store = VectorStoreService(db, index, index_to_study_id_map)
        
        # Initialize services with lazy loading
        gemini_service = GeminiService(gemini_api_key)
        openalex_service = OpenAlexService(app.config.get('OPENALEX_EMAIL'))
        crossref_service = CrossRefService(app.config.get('OPENALEX_EMAIL'))
        semantic_scholar_service = SemanticScholarService()
        
        rag_service = RAGVerificationService(
            gemini_service,
            openalex_service,
            crossref_service,
            semantic_scholar_service,
            db,
            vector_store
        )

        result = rag_service.process_claim_request(claim)

        if result.get("status") == "failed":
            # Handle specific errors if needed, otherwise return generic server error
            return jsonify({"status": "error", "message": result.get("error", "Processing failed")}), 500

        # Add total processing time to response
        result["total_processing_time"] = round(time.time() - start_time, 2)
        
        response = {
            "status": "success",
            "result": result
        }
        
        # Force garbage collection before returning
        gc.collect()
        
        return jsonify(response)

    except Exception as e:
        logger.exception(f"Unhandled exception during claim verification: {e}") # Log full traceback
        return jsonify({
            "status": "error",
            "error": "An internal server error occurred.",
            "detail": str(e) # Optionally include detail in debug mode
        }), 500
    finally:
        db.close() # Ensure session is closed
        # Force garbage collection
        gc.collect()

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    
    # Log memory configuration
    logger.info(f"Starting with configuration:")
    logger.info(f"LOW_MEMORY_MODE: {app.config.get('LOW_MEMORY_MODE')}")
    logger.info(f"MAX_EVIDENCE_TO_RETRIEVE: {app.config.get('MAX_EVIDENCE_TO_RETRIEVE')}")
    logger.info(f"MAX_EVIDENCE_TO_STORE: {app.config.get('MAX_EVIDENCE_TO_STORE')}")
    logger.info(f"RAG_TOP_K: {app.config.get('RAG_TOP_K')}")
    logger.info(f"EMBEDDING_BATCH_SIZE: {app.config.get('EMBEDDING_BATCH_SIZE')}")
    
    # Check initial memory usage if possible
    try:
        import psutil
        process = psutil.Process()
        logger.info(f"Initial memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    except ImportError:
        logger.info("psutil not available for memory tracking")
    
    # Use debug=True only for development, ensure it's False in production
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
