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
    MAX_EVIDENCE_TO_RETRIEVE=int(os.getenv('MAX_EVIDENCE_TO_RETRIEVE', '200')), # Increased from 20 to 200 per source
    MAX_EVIDENCE_TO_STORE=int(os.getenv('MAX_EVIDENCE_TO_STORE', '400')), # Increased from 50 to 400 total
    RAG_TOP_K=int(os.getenv('RAG_TOP_K', '20')), # Number of chunks for RAG analysis

)

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
    embedding_model = SentenceTransformer(embedding_model_name)
    # Determine embedding dimension dynamically
    dummy_embedding = embedding_model.encode(["test"])
    embedding_dimension = dummy_embedding.shape[1]
    # FAISS index (consider persisting this for larger applications)
    index = faiss.IndexFlatL2(embedding_dimension)
    index_to_study_id_map = {} # Map FAISS index to DB study ID
    current_index_pos = 0
    logger.info(f"Initialized FAISS index with dimension {embedding_dimension} using model {embedding_model_name}")
except Exception as e:
    logger.error(f"Failed to initialize Sentence Transformer model or FAISS index: {e}")
    embedding_model = None
    index = None
    embedding_dimension = 0 # Default or handle error state


class VectorStoreService:
    def __init__(self, db_session, faiss_index, index_map):
        self.db = db_session
        self.index = faiss_index
        self.index_map = index_map
        self.current_pos = max(index_map.keys()) + 1 if index_map else 0
        # Track the current session's embeddings for query-specific search
        self.current_session_indices = []
        self.current_session_study_ids = []

    def embed_and_store(self, studies):
        """Embeds abstracts and adds them to the FAISS index in batches."""
        if not embedding_model or not self.index:
            logger.error("Embedding model or FAISS index not available.")
            return

        # Filter studies with abstracts
        studies_with_abstracts = [study for study in studies if study.abstract]
        
        if not studies_with_abstracts:
            logger.warning("No abstracts found in studies to embed.")
            return
            
        total_studies = len(studies_with_abstracts)
        logger.info(f"Preparing to embed {total_studies} abstracts...")
        
        # Process in batches to avoid memory issues
        batch_size = 50  # Adjust based on memory constraints
        total_embedded = 0
        
        for i in range(0, total_studies, batch_size):
            batch = studies_with_abstracts[i:i+batch_size]
            batch_size_actual = len(batch)
            
            study_ids = []
            chunks_to_embed = []
            
            for study in batch:
                study_ids.append(study.id)
                chunks_to_embed.append(study.abstract)
            
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
                
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                # Continue with the next batch rather than failing completely
                
        logger.info(f"Completed embedding process. Total FAISS index size: {self.index.ntotal}")
        logger.info(f"Current session has {len(self.current_session_indices)} embeddings")

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
        if not embedding_model or not self.index:
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
                studies = self.db.query(Study).filter(Study.id.in_(study_ids)).all()
                for i, study in enumerate(studies):
                    if study.abstract:
                        abstracts_to_embed.append(study.abstract)
                        id_to_position[study.id] = i
            
            # If we have abstracts to embed, create temporary embeddings
            if abstracts_to_embed:
                logger.info(f"Re-embedding {len(abstracts_to_embed)} abstracts for query-specific search")
                embeddings = embedding_model.encode(abstracts_to_embed, show_progress_bar=False)
                embeddings_np = np.array(embeddings).astype('float32')
                
                # Add to temporary index
                temp_index.add(embeddings_np)
                
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
                
                # Retrieve abstracts from the database
                if relevant_study_ids:
                    relevant_studies = self.db.query(Study).filter(Study.id.in_(relevant_study_ids)).all()
                    study_dict = {s.id: s.abstract for s in relevant_studies if s.abstract}
                    
                    # Order by relevance
                    ordered_abstracts = []
                    for study_id in relevant_study_ids:
                        if study_id in study_dict:
                            ordered_abstracts.append(study_dict[study_id])
                    
                    logger.info(f"Retrieved {len(ordered_abstracts)} most relevant abstracts from query-specific pool")
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
            'filter': 'has_abstract:true,cited_by_count:>5', # Ensure abstracts and >5 citations
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
            if citation_count > 5:
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
            if citation_count > 5:
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
            if citation_count > 5:
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
        2.  Determine whether the evidence, taken as a whole, supports, refutes, or is insufficient to judge the claim.
        3.  Create TWO different summaries:
            a. First, provide a DETAILED SCIENTIFIC summary (3-5 sentences) that references specific evidence chunks. **Crucially, when referencing evidence chunk numbers (e.g., chunk 5, or chunks 5, 12, and 18), you MUST wrap the reference in the format `[EVIDENCE_CHUNK:NUMBERS]` where NUMBERS is a comma-separated list of the chunk numbers. Example: `... findings from [EVIDENCE_CHUNK:5,12,18] indicate ...` or `... as shown in [EVIDENCE_CHUNK:3] ...`. Do NOT use any other format for referencing chunks.**
            b. Second, provide a SIMPLIFIED summary (2-3 sentences) in plain language that explains your verdict to a general audience without technical jargon or references to specific evidence chunks.
        4.  Assign a confidence score between 0.0 (no confidence) and 1.0 (high confidence) to your verdict.
        5.  Output your verdict as one word: "Supported", "Refuted", or "Inconclusive".

        Return ONLY a JSON object with the keys "verdict", "detailed_reasoning", "simplified_reasoning", and "confidence". Do not include any other text, markdown formatting, or explanations outside the JSON structure.

        Example Output:
        {{
            "verdict": "Inconclusive",
            "detailed_reasoning": "Evidence [EVIDENCE_CHUNK:3,7,12] suggests potential health risks, while [EVIDENCE_CHUNK:5,9,15] indicate possible benefits. Methodological limitations noted in [EVIDENCE_CHUNK:2,8,14] and the need for more controlled trials are emphasized. The contradictory findings and limited long-term studies ([EVIDENCE_CHUNK:4,11]) prevent a definitive conclusion.",
            "simplified_reasoning": "The research shows mixed results about this claim. Some studies suggest potential risks, while others show possible benefits. Scientists agree that more research is needed before drawing firm conclusions.",
            "confidence": 0.6
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
                if "verdict" in result and ("detailed_reasoning" in result or "reasoning" in result) and "confidence" in result:
                    logger.info(f"Successfully analyzed claim. Verdict: {result['verdict']}")
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
                     "confidence": 0.0
                 }

        except Exception as e:
            logger.error(f"Error during Gemini RAG analysis: {e}")
            return {
                "verdict": "Error",
                "reasoning": f"An unexpected error occurred during analysis: {str(e)}",
                "detailed_reasoning": f"An unexpected error occurred during analysis: {str(e)}",
                "simplified_reasoning": "An unexpected error occurred during analysis.",
                "confidence": 0.0
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

        # 2. Retrieve Evidence - from all three sources
        max_results_per_source = app.config['MAX_EVIDENCE_TO_RETRIEVE']
        openalex_data = self.openalex.search_works_by_keyword(keywords, per_page=max_results_per_source)
        crossref_data = self.crossref.search_works_by_keyword(keywords, rows=max_results_per_source)
        semantic_scholar_data = self.semantic_scholar.search_works_by_keyword(keywords, limit=max_results_per_source)

        openalex_studies = self.openalex.process_results(openalex_data)
        crossref_studies = self.crossref.process_results(crossref_data)
        semantic_scholar_studies = self.semantic_scholar.process_results(semantic_scholar_data)

        all_studies_data = openalex_studies + crossref_studies + semantic_scholar_studies
        logger.info(f"Retrieved {len(openalex_studies)} studies from OpenAlex, {len(crossref_studies)} from CrossRef, {len(semantic_scholar_studies)} from Semantic Scholar.")

        # --- Enhanced Filtering & Deduplication ---
        seen_dois = set()
        seen_titles = {} # For deduplicating studies without DOIs
        unique_studies_data = []
        
        # First pass: process studies with DOIs (preferred)
        for study_data in all_studies_data:
            # Skip studies without abstracts or with very short abstracts
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
                
            doi = study_data.get('doi')
            if doi:
                if doi not in seen_dois:
                    unique_studies_data.append(study_data)
                    seen_dois.add(doi)
                    # Also track title to avoid duplicate non-DOI studies
                    title = study_data.get('title')
                    if title:  # Check if title exists before calling lower()
                        title_lower = title.lower()
                        seen_titles[title_lower] = True
        
        # Second pass: consider studies without DOIs based on title similarity
        for study_data in all_studies_data:
            # Skip studies without abstracts or with very short abstracts
            if not study_data.get('abstract') or len(study_data.get('abstract', '')) < 50:
                continue
                
            doi = study_data.get('doi')
            # Only process studies without DOIs
            if not doi:
                title = study_data.get('title')
                # Skip if title is None or empty
                if not title:
                    continue
                    
                title_lower = title.lower()
                # Skip if we've seen this title
                if title_lower in seen_titles:
                    continue
                    
                # Check for high title similarity with existing titles
                duplicate_found = False
                for existing_title in seen_titles:
                    # Simple title similarity check (can be enhanced with better text matching)
                    if existing_title and (existing_title in title_lower or title_lower in existing_title):
                        duplicate_found = True
                        break
                        
                if not duplicate_found:
                    unique_studies_data.append(study_data)
                    seen_titles[title_lower] = True

        # Reset vector store's session tracking for this new query
        self.vector_store.current_session_indices = []
        self.vector_store.current_session_study_ids = []

        # Limit total studies to store - already increased in config to 400
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

        # 3. Store Evidence in DB - optimized for larger study pools
        stored_studies = []
        batch_size = 50  # Process in batches to avoid memory issues with very large pools
        
        try:
            # Create a map of existing DOIs to avoid redundant queries
            existing_dois = {}
            if unique_studies_data:
                study_dois = [s.get('doi') for s in unique_studies_data if s.get('doi')]
                if study_dois:
                    existing_studies = self.db.query(Study.doi, Study.id).filter(Study.doi.in_(study_dois)).all()
                    existing_dois = {study.doi: study.id for study in existing_studies}
                    logger.info(f"Found {len(existing_dois)} already existing studies in database")
            
            # Process studies in batches
            for i in range(0, len(unique_studies_data), batch_size):
                batch = unique_studies_data[i:i+batch_size]
                batch_objects = []
                
                for study_data in batch:
                    doi = study_data.get('doi')
                    
                    # Skip if DOI exists and we already have it in the database
                    if doi and doi in existing_dois:
                        # Fetch the existing study if needed - inline query to avoid race conditions
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
                        study_obj = Study(
                            doi=study_data.get('doi'),
                            title=study_data.get('title'),
                            authors=study_data.get('authors'),
                            pub_date=study_data.get('pub_date'),
                            abstract=study_data.get('abstract'),
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
                                study_obj = Study(
                                    doi=study_data.get('doi'),
                                    title=study_data.get('title'),
                                    authors=study_data.get('authors'),
                                    pub_date=study_data.get('pub_date'),
                                    abstract=study_data.get('abstract'),
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
                    continue
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error storing evidence: {e}")
            return {"error": "Database error storing evidence.", "status": "failed"}
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error storing evidence: {e}")
            return {"error": "Unexpected error storing evidence.", "status": "failed"}

        # 4. Embed and Store in Vector DB
        logger.info(f"Embedding all {len(stored_studies)} studies in the vector database")
        self.vector_store.embed_and_store(stored_studies) # Pass the SQLAlchemy objects

        # 5. Retrieve Relevant Chunks via vector search on CURRENT CLAIM'S studies only
        top_k = app.config['RAG_TOP_K']
        logger.info(f"Performing vector search on this claim's specific pool of {len(self.vector_store.current_session_indices)} studies")
        relevant_chunks = self.vector_store.retrieve_query_specific_chunks(claim, top_k=top_k)
        
        # Only fallback if absolutely necessary and configured to do so
        if not relevant_chunks:
            logger.warning("Query-specific search returned no results")
            # This should rarely happen since we're searching the pool we just fetched
            # unless there was an error in embedding or the pool had no viable abstracts
            
            # Check if we have a reasonable number of global studies to search
            global_index_size = self.vector_store.index.ntotal
            if global_index_size > top_k:
                logger.info(f"Falling back to global search across all {global_index_size} studies in database")
                relevant_chunks = self.vector_store.retrieve_relevant_chunks(claim, top_k=top_k)
            else:
                logger.warning("No fallback possible - insufficient studies in global index")

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

        # 6. Analyze with LLM (Gemini)
        logger.info(f"Analyzing claim with {len(relevant_chunks)} most relevant abstracts via LLM")
        analysis_result = self.gemini.analyze_with_rag(claim, relevant_chunks)

        # 7. Format and Return Output
        # Retrieve details of the *actually used* evidence chunks for the response
        evidence_details = []
        if relevant_chunks:
             # Find the studies corresponding to the chunks used in analysis
             used_studies = self.db.query(Study).filter(Study.abstract.in_(relevant_chunks)).limit(top_k).all()
             # Create a dict for quick lookup, filtering out studies with low citation counts
             abstract_to_study = {study.abstract: study for study in used_studies if study.citation_count > 5}
             # Get details in the order the chunks were retrieved, skipping low-citation studies
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

        final_response = {
            "claim": claim,
            "verdict": analysis_result.get("verdict", "Error"),
            "reasoning": analysis_result.get("reasoning", "Analysis failed."),
            "detailed_reasoning": analysis_result.get("detailed_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "simplified_reasoning": analysis_result.get("simplified_reasoning", analysis_result.get("reasoning", "Analysis failed.")),
            "confidence": analysis_result.get("confidence", 0.0),
            "evidence": evidence_details, # Provide details of evidence used in RAG
            "keywords_used": keywords,
            "category": category,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

        logger.info(f"RAG verification completed for claim: '{claim}'. Verdict: {final_response['verdict']}")
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
    faiss_status = "ok" if index is not None else "unavailable"
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
            "vector_index": faiss_status,
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
    3. Stores evidence in DB.
    4. Embeds evidence & stores in Vector DB (FAISS).
    5. Retrieves relevant chunks via Vector Search.
    6. Analyzes claim + chunks via LLM for verdict.
    """
    data = request.get_json()

    if not data or 'claim' not in data or not data['claim'].strip():
        return jsonify({"error": "Missing or empty 'claim' in request body"}), 400

    claim = data['claim']

    # Get DB session and initialize services that depend on it
    db = next(get_db()) # Get session from generator
    try:
        # Pass the current db session and the global FAISS index/map
        vector_store = VectorStoreService(db, index, index_to_study_id_map)
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

        response = {
            "status": "success",
            "result": result
        }
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


# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    # Use debug=True only for development, ensure it's False in production
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
