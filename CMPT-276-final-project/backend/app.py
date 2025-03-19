# app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import time
import concurrent.futures
import google.generativeai as genai
from dotenv import load_dotenv

import logging

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
    OPENALEX_EMAIL=os.getenv('OPENALEX_EMAIL', 'factify@example.com'),
    MAX_PAPERS_TO_RETRIEVE=int(os.getenv('MAX_PAPERS_TO_RETRIEVE', '200')),
    TOP_PAPERS_TO_EVALUATE=int(os.getenv('TOP_PAPERS_TO_EVALUATE', '30')),
    CONFIDENCE_THRESHOLD=float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
)

# Initialize Gemini API
gemini_api_key = app.config.get('GOOGLE_API_KEY')
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite-001')
else:
    print("WARNING: GOOGLE_API_KEY not set. Gemini features will not work.")
    gemini_model = None


# OpenAlex Service
class OpenAlexService:
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email=None):
        self.email = email or app.config.get('OPENALEX_EMAIL')
        self.headers = {'User-Agent': f'Factify ({self.email})'}
    
    def _reconstruct_abstract_from_inverted_index(self, abstract_inverted_index):
        """
        Reconstructs the abstract text from OpenAlex's abstract_inverted_index format
        
        The abstract_inverted_index is a dictionary where:
        - keys are words
        - values are lists of positions where that word appears in the abstract
        
        This method reconstructs the original text by placing each word in its correct position.
        """
        if not abstract_inverted_index:
            return ""
        
        # Create a list to hold words at their positions
        max_position = 0
        for positions in abstract_inverted_index.values():
            if positions and max(positions) > max_position:
                max_position = max(positions)
        
        # Initialize words list with empty strings
        words = [""] * (max_position + 1)
        
        # Place each word in its position(s)
        for word, positions in abstract_inverted_index.items():
            for position in positions:
                words[position] = word
        
        # Join words with spaces to form the abstract
        return " ".join(words)
    
    def search_works(self, query, filter_params=None, page=1, per_page=10):
        """Search for academic works in OpenAlex"""
        params = {
            'search': query,
            'page': page,
            'per-page': per_page,
            # Do not use 'select' as it restricts the returned fields 
            # and can cause issues with the rest of our application
        }
        
        # Always add the has_abstract filter to only get papers with abstracts
        if filter_params:
            # Add to existing filters
            filters = []
            for key, value in filter_params.items():
                filters.append(f"{key}:{value}")
            filters.append("has_abstract:true")
            params['filter'] = ','.join(filters)
        else:
            # Create the filter if none exists
            params['filter'] = "has_abstract:true"
        
        response = requests.get(
            f"{self.BASE_URL}/works",
            params=params,
            headers=self.headers
        )
        
        # Handle rate limiting
        if response.status_code == 429:
            time.sleep(1)  # Wait before retrying
            return self.search_works(query, filter_params, page, per_page)
        
        response.raise_for_status()
        return response.json()
    
    def get_work_by_id(self, work_id):
        """Fetch a specific work by its OpenAlex ID"""
        # Handle both full URLs and bare IDs
        if work_id.startswith('http'):
            # Extract the ID from the URL
            work_id = work_id.split('/')[-1]
        
        # Add the "W" prefix if it's not already there
        if not work_id.startswith('W'):
            work_id = f"W{work_id}"
            
        response = requests.get(
            f"{self.BASE_URL}/works/{work_id}",
            params={'select': 'abstract_inverted_index,abstract'},  # Request abstract information
            headers=self.headers
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_author_works(self, author_id, page=1, per_page=10):
        """Get works by a specific author"""
        params = {
            'filter': f"author.id:{author_id}",
            'page': page,
            'per-page': per_page
        }
        
        response = requests.get(
            f"{self.BASE_URL}/works",
            params=params,
            headers=self.headers
        )
        
        response.raise_for_status()
        return response.json()
    
    def get_citation_timeline(self, work_id):
        """Get citation history for a work"""
        work = self.get_work_by_id(work_id)
        
        params = {
            'filter': f"cites:{work_id}",
            'page': 1,
            'per-page': 100,
            'sort': 'publication_date'
        }
        
        response = requests.get(
            f"{self.BASE_URL}/works",
            params=params,
            headers=self.headers
        )
        
        response.raise_for_status()
        
        # Process results to create a timeline
        citing_works = response.json().get('results', [])
        timeline = {}
        
        for citing_work in citing_works:
            year = citing_work.get('publication_year')
            if year:
                if year not in timeline:
                    timeline[year] = 0
                timeline[year] += 1
        
        return {
            'work_id': work_id,
            'title': work.get('title'),
            'timeline': [{'year': year, 'count': count} for year, count in sorted(timeline.items())]
        }


# Gemini Service
class GeminiService:
    def __init__(self, model=None):
        self.model = model
    
    def decompose_claim(self, claim):
        """Break down a claim into sub-claims for verification"""
        if not self.model:
            # Fallback when model isn't available
            return {
                "sub_claims": [
                    {"id": 1, "text": claim, "key_entities": []}
                ]
            }
            
        prompt = f"""
        As a fact-checking AI, your task is to break down the following claim into smaller, verifiable sub-claims.
        
        CLAIM: {claim}
        
        Please identify 3-5 distinct sub-claims that:
        1. Are specific and factual (not opinions)
        2. Can be independently verified
        3. Together cover the main assertions of the original claim
        
        Return the sub-claims in a JSON format with the following structure:
        {{
            "sub_claims": [
                {{
                    "id": 1,
                    "text": "First sub-claim statement",
                    "key_entities": ["entity1", "entity2"]
                }},
                ...
            ]
        }}
        
        IMPORTANT: Only return the JSON with no other text or explanations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            logger.info(f"Response by the Gemini model: {response}")
            
            # Extract JSON from response
            response_text = response.text
            if "```" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()

            logger.info(f"JSON response after extracting and stripping: {response_text}")
            
            return json.loads(response_text)
        except Exception as e:
            print(f"Error decomposing claim: {str(e)}")
            return {
                "sub_claims": [
                    {"id": 1, "text": claim, "key_entities": []}
                ]
            }
    
    def evaluate_evidence(self, claim, evidence_list):
        """Evaluate evidence against a claim to determine confidence score"""
        if not self.model or not evidence_list:
            # Fallback when model isn't available or no evidence
            return {
                "stance": "insufficient",
                "confidence_score": 0.0,
                "reasoning": "Insufficient evidence available",
                "key_support_points": [],
                "key_refutation_points": [],
                "evidence_gaps": ["No evidence provided"]
            }
        
        # Define the maximum evidence items per batch
        MAX_EVIDENCE_PER_BATCH = 15
        
        # If evidence list is small enough, process directly
        if len(evidence_list) <= MAX_EVIDENCE_PER_BATCH:
            return self._evaluate_single_batch(claim, evidence_list)
        
        # For larger evidence sets, process in batches
        logger.info(f"Processing large evidence set with {len(evidence_list)} items in batches")
        
        # Split evidence into batches
        evidence_batches = [
            evidence_list[i:i + MAX_EVIDENCE_PER_BATCH] 
            for i in range(0, len(evidence_list), MAX_EVIDENCE_PER_BATCH)
        ]
        
        batch_evaluations = []
        # Process each batch
        for i, batch in enumerate(evidence_batches):
            logger.info(f"Processing evidence batch {i+1}/{len(evidence_batches)}")
            batch_eval = self._evaluate_single_batch(claim, batch)
            batch_evaluations.append(batch_eval)
        
        # Combine the evaluations from all batches
        combined_evaluation = self._combine_evaluations(batch_evaluations)
        
        # Add a note about batch processing
        combined_evaluation["reasoning"] = "Evidence was processed in multiple batches due to its large volume. " + combined_evaluation["reasoning"]
        
        return combined_evaluation
    
    def _evaluate_single_batch(self, claim, evidence_batch):
        """Evaluate a single batch of evidence against a claim"""
        evidence_text = "\n\n".join([
            f"EVIDENCE {i+1}:\nTitle: {evidence.get('title', 'Untitled')}\n"
            f"Source: {evidence.get('source', 'Unknown')}\n"
            f"Year: {evidence.get('year', 'Unknown')}\n"
            f"Summary: {evidence.get('abstract', 'No abstract available')}"
            for i, evidence in enumerate(evidence_batch)
        ])
        
        # Add context about evidence volume
        evidence_context = ""
        if len(evidence_batch) < 5:
            evidence_context = "NOTE: Only a small number of relevant papers with abstracts were found. Please evaluate carefully based on this limited evidence."
        
        prompt = f"""
        As a fact-checking AI, analyze the following claim against the provided evidence.
        
        CLAIM: {claim}
        
        {evidence_context}
        
        {evidence_text}
        
        Evaluate whether the evidence supports, refutes, or is insufficient to judge the claim.
        
        Return your evaluation as JSON with the following structure:
        {{
            "stance": "supports/refutes/insufficient",
            "confidence_score": 0.0-1.0,
            "reasoning": "Your step-by-step analysis",
            "key_support_points": ["point1", "point2"...],
            "key_refutation_points": ["point1", "point2"...],
            "evidence_gaps": ["gap1", "gap2"...]
        }}
        
        IMPORTANT: Only return the JSON with no other text or explanations.
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            response_text = response.text
            if "```" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            logger.info(f"JSON response for evaluation for a single batch for claim: {claim} after extracting and stripping: {response_text}")
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error evaluating evidence batch: {str(e)}")
            return {
                "stance": "insufficient",
                "confidence_score": 0.0,
                "reasoning": "Failed to structure the analysis properly",
                "key_support_points": [],
                "key_refutation_points": [],
                "evidence_gaps": ["Technical error in processing the evidence"]
            }
    
    def _combine_evaluations(self, evaluations):
        """Combine multiple batch evaluations into a single evaluation result"""
        if not evaluations:
            return {
                "stance": "insufficient",
                "confidence_score": 0.0,
                "reasoning": "No evaluations to combine",
                "key_support_points": [],
                "key_refutation_points": [],
                "evidence_gaps": ["No evidence provided"]
            }
        
        # Count stances to determine the overall stance
        stance_counts = {"supports": 0, "refutes": 0, "insufficient": 0}
        for eval in evaluations:
            stance = eval.get("stance", "insufficient")
            stance_counts[stance] += 1
        
        # Determine the dominant stance
        if stance_counts["supports"] > stance_counts["refutes"]:
            overall_stance = "supports"
        elif stance_counts["refutes"] > stance_counts["supports"]:
            overall_stance = "refutes"
        else:
            overall_stance = "insufficient"
        
        # Calculate the average confidence for the dominant stance
        relevant_evals = [e for e in evaluations if e.get("stance") == overall_stance]
        if relevant_evals:
            overall_confidence = sum(e.get("confidence_score", 0) for e in relevant_evals) / len(relevant_evals)
        else:
            overall_confidence = 0.0
        
        # Compile reasoning points and evidence
        all_reasoning = []
        all_support_points = []
        all_refutation_points = []
        all_evidence_gaps = []
        
        for i, eval in enumerate(evaluations):
            batch_num = i + 1
            if eval.get("reasoning"):
                all_reasoning.append(f"Batch {batch_num}: {eval.get('reasoning')}")
            
            all_support_points.extend(eval.get("key_support_points", []))
            all_refutation_points.extend(eval.get("key_refutation_points", []))
            all_evidence_gaps.extend(eval.get("evidence_gaps", []))
        
        # Remove duplicates while preserving order
        def deduplicate(items):
            seen = set()
            return [x for x in items if not (x in seen or seen.add(x))]
        
        all_support_points = deduplicate(all_support_points)
        all_refutation_points = deduplicate(all_refutation_points)
        all_evidence_gaps = deduplicate(all_evidence_gaps)
        
        # Compile combined reasoning
        combined_reasoning = "Based on analysis across multiple evidence batches:\n" + "\n\n".join(all_reasoning)
        
        # Limit the number of points to prevent excessively long results
        MAX_POINTS = 10
        
        return {
            "stance": overall_stance,
            "confidence_score": overall_confidence,
            "reasoning": combined_reasoning,
            "key_support_points": all_support_points[:MAX_POINTS],
            "key_refutation_points": all_refutation_points[:MAX_POINTS],
            "evidence_gaps": all_evidence_gaps[:MAX_POINTS]
        }

    def generate_summary(self, claim, verification_results):
        """Generate a narrative text summary of the verification results"""
        if not self.model:
            return "Summary generation is not available without a language model."
            
        # Prepare the input for the summary by extracting key information
        sub_claims = verification_results.get('sub_claims', [])
        verdict = verification_results.get('verdict', 'Inconclusive')
        confidence = verification_results.get('overall_confidence', 0)
        
        # Count stances across sub-claims
        stance_counts = {"supports": 0, "refutes": 0, "insufficient": 0, "error": 0}
        for sub_claim in sub_claims:
            stance = sub_claim.get('evaluation', {}).get('stance', 'insufficient')
            stance_counts[stance] += 1
        
        # Collect key points
        key_supports = []
        key_refutations = []
        
        for sub_claim in sub_claims:
            eval_data = sub_claim.get('evaluation', {})
            key_supports.extend(eval_data.get('key_support_points', [])[:2])  # Get top 2 support points
            key_refutations.extend(eval_data.get('key_refutation_points', [])[:2])  # Get top 2 refutation points
        
        # Remove duplicates
        key_supports = list(set(key_supports))[:5]  # Limit to top 5
        key_refutations = list(set(key_refutations))[:5]  # Limit to top 5
        
        # Count total papers analyzed
        total_papers = sum(sub_claim.get('evidence_count', 0) for sub_claim in sub_claims)
        
        # Build the prompt for the LLM
        prompt = f"""
        As a scientific research analyst, write a concise, objective summary of the verification results for the following claim:
        
        CLAIM: {claim}
        
        THE VERIFICATION RESULTS:
        - Overall verdict: {verdict} (confidence score: {confidence:.2f})
        - {len(sub_claims)} sub-claims were analyzed using {total_papers} academic papers
        - Stance distribution: {stance_counts['supports']} supporting, {stance_counts['refutes']} refuting, {stance_counts['insufficient']} insufficient evidence
        
        KEY SUPPORTING EVIDENCE:
        {' '.join([f"- {point}" for point in key_supports])}
        
        KEY REFUTING EVIDENCE:
        {' '.join([f"- {point}" for point in key_refutations])}
        
        Please provide a concise (4-6 sentences) summary that:
        1. Explains the main finding regarding the claim's validity
        2. Mentions the strength of evidence (number of papers and their findings)
        3. Highlights the most significant supporting and/or countering evidence
        4. Provides a balanced conclusion about what the research indicates

        Write in a clear, professional tone suitable for a fact-checking platform. Avoid technical jargon.
        Return ONLY the summary paragraph with no additional text, headers or formatting.
        """
        
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Clean up any markdown or other formatting
            if summary.startswith("```") and summary.endswith("```"):
                summary = summary[3:-3].strip()
                
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate a summary due to a technical error."


# Verification Service
class VerificationService:
    def __init__(self, openalex_service, gemini_service):
        self.openalex = openalex_service
        self.gemini = gemini_service
    
    # Helper method to reconstruct abstracts - delegates to the OpenAlexService implementation
    def _reconstruct_abstract_from_inverted_index(self, abstract_inverted_index):
        """Delegates to OpenAlexService to reconstruct an abstract from the inverted index format"""
        return self.openalex._reconstruct_abstract_from_inverted_index(abstract_inverted_index)
    
    def verify_claim(self, claim):
        """Verify a claim by decomposing it, finding evidence, and evaluating"""
        # Step 1: Decompose the claim into sub-claims
        logger.info(f"Starting verification for claim: '{claim}'")
        decomposition = self.gemini.decompose_claim(claim)
        sub_claims = decomposition.get('sub_claims', [])
        
        # If no sub-claims were found, use the original claim
        if not sub_claims:
            logger.warning("No sub-claims generated, using the original claim")
            sub_claims = [{"id": 1, "text": claim, "key_entities": []}]
        
        logger.info(f"Generated {len(sub_claims)} sub-claims for verification")
        
        # Step 2: Gather evidence for each sub-claim in parallel
        verification_results = []
        
        # Limit the number of parallel workers to avoid overwhelming the system
        max_workers = min(5, len(sub_claims))
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_subclaim = {executor.submit(self._verify_sub_claim, sub_claim): sub_claim for sub_claim in sub_claims}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_subclaim):
                sub_claim = future_to_subclaim[future]
                try:
                    result = future.result()
                    verification_results.append(result)
                    logger.info(f"Completed verification for sub-claim: '{sub_claim['text']}' with stance: {result.get('evaluation', {}).get('stance', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error during verification of sub-claim '{sub_claim['text']}': {str(e)}")
                    verification_results.append({
                        "sub_claim": sub_claim["text"],
                        "evidence": [],
                        "evaluation": {
                            "stance": "error",
                            "confidence_score": 0,
                            "reasoning": f"Error during verification: {str(e)}"
                        }
                    })
        
        # Step 3: Calculate overall confidence score
        if verification_results:
            overall_confidence = self._calculate_overall_confidence(verification_results)
        else:
            overall_confidence = 0
        
        # Determine the verdict
        verdict = self._determine_verdict(overall_confidence)
        logger.info(f"Claim verification completed with verdict: {verdict}, confidence: {overall_confidence:.2f}")
        
        # Step 4: Prepare verification results
        result = {
            "original_claim": claim,
            "sub_claims": verification_results,
            "overall_confidence": overall_confidence,
            "verdict": verdict
        }
        
        # Step 5: Generate a narrative summary
        try:
            summary = self.gemini.generate_summary(claim, result)
            result["text_summary"] = summary
            logger.info("Generated narrative summary for the verification results")
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            result["text_summary"] = "A summary could not be generated for these verification results."
        
        return result
    
    def _verify_sub_claim(self, sub_claim):
        """Verify an individual sub-claim"""
        # Get evidence for the sub-claim
        query = sub_claim["text"]
        logger.info(f"Verifying sub-claim: '{query}'")
        
        # Gather evidence
        evidence = self._gather_evidence(query)
        
        if not evidence:
            logger.warning(f"No evidence found for sub-claim: '{query}'")
            return {
                "sub_claim": sub_claim["text"],
                "key_entities": sub_claim.get("key_entities", []),
                "evidence": [],
                "evaluation": {
                    "stance": "insufficient",
                    "confidence_score": 0.0,
                    "reasoning": "No evidence was found to evaluate this claim.",
                    "key_support_points": [],
                    "key_refutation_points": [],
                    "evidence_gaps": ["No relevant academic papers with abstracts found"]
                }
            }
        
        # Enhanced feedback for limited evidence
        if len(evidence) < 5:  # If we have very few papers
            logger.warning(f"Very limited evidence ({len(evidence)} papers) for sub-claim: '{query}'")
            # Still proceed with evaluation, but note the limitation
            
        # Evaluate the evidence against the sub-claim
        logger.info(f"Evaluating {len(evidence)} evidence items for sub-claim: '{query}'")
        evaluation = self.gemini.evaluate_evidence(query, evidence)
        
        # If we had very limited evidence, add this information to the reasoning
        if len(evidence) < 5:
            if "reasoning" in evaluation:
                evaluation["reasoning"] = f"[LIMITED EVIDENCE WARNING: Only {len(evidence)} papers with abstracts were found] " + evaluation["reasoning"]
            if "evidence_gaps" in evaluation and isinstance(evaluation["evidence_gaps"], list):
                evaluation["evidence_gaps"].insert(0, f"Limited number of papers with abstracts ({len(evidence)}) for evaluation")
        
        # Return all evidence to show all 30 papers 
        return {
            "sub_claim": sub_claim["text"],
            "key_entities": sub_claim.get("key_entities", []),
            "evidence": evidence,  # Include all evidence
            "evidence_count": len(evidence),
            "evaluation": evaluation
        }
    
    def _gather_evidence(self, query):
        """Gather evidence for a claim from OpenAlex"""
        # Get the paper retrieval configuration from app settings
        max_papers_to_retrieve = app.config.get('MAX_PAPERS_TO_RETRIEVE', 200)
        top_papers_to_evaluate = app.config.get('TOP_PAPERS_TO_EVALUATE', 30)
        
        try:
            # Search for academic papers - retrieve more papers initially
            # This now includes the has_abstract:true filter from the search_works method
            # We'll leverage OpenAlex's built-in relevance sorting which is automatic when using the search parameter
            search_results = self.openalex.search_works(query, per_page=max_papers_to_retrieve)
            
            # Make sure we got a valid response
            if not search_results or not isinstance(search_results, dict):
                logger.error(f"Invalid response from OpenAlex API for query: '{query}'")
                return []
            
            # Get all results
            papers = search_results.get('results', [])
            
            # Make sure papers is a list
            if not isinstance(papers, list):
                logger.error(f"Expected results to be a list but got {type(papers)} for query: '{query}'")
                papers = []

            logger.info(f"Retrieved {len(papers)} papers with abstracts for query: '{query}'")
            
            # No need to filter for abstracts anymore, as we only get papers with abstracts from the API
            
            if len(papers) == 0:
                logger.warning(f"No papers with abstracts found for query: '{query}'")
                return []
            
            # Use OpenAlex's built-in relevance sorting instead of manual sorting
            # We're removing this line: papers.sort(key=lambda x: x.get('cited_by_count', 0), reverse=True)
            
            # Select top papers for evaluation
            top_papers = papers[:top_papers_to_evaluate]
            
            logger.info(f"Selected top {len(top_papers)} papers for evaluation (sorted by OpenAlex relevance score)")
            
            # Print the titles of the papers for logging purposes
            def get_papers_titles(papers):
                for paper in papers:
                    logger.info(f"Paper title: {paper.get('title') or paper.get('display_name', 'No title')} with {paper.get('cited_by_count', 0)} citations")
            
            get_papers_titles(top_papers)
            
            evidence_list = []
            for paper in top_papers:
                try:
                    # Get the abstract using various fallback methods
                    abstract = ""
                    
                    # Try to get the abstract from abstract_inverted_index if available
                    if paper.get('abstract_inverted_index'):
                        try:
                            # Use the OpenAlexService's method to reconstruct the abstract
                            abstract = self._reconstruct_abstract_from_inverted_index(
                                paper.get('abstract_inverted_index')
                            )
                        except Exception as e:
                            logger.warning(f"Error reconstructing abstract from inverted index: {str(e)}")
                            # Fall through to next method
                    
                    # If we couldn't get an abstract from the inverted index or it's empty, try other fields
                    if not abstract or abstract.strip() == "":
                        # Try abstract field directly if it exists
                        if paper.get('abstract'):
                            abstract = paper.get('abstract')
                        # Try getting a fresh copy of the paper by ID which might include the abstract
                        elif paper.get('id'):
                            try:
                                # Extract ID from URL
                                paper_id = paper.get('id').split('/')[-1] if paper.get('id').startswith('http') else paper.get('id')
                                
                                # Log that we're fetching detailed paper info
                                logger.info(f"Fetching detailed info for paper ID: {paper_id}")
                                
                                # Request the paper details with a timeout to avoid hanging
                                paper_details = self.openalex.get_work_by_id(paper_id)
                                
                                if paper_details:
                                    # Check if we got abstract information
                                    if paper_details.get('abstract_inverted_index') or paper_details.get('abstract'):
                                        logger.info(f"Successfully retrieved abstract for paper ID: {paper_id}")
                                        
                                        # Use the detailed paper info instead
                                        if paper_details.get('abstract_inverted_index'):
                                            try:
                                                abstract = self._reconstruct_abstract_from_inverted_index(
                                                    paper_details.get('abstract_inverted_index')
                                                )
                                                logger.info(f"Successfully reconstructed abstract from inverted index for paper ID: {paper_id}")
                                            except Exception as e:
                                                logger.warning(f"Error reconstructing abstract from inverted index for paper ID {paper_id}: {str(e)}")
                                                # Try the direct abstract next
                                                if paper_details.get('abstract'):
                                                    abstract = paper_details.get('abstract')
                                        elif paper_details.get('abstract'):
                                            abstract = paper_details.get('abstract')
                                    else:
                                        logger.warning(f"No abstract found in detailed paper info for ID: {paper_id}")
                            except Exception as e:
                                logger.warning(f"Error retrieving detailed paper info for ID {paper.get('id')}: {str(e)}")
                                # Continue with what we have
                    
                    # If still no abstract, provide a fallback message
                    if not abstract or abstract.strip() == "":
                        abstract = "Abstract is not available in a usable format"
                    
                    # Prepare a concise abstract - truncate if too long
                    if len(abstract) > 500:  # Limit abstract length
                        abstract = abstract[:500] + "... [truncated]"
                    
                    # Get title with fallback
                    title = paper.get('title') or paper.get('display_name', 'Untitled')
                    
                    # Get source with fallback for different API response formats
                    source = 'Unknown'
                    if paper.get('primary_location') and paper.get('primary_location', {}).get('source'):
                        source = paper.get('primary_location', {}).get('source', {}).get('display_name', 'Unknown')
                    elif paper.get('host_venue'):
                        source = paper.get('host_venue', {}).get('display_name', 'Unknown')
                    
                    # Get all authors, not just the first 3
                    evidence = {
                        "id": paper.get('id'),
                        "title": title,
                        "abstract": abstract,
                        "year": paper.get('publication_year'),
                        "publication_date": paper.get('publication_date'),
                        "source": source,
                        "doi": paper.get('doi'),
                        "citation_count": paper.get('cited_by_count'),
                        "author_count": len(paper.get('authorships', [])),
                        "authors": [author.get('author', {}).get('display_name') 
                                    for author in paper.get('authorships', [])],
                        "url": paper.get('doi_url') or paper.get('url') or f"https://doi.org/{paper.get('doi')}" if paper.get('doi') else None
                    }
                    evidence_list.append(evidence)
                except Exception as e:
                    logger.error(f"Error processing paper: {str(e)}")
                    # Continue with the next paper
            
            logger.info(f"Processed {len(evidence_list)} evidence items for evaluation")
            return evidence_list
        except Exception as e:
            logger.error(f"Error gathering evidence: {str(e)}")
            return []
    
    def _calculate_overall_confidence(self, verification_results):
        """Calculate overall confidence based on sub-claim evaluations"""
        if not verification_results:
            return 0.0
        
        # Get confidence scores for all sub-claims
        scores = []
        for result in verification_results:
            evaluation = result.get('evaluation', {})
            
            # Adjust confidence score based on stance
            stance = evaluation.get('stance', 'insufficient')
            confidence = evaluation.get('confidence_score', 0.0)
            
            if stance == 'supports':
                scores.append(confidence)
            elif stance == 'refutes':
                scores.append(-confidence)  # Negative for refutation
            else:  # insufficient
                scores.append(0.0)
        
        # Calculate weighted average
        if not scores:
            return 0.0
            
        magnitude = sum(abs(score) for score in scores) / len(scores)
        direction = 1 if sum(scores) >= 0 else -1
        
        return direction * magnitude
    
    def _determine_verdict(self, confidence):
        """Determine the verdict based on the confidence score"""
        threshold = app.config.get('CONFIDENCE_THRESHOLD', 0.7)
        
        if confidence >= threshold:
            return "Verified"
        elif confidence <= -threshold:
            return "Refuted"
        else:
            return "Inconclusive"

    def evaluate_author_credibility(self, author_id):
        """Evaluate author credibility based on publications and citations"""
        try:
            # Get author's works
            works = self.openalex.get_author_works(author_id, per_page=100)
            
            if not works.get('results'):
                return {
                    "author_id": author_id,
                    "credibility_score": 0,
                    "publication_count": 0,
                    "total_citations": 0,
                    "analysis": "No publications found for this author"
                }
            
            # Calculate metrics
            publication_count = works.get('meta', {}).get('count', 0)
            total_citations = sum(work.get('cited_by_count', 0) for work in works.get('results', []))
            avg_citations = total_citations / publication_count if publication_count > 0 else 0
            
            # Get h-index (simplified calculation)
            citation_counts = sorted([work.get('cited_by_count', 0) for work in works.get('results', [])], reverse=True)
            h_index = 0
            for i, citations in enumerate(citation_counts):
                if citations >= i + 1:
                    h_index = i + 1
                else:
                    break
            
            # Calculate a simplified credibility score
            credibility_score = min(1.0, (h_index / 20) * 0.7 + (avg_citations / 50) * 0.3)
            
            return {
                "author_id": author_id,
                "credibility_score": credibility_score,
                "publication_count": publication_count,
                "total_citations": total_citations,
                "h_index": h_index,
                "avg_citations_per_paper": avg_citations,
                "analysis": f"Author has {publication_count} publications with {total_citations} total citations and an h-index of {h_index}"
            }
        except Exception as e:
            print(f"Error evaluating author credibility: {str(e)}")
            return {
                "author_id": author_id,
                "credibility_score": 0,
                "error": str(e)
            }


# Initialize services
openalex_service = OpenAlexService()
gemini_service = GeminiService(gemini_model)
verification_service = VerificationService(openalex_service, gemini_service)

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "Factify API",
        "version": "1.0.0"
    })

# Claim verification endpoints
@app.route('/api/verification/claim', methods=['POST'])
def verify_claim():
    """Verify a claim"""
    data = request.get_json()
    
    if not data or 'claim' not in data:
        return jsonify({"error": "Missing 'claim' in request body"}), 400
    
    claim = data['claim']
    
    try:
        start_time = time.time()
        result = verification_service.verify_claim(claim)
        processing_time = time.time() - start_time
        
        response = {
            "status": "success",
            "processing_time_seconds": processing_time,
            "result": result
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/verification/author/<author_id>/credibility', methods=['GET'])
def evaluate_author_credibility(author_id):
    """Evaluate the credibility of an author"""
    try:
        result = verification_service.evaluate_author_credibility(author_id)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Evidence retrieval endpoints
@app.route('/api/evidence/search', methods=['GET'])
def search_evidence():
    """Search for evidence"""
    query = request.args.get('query')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        results = openalex_service.search_works(query, page=page, per_page=per_page)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/evidence/work/<work_id>', methods=['GET'])
def get_work(work_id):
    """Get details for a specific work"""
    try:
        work = openalex_service.get_work_by_id(work_id)
        return jsonify({"status": "success", "work": work})
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/evidence/work/<work_id>/citations', methods=['GET'])
def get_citation_timeline(work_id):
    """Get citation timeline for a work"""
    try:
        timeline = openalex_service.get_citation_timeline(work_id)
        return jsonify({"status": "success", "timeline": timeline})
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    app.run(host="0.0.0.0", port=port, debug=app.config.get("DEBUG", False))
