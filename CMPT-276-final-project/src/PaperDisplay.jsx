import React, { useState, useEffect } from 'react';
import ResearchPaperCard from './ResearchPaperCard';

const PaperDisplay = ({ claim }) => {
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Function to fetch verification results from API
    const fetchVerificationResults = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8080/api/verification/claim', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            claim: claim
          }),
        });
        
        if (!response.ok) {
          throw new Error(`API request failed with status ${response.status}`);
        }
        
        const data = await response.json();
        setVerificationResult(data.result);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching verification results:", err);
        setError('Failed to verify claim. Please try again.');
        setLoading(false);
      }
    };

    if (claim) {
      fetchVerificationResults();
    }
  }, [claim]); // Re-fetch when claim changes

  // Calculate total papers analyzed across all sub-claims
  const calculateTotalPapers = (result) => {
    if (!result || !result.sub_claims) return 0;
    return result.sub_claims.reduce((total, subClaim) => 
      total + (subClaim.evidence_count || 0), 0);
  };

  // Extract all papers from all sub-claims and get the top 30 most relevant
  const getTopRelevantPapers = (result) => {
    if (!result || !result.sub_claims) return [];
    
    // Collect all papers from all sub-claims
    const allPapers = [];
    result.sub_claims.forEach(subClaim => {
      if (subClaim.evidence && subClaim.evidence.length > 0) {
        allPapers.push(...subClaim.evidence);
      }
    });
    
    // Remove duplicates based on paper ID
    const uniquePapers = [];
    const paperIds = new Set();
    
    allPapers.forEach(paper => {
      if (paper.id && !paperIds.has(paper.id)) {
        paperIds.add(paper.id);
        uniquePapers.push(paper);
      } else if (!paper.id) {
        // If no ID is available, include the paper anyway
        uniquePapers.push(paper);
      }
    });
    
    // Return top 30 (OpenAlex already sorts by relevance)
    return uniquePapers.slice(0, 30);
  };

  // Gather key findings from all sub-claims
  const getOverallFindings = (result) => {
    if (!result || !result.sub_claims) return { supports: [], refutes: [], gaps: [] };
    
    const findings = {
      supports: [],
      refutes: [],
      gaps: []
    };
    
    result.sub_claims.forEach(subClaim => {
      const evaluation = subClaim.evaluation || {};
      
      // Get key support points
      if (evaluation.key_support_points && evaluation.key_support_points.length > 0) {
        findings.supports = [...findings.supports, ...evaluation.key_support_points.slice(0, 2)];
      }
      
      // Get key refutation points
      if (evaluation.key_refutation_points && evaluation.key_refutation_points.length > 0) {
        findings.refutes = [...findings.refutes, ...evaluation.key_refutation_points.slice(0, 2)];
      }
      
      // Get evidence gaps
      if (evaluation.evidence_gaps && evaluation.evidence_gaps.length > 0) {
        findings.gaps = [...findings.gaps, ...evaluation.evidence_gaps.slice(0, 2)];
      }
    });
    
    // Remove duplicates
    findings.supports = [...new Set(findings.supports)];
    findings.refutes = [...new Set(findings.refutes)];
    findings.gaps = [...new Set(findings.gaps)];
    
    return findings;
  };

  // Handle loading and error states
  if (loading) return <div className="loading-indicator">Verifying claim and gathering papers...</div>;
  if (error) return <div className="error-message">{error}</div>;
  if (!verificationResult) return <div className="no-results">No verification results for "{claim}"</div>;

  const totalPapers = calculateTotalPapers(verificationResult);
  const findings = getOverallFindings(verificationResult);
  const topRelevantPapers = getTopRelevantPapers(verificationResult);

  return (
    <div className="verification-results-section">
      <h2 className="results-title">Verification Results for: "{claim}"</h2>
      
      <div className="verdict-summary">
        <h3>Verdict: <span className={`verdict-${verificationResult.verdict.toLowerCase()}`}>{verificationResult.verdict}</span></h3>
        <p>Confidence Score: {verificationResult.overall_confidence.toFixed(2)}</p>
      </div>
      
      {/* LLM Text Summary */}
      {verificationResult.text_summary && (
        <div className="text-summary">
          <p>{verificationResult.text_summary}</p>
          <div className="text-summary-decoration"></div>
        </div>
      )}
      
      {/* Research Analysis Summary */}
      <div className="analysis-summary">
        <h3>Research Analysis Summary</h3>
        
        <div className="summary-content">
          <p className="summary-stat">
            <span className="stat-value">{verificationResult.sub_claims?.length || 0}</span> sub-claims analyzed from 
            <span className="stat-value"> {totalPapers}</span> academic papers
          </p>
          
          <div className="summary-findings">
            {findings.supports.length > 0 && (
              <div className="finding-group supports">
                <h4>Key Supporting Evidence:</h4>
                <ul>
                  {findings.supports.slice(0, 3).map((point, index) => (
                    <li key={`support-summary-${index}`}>{point}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {findings.refutes.length > 0 && (
              <div className="finding-group refutes">
                <h4>Key Refuting Evidence:</h4>
                <ul>
                  {findings.refutes.slice(0, 3).map((point, index) => (
                    <li key={`refute-summary-${index}`}>{point}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {findings.gaps.length > 0 && (
              <div className="finding-group gaps">
                <h4>Evidence Gaps:</h4>
                <ul>
                  {findings.gaps.slice(0, 3).map((point, index) => (
                    <li key={`gap-summary-${index}`}>{point}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          
          <p className="summary-conclusion">
            Based on our analysis of the academic literature, the claim {verificationResult.verdict.toLowerCase() === 'verified' ? 
              'is supported by substantial evidence' : 
              verificationResult.verdict.toLowerCase() === 'refuted' ? 
                'is contradicted by substantial evidence' : 
                'has insufficient or conflicting evidence'}.
          </p>
        </div>
      </div>
      
      {/* Top Relevant Papers Section */}
      <div className="top-relevant-papers-section">
        <h3>Top 30 Most Relevant Research Papers</h3>
        
        {topRelevantPapers.length > 0 ? (
          <div className="papers-container">
            {topRelevantPapers.map((paper, paperIndex) => (
              <div key={`paper-${paperIndex}`} className="paper-card-wrapper">
                <ResearchPaperCard 
                  title={paper.title || "Untitled Research"}
                  author={paper.authors ? paper.authors.join(', ') : "Unknown Authors"}
                  date={paper.publication_date || paper.year || "Unknown Date"}
                  abstract={paper.abstract || "No abstract provided for this research paper."}
                  categories={[paper.source || "Academic Source"]}
                  publisher={paper.source || ""}
                  badgeText={"Citations: " + (paper.citation_count || 0)}
                  doi={paper.doi || ""}
                  published={paper.publication_date || paper.year || ""}
                  source={paper.source || ""}
                  url={paper.url || (paper.doi ? `https://doi.org/${paper.doi}` : "")}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="no-evidence">No research papers found to support this claim.</div>
        )}
      </div>
      
      {/* Sub-claims Analysis without Papers (moved to top relevant papers section) */}
      {verificationResult.sub_claims && verificationResult.sub_claims.length > 0 && (
        <div className="sub-claims-section">
          <h3>Sub-claims Analysis</h3>
          
          {verificationResult.sub_claims.map((subClaim, index) => (
            <div key={`subclaim-${index}`} className="sub-claim-container">
              <div className="sub-claim-header">
                <h4>Sub-claim {index + 1}: {subClaim.sub_claim}</h4>
                
                <div className="sub-claim-evaluation">
                  <p className={`stance-${subClaim.evaluation?.stance}`}>
                    Stance: {subClaim.evaluation?.stance}
                  </p>
                  <p>Confidence: {subClaim.evaluation?.confidence_score.toFixed(2)}</p>
                </div>
              </div>
              
              {subClaim.key_entities && subClaim.key_entities.length > 0 && (
                <div className="key-entities">
                  <h5>Key Entities:</h5>
                  <ul>
                    {subClaim.key_entities.map((entity, entityIndex) => (
                      <li key={`entity-${entityIndex}`}>{entity}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {subClaim.evaluation?.key_support_points && subClaim.evaluation.key_support_points.length > 0 && (
                <div className="key-points">
                  <h5>Key Support Points:</h5>
                  <ul>
                    {subClaim.evaluation.key_support_points.map((point, pointIndex) => (
                      <li key={`support-${pointIndex}`}>{point}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {subClaim.evaluation?.key_refutation_points && subClaim.evaluation.key_refutation_points.length > 0 && (
                <div className="key-points">
                  <h5>Key Refutation Points:</h5>
                  <ul>
                    {subClaim.evaluation.key_refutation_points.map((point, pointIndex) => (
                      <li key={`refute-${pointIndex}`}>{point}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PaperDisplay;
