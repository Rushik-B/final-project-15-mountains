import React, { useState, useEffect, useRef } from 'react';
import ResearchPaperCard from './ResearchPaperCard';

const PaperDisplay = ({ claim }) => {
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showSimplified, setShowSimplified] = useState(false);
  const paperRefs = useRef({});

  useEffect(() => {
    // Function to fetch verification results from API
    const fetchVerificationResults = async () => {
      setLoading(true);
      try {
        const response = await fetch('http://localhost:8080/api/verify_claim', {
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

  // Toggle between detailed and simplified reasoning
  const toggleSimplified = () => {
    setShowSimplified(!showSimplified);
  };

  // Scroll to paper reference when number is clicked
  const scrollToPaper = (index) => {
    const paperRef = paperRefs.current[index];
    if (paperRef) {
      paperRef.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Parse evidence numbers from detailed reasoning and make them clickable
  const parseDetailedReasoning = (text) => {
    if (!text) return null;
    
    // Regular expression to find the custom evidence chunk references
    const regex = /\[EVIDENCE_CHUNK:(\d+(?:\s*,\s*\d+)*)\]/g;
    const parts = [];
    let lastIndex = 0;
    let match;
    
    // Process each match
    while ((match = regex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        parts.push(text.substring(lastIndex, match.index));
      }
      
      // Extract the numbers string and clean it up
      const numbersStr = match[1].replace(/\s+/g, '');
      // Split into individual numbers and filter out any empty strings
      const numbers = numbersStr.split(',').filter(n => n.length > 0);
      
      // Create a wrapper span for the entire evidence group
      parts.push(
        <span key={`evidence-group-${match.index}`} className="evidence-group">
          [
          {numbers.map((numStr, i) => {
            const num = parseInt(numStr.trim());
            if (!isNaN(num)) {
              return (
                <React.Fragment key={`evidence-${num}-${i}`}>
                  <span
                    className="evidence-link"
                    onClick={() => scrollToPaper(num - 1)}
                    title={`Jump to Evidence #${num}`}
                  >
                    {num}
                  </span>
                  {i < numbers.length - 1 ? ', ' : ''}
                </React.Fragment>
              );
            }
            return null;
          })}
          ]
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }
    
    // Add any remaining text after the last match
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts;
  };

  // Get verdict icon based on result
  const getVerdictIcon = (verdict) => {
    switch(verdict.toLowerCase()) {
      case 'supported':
        return '✅';
      case 'refuted':
        return '❌';
      case 'inconclusive':
        return '⚠️';
      default:
        return '❓';
    }
  };

  // Get verdict color class
  const getVerdictColorClass = (verdict) => {
    switch(verdict.toLowerCase()) {
      case 'supported':
        return 'supported';
      case 'refuted':
        return 'refuted';
      case 'inconclusive':
      default:
        return 'inconclusive';
    }
  };

  // Handle loading and error states
  if (loading) return (
    <div className="loading-container">
      <div className="loading-spinner"></div>
      <p>Verifying claim and gathering research evidence...</p>
    </div>
  );
  
  if (error) return <div className="error-message">{error}</div>;
  if (!verificationResult) return <div className="no-results">No verification results for "{claim}"</div>;

  // Format evidence details for display
  const evidenceDetails = verificationResult.evidence || [];
  const verdictClass = getVerdictColorClass(verificationResult.verdict);
  
  // Get the appropriate reasoning based on simplified state
  const detailedReasoning = verificationResult.detailed_reasoning || verificationResult.reasoning;
  const simplifiedReasoning = verificationResult.simplified_reasoning || verificationResult.reasoning;
  const reasoningToShow = showSimplified ? simplifiedReasoning : detailedReasoning;
  
  return (
    <div className="verification-results-section">
      <h2 className="results-title">Verification Results</h2>
      
      {/* LLM Summary Card - Enhanced Modern Design */}
      <div className="llm-summary-card">
        <div className="claim-text">
          <span className="quote-mark">"</span>
          <p>{claim}</p>
          <span className="quote-mark">"</span>
        </div>
        
        <div className={`verdict-badge ${verdictClass}`}>
          <span className="verdict-icon">{getVerdictIcon(verificationResult.verdict)}</span>
          <span className="verdict-text">{verificationResult.verdict}</span>
        </div>
        
        <div className="confidence-meter">
          <div className="confidence-label">Confidence</div>
          <div className="confidence-bar-container">
            <div 
              className={`confidence-bar ${verdictClass}`} 
              style={{width: `${(verificationResult.confidence || 0) * 100}%`}}
            ></div>
          </div>
          <div className="confidence-value">{((verificationResult.confidence || 0) * 100).toFixed(0)}%</div>
        </div>
        
        {/* LLM Reasoning/Summary with Simplify Button */}
        {reasoningToShow && (
          <div className="reasoning-container">
            <div className="reasoning-header">
              <h3 className="reasoning-title">
                {showSimplified ? "Simplified Summary" : "Technical Analysis"} 
                <span className="paper-count-badge">
                  <span className="count-value">{evidenceDetails.length}</span>
                  <span className="count-label">papers analyzed</span>
                </span>
              </h3>
            </div>
            
            <div className="reasoning-content">
              <button 
                className={`simplify-button ${showSimplified ? 'active' : ''}`}
                onClick={toggleSimplified}
                aria-label={showSimplified ? "Show Technical Analysis" : "Simplify Summary"}
                title={showSimplified ? "Show Technical Analysis" : "Simplify Summary"}
              >
                {/* SVG icon for simplify/tech toggle */}
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  {showSimplified ? (
                    // Technical icon (graph-like)
                    <>
                      <line x1="3" y1="12" x2="21" y2="12"></line>
                      <line x1="3" y1="6" x2="21" y2="6"></line>
                      <line x1="3" y1="18" x2="21" y2="18"></line>
                    </>
                  ) : (
                    // Simplify icon (magic wand-like)
                    <>
                      <path d="M18 2l3 3-3 3-3-3 3-3z"></path>
                      <path d="M16 16l-9-9"></path>
                      <path d="M11.1 3.6a30 30 0 0 0-6.023 12.521"></path>
                    </>
                  )}
                </svg>
                <span>{showSimplified ? "Technical" : "Simplify"}</span>
              </button>
              
              <div className="reasoning-text">
                {showSimplified 
                  ? simplifiedReasoning
                  : parseDetailedReasoning(detailedReasoning)}
              </div>
            </div>
          </div>
        )}
        
        {/* Keywords Chips */}
        {verificationResult.keywords_used && verificationResult.keywords_used.length > 0 && (
          <div className="keywords-container">
            <h4 className="keywords-title">Keywords</h4>
            <div className="keywords-chips">
              {verificationResult.keywords_used.map((keyword, index) => (
                <span key={`keyword-${index}`} className="keyword-chip">{keyword}</span>
              ))}
            </div>
          </div>
        )}
        
        {/* Research Category */}
        <div className="category-tag">
          <span className="category-icon">🔬</span>
          <span className="category-text">{verificationResult.category || "Uncategorized"}</span>
        </div>
      </div>
      
      {/* Remove the duplicate Research Analysis Summary section */}
      {/* Instead, add the papers directly */}
      <div className="top-relevant-papers-section" id="evidence-papers">
        <h3>Evidence From Research Papers</h3>
        
        {evidenceDetails.length > 0 ? (
          <div className="papers-container">
            {evidenceDetails.map((paper, paperIndex) => (
              <div 
                key={`paper-${paperIndex}`} 
                className="paper-card-wrapper"
                ref={el => paperRefs.current[paperIndex + 1] = el}
                id={`paper-${paperIndex + 1}`}
              >
                <ResearchPaperCard 
                  title={paper.title || "Untitled Research"}
                  author={""}
                  date={paper.pub_date || ""}
                  abstract={paper.abstract || "No abstract available"}
                  categories={[]}
                  publisher={""}
                  badgeText={`Evidence #${paperIndex + 1}`}
                  doi={paper.doi || ""}
                  published={paper.pub_date || "Date not available"}
                  source={paper.source_api || "Unknown source"}
                  url={paper.link || (paper.doi ? `https://doi.org/${paper.doi}` : "")}
                  citation_count={paper.citation_count || 0}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="no-evidence">No research papers found to support this claim.</div>
        )}
      </div>
      
      {/* Processing Time */}
      <div className="processing-info">
        <p>Processing time: {verificationResult.processing_time_seconds.toFixed(2)} seconds</p>
      </div>
      
      <style jsx>{`
        /* Styling for the simplify button */
        .reasoning-content {
          position: relative;
        }
        
        .simplify-button {
          position: absolute;
          top: -40px;
          right: 0;
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 0.85rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          background-color: #f0f5ff;
          border: 1px solid #d1e1ff;
          color: #2563eb;
          z-index: 2;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .simplify-button.active {
          background-color: #2563eb;
          color: white;
          border-color: #2563eb;
        }
        
        .simplify-button:hover {
          background-color: ${showSimplified ? '#1d4ed8' : '#e0eaff'};
          border-color: ${showSimplified ? '#1d4ed8' : '#93b4ff'};
          transform: translateY(-1px);
          box-shadow: 0 3px 5px rgba(0, 0, 0, 0.15);
        }
        
        .simplify-button svg {
          transition: all 0.2s ease;
        }
        
        .reasoning-title {
          margin-bottom: 0;
        }
        
        .reasoning-text {
          margin-top: 16px;
        }
      `}</style>
    </div>
  );
};

export default PaperDisplay;
