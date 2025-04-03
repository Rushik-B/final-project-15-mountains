import React, { useState, useEffect, useRef } from 'react';
import ResearchPaperCard from './ResearchPaperCard';
import MagneticElement from './MagneticElement';

const PaperDisplay = ({ claim }) => {
  const [verificationResult, setVerificationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showSimplified, setShowSimplified] = useState(false);
  const [loadingStage, setLoadingStage] = useState("initializing");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const paperRefs = useRef({});
  const timerRef = useRef(null);
  const summaryRef = useRef(null);
  const filterRef = useRef(null);
  const evidenceSectionRef = useRef(null);
  const [showBackToTop, setShowBackToTop] = useState(false);
  
  // Original and filtered evidence
  const [originalEvidence, setOriginalEvidence] = useState([]);
  const [filteredEvidence, setFilteredEvidence] = useState([]);
  
  // Filter state variables
  const [sourceFilter, setSourceFilter] = useState('All Sources');
  const [startYear, setStartYear] = useState(2000);
  const [endYear, setEndYear] = useState(2025);
  const [relevanceScore, setRelevanceScore] = useState(0);
  const [sortBy, setSortBy] = useState('Relevance');
  
  // Min and max years for date range
  const MIN_YEAR = 1980;
  const MAX_YEAR = 2025;
  const YEAR_RANGE = MAX_YEAR - MIN_YEAR;

  // Add scroll tracking
  useEffect(() => {
    const handleScroll = () => {
      // Only show back to top button when scrolled down to the evidence section
      if (evidenceSectionRef.current) {
        const evidencePosition = evidenceSectionRef.current.getBoundingClientRect().top;
        // Show button when evidence section is at the top of viewport or above it
        setShowBackToTop(evidencePosition <= 100);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll back to summary function
  const scrollToSummary = () => {
    if (summaryRef.current) {
      summaryRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  useEffect(() => {
    // Function to fetch verification results from API
    const fetchVerificationResults = async () => {
      setLoading(true);
      setLoadingStage("initializing");
      setLoadingProgress(5);
      
      // Simulate the various loading stages with realistic timings
      const loadingStages = [
        { stage: "extracting_keywords", progress: 15, time: 2000 },
        { stage: "searching_openalex", progress: 30, time: 3000 },
        { stage: "searching_crossref", progress: 40, time: 2000 },
        { stage: "retrieving_papers", progress: 50, time: 4000 },
        { stage: "embedding_abstracts", progress: 65, time: 3000 },
        { stage: "semantic_search", progress: 75, time: 2000 },
        { stage: "analyzing_evidence", progress: 90, time: 3000 }
      ];
      
      // Setup progressive loading stages
      let currentStageIndex = 0;
      
      const progressThroughStages = () => {
        if (currentStageIndex < loadingStages.length) {
          const currentStage = loadingStages[currentStageIndex];
          setLoadingStage(currentStage.stage);
          setLoadingProgress(currentStage.progress);
          
          timerRef.current = setTimeout(() => {
            currentStageIndex++;
            progressThroughStages();
          }, currentStage.time);
        }
      };
      
      // Start the loading stage simulation
      progressThroughStages();
      
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
        // Clear any remaining timers
        if (timerRef.current) {
          clearTimeout(timerRef.current);
        }
        
        setLoadingStage("completed");
        setLoadingProgress(100);
        
        // Small delay before showing results to ensure user sees 100%
        setTimeout(() => {
          setVerificationResult(data.result);
          
          // Process the evidence array - assign permanent evidence numbers
          if (data.result && data.result.evidence && Array.isArray(data.result.evidence)) {
            // Assign evidence numbers to each paper based on original order
            const processedEvidence = data.result.evidence.map((paper, index) => ({
              ...paper,
              // Evidence numbers are 1-indexed since that's how they're referenced in the text
              evidenceNumber: index + 1
            }));
            
            setOriginalEvidence(processedEvidence);
            setFilteredEvidence(processedEvidence); // Initial filtered evidence is the same as original
          }
          
          setLoading(false);
        }, 500);
      } catch (err) {
        console.error("Error fetching verification results:", err);
        setError('Failed to verify claim. Please try again.');
        setLoading(false);
        // Clear any remaining timers
        if (timerRef.current) {
          clearTimeout(timerRef.current);
        }
      }
    };

    if (claim) {
      fetchVerificationResults();
    }
    
    // Cleanup timers when component unmounts
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [claim]); // Re-fetch when claim changes

  // Apply filters and sorting whenever filter settings change
  useEffect(() => {
    if (!originalEvidence || originalEvidence.length === 0) return;
    
    let filtered = [...originalEvidence];
    
    // Apply source filter
    if (sourceFilter !== 'All Sources') {
      filtered = filtered.filter(paper => 
        paper.source_api && paper.source_api.toLowerCase() === sourceFilter.toLowerCase()
      );
    }
    
    // Apply date range filter
    filtered = filtered.filter(paper => {
      if (!paper.pub_date) return true;
      
      // Extract year from publication date
      const yearMatch = paper.pub_date.match(/\d{4}/);
      if (!yearMatch) return true;
      
      const pubYear = parseInt(yearMatch[0]);
      return pubYear >= startYear && pubYear <= endYear;
    });
    
    // Apply sort
    if (sortBy === 'Date (Newest)') {
      filtered.sort((a, b) => {
        if (!a.pub_date) return 1;
        if (!b.pub_date) return -1;
        return new Date(b.pub_date) - new Date(a.pub_date);
      });
    } else if (sortBy === 'Date (Oldest)') {
      filtered.sort((a, b) => {
        if (!a.pub_date) return 1;
        if (!b.pub_date) return -1;
        return new Date(a.pub_date) - new Date(b.pub_date);
      });
    } else if (sortBy === 'Citations') {
      filtered.sort((a, b) => (b.citation_count || 0) - (a.citation_count || 0));
    } else if (sortBy === 'Relevance') {
      // Sort by original order (relevance)
      filtered.sort((a, b) => a.evidenceNumber - b.evidenceNumber);
    }
    
    setFilteredEvidence(filtered);
  }, [originalEvidence, sourceFilter, startYear, endYear, relevanceScore, sortBy]);

  // Toggle between detailed and simplified reasoning
  const toggleSimplified = () => {
    setShowSimplified(!showSimplified);
  };

  // Scroll to paper reference when number is clicked
  const scrollToPaper = (evidenceNumber) => {
    const paperElement = document.querySelector(`[data-evidence-number="${evidenceNumber}"]`);
    if (paperElement) {
      paperElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Get loading stage text
  const getLoadingStageText = (stage) => {
    switch(stage) {
      case "initializing":
        return "Initializing verification process...";
      case "extracting_keywords":
        return "Extracting keywords from your claim...";
      case "searching_openalex":
        return "Searching OpenAlex scientific database...";
      case "searching_crossref":
        return "Searching CrossRef research papers...";
      case "retrieving_papers":
        return "Retrieving relevant research papers...";
      case "embedding_abstracts":
        return "Processing research abstracts...";
      case "semantic_search":
        return "Finding most relevant scientific evidence...";
      case "analyzing_evidence":
        return "Analyzing evidence from academic sources...";
      case "verifying_claim":
        return "Evaluating claim accuracy against scientific consensus...";
      case "completed":
        return "Verification complete, displaying results...";
      default:
        return "Processing your request...";
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
        // Split paragraphs for better readability
        const textBefore = text.substring(lastIndex, match.index);
        parts.push(textBefore);
      }
      
      // Extract the numbers string and clean it up
      const numbersStr = match[1].replace(/\s+/g, '');
      // Split into individual numbers and filter out any empty strings
      const numbers = numbersStr.split(',').filter(n => n.length > 0);
      
      // Limit the number of evidence links displayed if there are too many
      const displayLimit = 10;
      const hasMoreNumbers = numbers.length > displayLimit;
      const displayNumbers = hasMoreNumbers ? numbers.slice(0, displayLimit) : numbers;
      
      // Create a wrapper span for the entire evidence group
      parts.push(
        <span key={`evidence-group-${match.index}`} className="evidence-group">
          [
          {displayNumbers.map((numStr, i) => {
            const num = parseInt(numStr.trim());
            if (!isNaN(num)) {
              // Check if this evidence number exists in our filtered results
              const paperExists = filteredEvidence.some(p => p.evidenceNumber === num);
              
              return (
                <React.Fragment key={`evidence-${num}-${i}`}>
                  <span
                    className={`evidence-link ${!paperExists ? 'evidence-link-filtered' : ''}`}
                    onClick={() => scrollToPaper(num)}
                    title={`Jump to Evidence #${num}${!paperExists ? ' (filtered out)' : ''}`}
                  >
                    {num}
                  </span>
                  {i < displayNumbers.length - 1 ? ', ' : ''}
                </React.Fragment>
              );
            }
            return null;
          })}
          {hasMoreNumbers && <span className="more-evidence">+{numbers.length - displayLimit} more</span>}
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

  // Get verdict icon based on accuracy score
  const getVerdictIcon = (accuracyScore) => {
    if (accuracyScore >= 0.8) return '✅'; // Highly accurate
    if (accuracyScore >= 0.5) return '⚠️'; // Somewhat accurate
    if (accuracyScore >= 0.2) return '⚠️'; // Somewhat inaccurate
    return '❌'; // Highly inaccurate
  };

  // Get verdict text based on accuracy score
  const getVerdictText = (accuracyScore) => {
    if (accuracyScore >= 0.8) return 'Highly Supported';
    if (accuracyScore >= 0.5) return 'Moderately Supported';
    if (accuracyScore >= 0.2) return 'Weakly Supported';
    return 'Not Supported';
  };

  // Get verdict color class based on accuracy score
  const getVerdictColorClass = (accuracyScore) => {
    if (accuracyScore >= 0.8) return 'supported';
    if (accuracyScore >= 0.5) return 'partially-supported';
    if (accuracyScore >= 0.2) return 'inconclusive';
    return 'refuted';
  };

  // Function to handle date range slider change
  const handleDateRangeChange = (e, handle, values) => {
    setStartYear(Math.round(values[0]));
    setEndYear(Math.round(values[1]));
  };

  // Main render method
  if (loading) return (
    <div className="loading-container">
      <div className="loading-progress-container">
        <div 
          className="loading-progress-bar" 
          style={{width: `${loadingProgress}%`}}
        ></div>
      </div>
      <div className="loading-stage">
        <div className="loading-spinner"></div>
        <p>{getLoadingStageText(loadingStage)}</p>
        <p className="loading-percentage">{loadingProgress}% complete</p>
      </div>
      <div className="loading-info">
        <p>We're searching through scientific literature to verify your claim.</p>
        <p className="loading-detail">This process analyzes multiple research papers and typically takes around 10-15 seconds depending on complexity.</p>
        {loadingStage === "analyzing_evidence" && 
          <p className="loading-almost-done">Almost done! Our AI is reviewing the evidence and calculating the claim's accuracy score...</p>
        }
      </div>
    </div>
  );
  
  if (error) return <div className="error-message">{error}</div>;
  if (!verificationResult) return <div className="no-results">No verification results for "{claim}"</div>;

  // Format evidence details for display
  const evidenceDetails = verificationResult.evidence || [];
  // Use accuracy score from backend (fallback to confidence for backward compatibility)
  const accuracyScore = verificationResult.accuracy_score || verificationResult.confidence || 0;
  const verdictClass = getVerdictColorClass(accuracyScore);
  
  // Get the appropriate reasoning based on simplified state
  const detailedReasoning = verificationResult.detailed_reasoning || verificationResult.reasoning;
  const simplifiedReasoning = verificationResult.simplified_reasoning || verificationResult.reasoning;
  const reasoningToShow = showSimplified ? simplifiedReasoning : detailedReasoning;
  
  return (
    <div className="verification-results-section">
      <h2 className="results-title" ref={summaryRef}>Verification Results</h2>
      
      {/* LLM Summary Card - Enhanced Modern Design */}
      <div className="llm-summary-card">
        <div className="claim-text">
          <span className="quote-mark">"</span>
          <p>{claim}</p>
          <span className="quote-mark">"</span>
        </div>
        
        <div className="verdict-container">
          <div className={`verdict-badge ${verdictClass}`}>
            <span className="verdict-icon">{getVerdictIcon(accuracyScore)}</span>
            <span className="verdict-text">{getVerdictText(accuracyScore)}</span>
          </div>
          
          <div className="accuracy-display">
            <div className="accuracy-label">Accuracy Score</div>
            <div className="accuracy-bar-container">
              <div 
                className={`accuracy-bar ${verdictClass}`} 
                style={{width: `${(accuracyScore || 0) * 100}%`}}
              >
                <span className="accuracy-value">{(accuracyScore * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* LLM Reasoning/Summary with Simplify Button */}
        {reasoningToShow && (
          <div className="reasoning-container">
            <div className="reasoning-header">
              <h3 className="reasoning-title">
                {showSimplified ? "Simplified Summary" : "Technical Analysis"} 
                <span className="paper-count-badge">
                  <span className="count-value">{filteredEvidence.length}</span>
                  <span className="count-label">papers analyzed</span>
                </span>
              </h3>
            </div>
            
            <div className="reasoning-content">
              <div className="simplify-button-container">
                <MagneticElement strength={35} distance={80}>
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
                </MagneticElement>
              </div>
              
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
      
      {/* Filtering Box */}
      <div className="evidence-filter-box" ref={filterRef}>
        <div className="filter-header">
          <h3>Filter Evidence</h3>
          <span className="filter-results-count">{filteredEvidence.length} papers found</span>
        </div>
        
        <div className="filter-controls">
          <div className="filter-section">
            <div className="filter-group">
              <label>Source</label>
              <select 
                value={sourceFilter} 
                onChange={(e) => setSourceFilter(e.target.value)}
                className="filter-select"
              >
                <option value="All Sources">All Sources</option>
                <option value="crossref">CrossRef</option>
                <option value="openalex">OpenAlex</option>
                <option value="semantic_scholar">Semantic Scholar</option>
              </select>
            </div>
          </div>
          
          <div className="filter-section">
            <div className="filter-group">
              <label>Sort By</label>
              <select 
                value={sortBy} 
                onChange={(e) => setSortBy(e.target.value)}
                className="filter-select"
              >
                <option value="Relevance">Relevance</option>
                <option value="Date (Newest)">Date (Newest)</option>
                <option value="Date (Oldest)">Date (Oldest)</option>
                <option value="Citations">Citations</option>
              </select>
            </div>
          </div>
          
          <div className="filter-section year-range-section">
            <div className="filter-group">
              <div className="date-range-header">
                <label>Publication Year Range</label>
                <span className="date-range-value">{startYear} - {endYear}</span>
              </div>
              <div className="date-slider-container">
                <input 
                  type="range" 
                  min={MIN_YEAR} 
                  max={MAX_YEAR} 
                  value={startYear}
                  onChange={(e) => {
                    const newStart = parseInt(e.target.value);
                    if (newStart <= endYear) {
                      setStartYear(newStart);
                    }
                  }}
                  className="date-slider date-slider-start"
                />
                <input 
                  type="range" 
                  min={MIN_YEAR} 
                  max={MAX_YEAR} 
                  value={endYear}
                  onChange={(e) => {
                    const newEnd = parseInt(e.target.value);
                    if (newEnd >= startYear) {
                      setEndYear(newEnd);
                    }
                  }}
                  className="date-slider date-slider-end"
                />
                <div className="slider-track"></div>
                <div 
                  className="slider-range" 
                  style={{
                    left: `${((startYear - MIN_YEAR) / YEAR_RANGE) * 100}%`,
                    width: `${((endYear - startYear) / YEAR_RANGE) * 100}%`
                  }}
                ></div>
                <div className="year-markers">
                  <span className="year-marker">{MIN_YEAR}</span>
                  <span className="year-marker">{MIN_YEAR + Math.round(YEAR_RANGE/4)}</span>
                  <span className="year-marker">{MIN_YEAR + Math.round(YEAR_RANGE/2)}</span>
                  <span className="year-marker">{MIN_YEAR + Math.round(3*YEAR_RANGE/4)}</span>
                  <span className="year-marker">{MAX_YEAR}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Evidence Papers Section */}
      <div className="top-relevant-papers-section" id="evidence-papers" ref={evidenceSectionRef}>
        <h3>Evidence From Research Papers</h3>
        
        {filteredEvidence.length > 0 ? (
          <div className="papers-container">
            {filteredEvidence.map((paper, displayIndex) => (
              <div 
                key={`paper-${paper.evidenceNumber}`} 
                className="paper-card-wrapper"
                ref={el => paperRefs.current[paper.evidenceNumber] = el}
                id={`paper-${paper.evidenceNumber}`}
                data-evidence-number={paper.evidenceNumber}
              >
                <ResearchPaperCard 
                  title={paper.title || "Untitled Research"}
                  author={""}
                  date={paper.pub_date || ""}
                  abstract={paper.abstract || "No abstract available"}
                  categories={[]}
                  publisher={""}
                  badgeText={`Evidence #${paper.evidenceNumber}`}
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
          <div className="no-evidence">
            {originalEvidence && originalEvidence.length > 0 ? 
              'No papers match your current filter settings.' : 
              'No research papers found to support this claim.'
            }
          </div>
        )}
      </div>
      
      {/* Back to Top Button - Only visible when scrolled to evidence section */}
      {showBackToTop && (
        <div className="back-to-top-wrapper">
          <button 
            className="back-to-top-button" 
            onClick={scrollToSummary}
            aria-label="Back to Summary"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 15l-6-6-6 6"/>
            </svg>
            <span>Back to Summary</span>
          </button>
        </div>
      )}
      
      {/* Processing Time */}
      <div className="processing-info">
        <p>Processing time: {verificationResult.processing_time_seconds?.toFixed(2) || 0} seconds</p>
      </div>
      
      <style jsx>{`
        /* Styling for the simplify button */
        .reasoning-content {
          position: relative;
        }
        
        .simplify-button-container {
          position: absolute;
          top: -40px;
          right: 0;
          z-index: 2;
        }
        
        .simplify-button {
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
        
        /* Evidence Filter Box Styling - Modern Tailwind-like */
        .evidence-filter-box {
          background-color: rgba(18, 24, 40, 0.8);
          border-radius: 1rem;
          padding: 1.5rem;
          margin: 1.5rem 0;
          box-shadow: 0 8px 20px -5px rgba(0, 0, 0, 0.15), 0 0 15px rgba(59, 130, 246, 0.2), 0 0 3px rgba(99, 102, 241, 0.3);
          border: 1px solid rgba(59, 130, 246, 0.2);
          backdrop-filter: blur(12px);
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
          max-width: 90%;
          margin-left: auto;
          margin-right: auto;
          color: #e5e7eb;
        }
        
        .evidence-filter-box::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 0.25rem;
          background: linear-gradient(90deg, #2563eb, #7c3aed);
          opacity: 0.7;
        }
        
        .evidence-filter-box:hover {
          transform: translateY(-2px);
          box-shadow: 0 14px 28px -5px rgba(0, 0, 0, 0.2), 0 0 20px rgba(59, 130, 246, 0.3), 0 0 8px rgba(99, 102, 241, 0.4);
          border-color: rgba(59, 130, 246, 0.3);
        }
        
        .filter-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.25rem;
          padding-bottom: 0.75rem;
          border-bottom: 1px solid rgba(75, 85, 99, 0.5);
        }
        
        .evidence-filter-box h3 {
          margin: 0;
          font-size: 1.25rem;
          color: #e5e7eb;
          font-weight: 600;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        
        .evidence-filter-box h3::before {
          content: '🔍';
          font-size: 1.125rem;
        }
        
        .filter-results-count {
          font-size: 0.875rem;
          color: #d1d5db;
          background-color: rgba(31, 41, 55, 0.7);
          padding: 0.375rem 0.875rem;
          border-radius: 9999px;
          font-weight: 500;
          display: flex;
          align-items: center;
          border: 1px solid rgba(75, 85, 99, 0.5);
          backdrop-filter: blur(4px);
        }
        
        .filter-results-count::before {
          content: '';
          display: inline-block;
          width: 0.5rem;
          height: 0.5rem;
          background-color: ${filteredEvidence.length > 0 ? '#10b981' : '#ef4444'};
          border-radius: 50%;
          margin-right: 0.5rem;
          box-shadow: 0 0 5px ${filteredEvidence.length > 0 ? 'rgba(16, 185, 129, 0.5)' : 'rgba(239, 68, 68, 0.5)'};
        }
        
        .filter-controls {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
          width: 100%;
        }
        
        .filter-section {
          width: 100%;
        }
        
        .year-range-section {
          margin-top: 0.5rem;
        }
        
        .filter-divider {
          display: none;
        }
        
        .filter-group {
          display: flex;
          flex-direction: column;
          width: 100%;
        }
        
        .filter-group label {
          font-size: 0.875rem;
          font-weight: 500;
          margin-bottom: 0.5rem;
          color: #d1d5db;
          display: inline-flex;
          align-items: center;
        }
        
        .filter-select {
          padding: 0.625rem 1rem;
          border-radius: 0.5rem;
          border: 1px solid rgba(75, 85, 99, 0.6);
          background-color: rgba(31, 41, 55, 0.6);
          color: #e5e7eb;
          font-size: 0.95rem;
          transition: all 0.15s ease;
          width: 100%;
          appearance: none;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%23d1d5db' stroke-width='2'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' d='M19 9l-7 7-7-7' /%3E%3C/svg%3E");
          background-position: right 0.75rem center;
          background-repeat: no-repeat;
          background-size: 1rem;
          padding-right: 2.5rem;
          cursor: pointer;
          backdrop-filter: blur(4px);
        }
        
        .filter-select:focus {
          outline: none;
          border-color: rgba(59, 130, 246, 0.8);
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25);
          background-color: rgba(31, 41, 55, 0.8);
        }
        
        .filter-select:hover {
          border-color: rgba(96, 165, 250, 0.6);
          background-color: rgba(31, 41, 55, 0.8);
        }
        
        .date-range-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.75rem;
        }
        
        .date-range-value {
          font-size: 0.875rem;
          font-weight: 600;
          color: #60a5fa;
          background-color: rgba(30, 58, 138, 0.5);
          padding: 0.25rem 0.75rem;
          border-radius: 0.5rem;
          display: inline-flex;
          border: 1px solid rgba(59, 130, 246, 0.3);
          backdrop-filter: blur(4px);
        }
        
        .date-slider-container {
          position: relative;
          width: 100%;
          height: 40px;
          margin: 0.5rem 0 1.5rem;
        }
        
        .date-slider {
          -webkit-appearance: none;
          appearance: none;
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 8px;
          background: transparent;
          z-index: 2;
          pointer-events: none;
        }
        
        .date-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 1.125rem;
          height: 1.125rem;
          background: linear-gradient(135deg, #3b82f6, #6366f1);
          border-radius: 50%;
          cursor: pointer;
          pointer-events: auto;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
          z-index: 3;
          transition: all 0.2s ease;
          border: 2px solid rgba(30, 58, 138, 0.8);
        }
        
        .date-slider::-webkit-slider-thumb:hover, 
        .date-slider:active::-webkit-slider-thumb {
          transform: scale(1.25);
          box-shadow: 0 0 0 5px rgba(59, 130, 246, 0.3);
        }
        
        .date-slider::-moz-range-thumb {
          width: 1.125rem;
          height: 1.125rem;
          background: linear-gradient(135deg, #3b82f6, #6366f1);
          border-radius: 50%;
          cursor: pointer;
          pointer-events: auto;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
          z-index: 3;
          border: 2px solid rgba(30, 58, 138, 0.8);
          transition: all 0.2s ease;
        }
        
        .date-slider::-moz-range-thumb:hover,
        .date-slider:active::-moz-range-thumb {
          transform: scale(1.25);
          box-shadow: 0 0 0 5px rgba(59, 130, 246, 0.3);
        }
        
        .slider-track {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 0.5rem;
          background: rgba(55, 65, 81, 0.6);
          border-radius: 9999px;
          z-index: 1;
        }
        
        .slider-range {
          position: absolute;
          top: 0;
          height: 0.5rem;
          background: linear-gradient(90deg, rgba(59, 130, 246, 0.8), rgba(99, 102, 241, 0.8));
          border-radius: 9999px;
          z-index: 1;
          box-shadow: 0 0 6px rgba(59, 130, 246, 0.5);
        }
        
        .year-markers {
          position: absolute;
          top: 1.25rem;
          left: 0;
          width: 100%;
          display: flex;
          justify-content: space-between;
          margin-top: 0.375rem;
        }
        
        .year-marker {
          font-size: 0.75rem;
          color: #9ca3af;
          position: relative;
        }
        
        .year-marker::before {
          content: '';
          position: absolute;
          top: -0.75rem;
          left: 50%;
          transform: translateX(-50%);
          width: 1px;
          height: 0.375rem;
          background-color: rgba(156, 163, 175, 0.5);
        }
        
        /* Style for filtered evidence links */
        :global(.evidence-link-filtered) {
          color: #9ca3af !important;
          text-decoration: line-through;
          background-color: rgba(156, 163, 175, 0.1);
          cursor: not-allowed;
        }
        
        :global(.evidence-link-filtered:hover) {
          color: #6b7280 !important;
          background-color: rgba(156, 163, 175, 0.2);
          transform: none;
          box-shadow: none;
        }
        
        @media (min-width: 768px) {
          .evidence-filter-box {
            max-width: 700px;
          }
        }
        
        /* Verdict and Accuracy display */
        .verdict-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin: 1.5rem 0;
          padding: 1rem;
          background-color: rgba(249, 250, 251, 0.05);
          border-radius: 1rem;
          border: 1px solid rgba(229, 231, 235, 0.1);
          backdrop-filter: blur(8px);
        }
        
        .verdict-badge {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem 1.5rem;
          border-radius: 0.75rem;
          margin-bottom: 1.25rem;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          width: auto;
          transition: all 0.3s ease;
        }
        
        .verdict-badge:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        .verdict-icon {
          font-size: 1.5rem;
        }
        
        .verdict-text {
          font-size: 1.125rem;
          font-weight: 600;
        }
        
        .accuracy-display {
          width: 100%;
          max-width: 400px;
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        
        .accuracy-label {
          font-size: 0.875rem;
          font-weight: 500;
          color: #9ca3af;
          margin-bottom: 0.5rem;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .accuracy-bar-container {
          height: 1.75rem;
          background-color: rgba(55, 65, 81, 0.5);
          border-radius: 9999px;
          overflow: hidden;
          width: 100%;
          position: relative;
          margin-bottom: 0.5rem;
          box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
        }
        
        .accuracy-bar {
          height: 100%;
          border-radius: 9999px;
          transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
          display: flex;
          align-items: center;
          justify-content: flex-end;
        }

        .accuracy-bar::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(
            90deg,
            transparent,
            rgba(255, 255, 255, 0.55),
            transparent
          );
          transform: translateX(-100%);
          animation: shimmer 2.5s infinite linear;
          z-index: 1;
          border-radius: 9999px;
        }

        @keyframes shimmer {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }

        .accuracy-value {
          font-size: 0.85rem;
          font-weight: 700;
          color: #ffffff;
          position: absolute;
          right: 10px;
          top: 50%;
          transform: translateY(-50%);
          text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
          padding: 0 8px;
          z-index: 2;
        }
        
        .accuracy-bar.supported {
          background: #10b981;
          box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        .accuracy-bar.partially-supported {
          background: #f59e0b;
          box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        
        .accuracy-bar.inconclusive {
          background: #f59e0b;
          box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        
        .accuracy-bar.refuted {
          background: #ef4444;
          box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        
        :global(@keyframes shimmer) {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(100%);
          }
        }
        
        /* Verdict color classes */
        .supported {
          background-color: rgba(16, 185, 129, 0.2);
          border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .accuracy-bar.supported {
          background: #10b981;
          box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }
        
        .partially-supported {
          background-color: rgba(245, 158, 11, 0.2);
          border: 1px solid rgba(245, 158, 11, 0.3);
        }
        
        .accuracy-bar.partially-supported {
          background: #f59e0b;
          box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        
        .inconclusive {
          background-color: rgba(245, 158, 11, 0.15);
          border: 1px solid rgba(245, 158, 11, 0.2);
        }
        
        .accuracy-bar.inconclusive {
          background: #f59e0b;
          box-shadow: 0 0 10px rgba(245, 158, 11, 0.5);
        }
        
        .refuted {
          background-color: rgba(239, 68, 68, 0.2);
          border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .accuracy-bar.refuted {
          background: #ef4444;
          box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        
        /* Back to Top Button - Enhanced styling */
        .back-to-top-wrapper {
          position: fixed;
          bottom: 2rem;
          right: 2rem;
          z-index: 9999;
        }
        
        .back-to-top-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          background: #3b82f6;
          color: white;
          border: none;
          border-radius: 9999px;
          padding: 12px 20px;
          font-size: 1rem;
          font-weight: 600;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25), 0 0 20px rgba(59, 130, 246, 0.5);
          cursor: pointer;
          transition: all 0.3s ease;
        }
        
        .back-to-top-button:hover {
          background: #2563eb;
          transform: translateY(-5px);
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3), 0 0 30px rgba(59, 130, 246, 0.6);
        }
        
        @media (max-width: 768px) {
          .back-to-top-button {
            padding: 10px;
          }
          
          .back-to-top-button span {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};

export default PaperDisplay;
