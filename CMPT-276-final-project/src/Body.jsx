import React, { useState } from 'react';
import PaperDisplay from './PaperDisplay';
import MagneticElement from './MagneticElement';

export default function Body() {
  const [showExamples, setShowExamples] = useState(true);
  const [showResults, setShowResults] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [submittedQuery, setSubmittedQuery] = useState('');

  const handleSearch = (event) => {
    if (event) {
      event.preventDefault(); // Prevent form submission default behavior
    }
    if (searchQuery.trim()) {
      setShowExamples(false);
      setShowResults(true);
      setSubmittedQuery(searchQuery);
    }
  };

  const handleExampleClick = (query) => {
    setSearchQuery(query);
    setShowExamples(false);
    setShowResults(true);
    setSubmittedQuery(query);
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      handleSearch(event);
    }
  };

  return (
    <main>
      <section className="hero">
        <div className="container">
          <h1 className="hero-title">
            Your <span className="highlight">AI-powered</span> lens into<br />
            checking<br />
            facts.
          </h1>
          
          <div className="search-container">
            <input
              type="text"
              className="search-input"
              placeholder="Type a scientific claim or topic to explore :)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <div className="search-button-wrapper">
              <MagneticElement strength={40}>
                <button className="search-submit" onClick={handleSearch}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </MagneticElement>
            </div>
          </div>

          {showExamples && (
            <div className="example-queries">
              <h2 className="example-title">
                <span className="sparkle">✨</span> Try Example Queries <span className="sparkle">✨</span>
              </h2>
              <div className="query-buttons">
                <MagneticElement strength={25} distance={80}>
                  <button className="query-button" onClick={() => handleExampleClick("Are artificial sweeteners bad for health?")}>
                    Are artificial sweeteners bad for health?
                  </button>
                </MagneticElement>
                <MagneticElement strength={25} distance={80}>
                  <button className="query-button" onClick={() => handleExampleClick("A high-protein diet is effective for building muscle mass")}>
                    A high-protein diet is effective for building muscle mass
                  </button>
                </MagneticElement>
                <MagneticElement strength={25} distance={80}>
                  <button className="query-button" onClick={() => handleExampleClick("Reducing meat consumption can significantly lower an individual's carbon footprint")}>
                    Reducing meat consumption can significantly lower an individual's carbon footprint
                  </button>
                </MagneticElement>
                <MagneticElement strength={25} distance={80}>
                  <button className="query-button" onClick={() => handleExampleClick("Aspirin can help prevent heart attacks in high-risk individuals")}>
                    Aspirin can help prevent heart attacks in high-risk individuals
                  </button>
                </MagneticElement>
                <MagneticElement strength={25} distance={80}>
                  <button className="query-button" onClick={() => handleExampleClick("Regular exercise can improve cognitive function and mental health")}>
                    Regular exercise can improve cognitive function and mental health
                  </button>
                </MagneticElement>
              </div>
            </div>
          )}

          {showResults && <PaperDisplay claim={submittedQuery} />}
        </div>
      </section>
    </main>
  );
}
