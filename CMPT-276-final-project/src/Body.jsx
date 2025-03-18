export default function Body() {















    
    return(
        <main>
        <section className="hero">
          <div className="container">
            <h1 className="hero-title">
              Your <span className="highlight">AI-powered</span> lens into<br />
              checking<br />
              facts.
            </h1>
            
            <div className="search-container">
              <input type="text" className="search-input" placeholder="Type a scientific claim or topic to explore :)"/>
              <button className="search-submit">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
            </div>

            <div className="example-queries">
              <h2 className="example-title">
                <span className="sparkle">✨</span> Try Example Queries <span className="sparkle">✨</span>
              </h2>
              <div className="query-buttons">
                <button className="query-button">Are artificial sweeteners bad for health?</button>
                <button className="query-button">A high-protein diet is effective for building muscle mass</button>
                <button className="query-button">Reducing meat consumption can significantly lower an individual's carbon footprint</button>
                <button className="query-button">Aspirin can help prevent heart attacks in high-risk individuals</button>
                <button className="query-button">Regular exercise can improve cognitive function and mental health</button>
              </div>
            </div>
          </div>
        </section>
      </main> 
    )
}