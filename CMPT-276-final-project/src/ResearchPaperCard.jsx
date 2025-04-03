import React from 'react';

const ResearchPaperCard = ({
  title,
  author,
  date,
  abstract,
  categories = [],
  publisher = '',
  badgeText = '',
  doi = '10.1002/abc.12345',
  published = '28/01/2023',
  source = 'crossref',
  url = '',
  citation_count = 0
}) => {
  // Format date to DD/MM/YYYY if provided as a Date object
  const formattedDate = typeof date === 'object' && date instanceof Date
    ? `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getFullYear()}`
    : date;

  // Helper function to get the appropriate CSS class for each source
  const getSourceClass = (sourceString) => {
    const sourceLower = sourceString.toLowerCase();
    
    if (sourceLower.includes('crossref')) return 'source-crossref';
    if (sourceLower.includes('openalex')) return 'source-openalex';
    if (sourceLower.includes('semantic')) return 'source-semantic-scholar';
    
    // Default fallback
    return 'source-default';
  };

  return (
    <div className="research-paper-card">
      {/* Badge */}
      {badgeText && (
        <div className="badge">
          {badgeText}
        </div>
      )}
      
      {/* Title Section - Now Clickable */}
      <h1 className="paper-title">
        {url ? (
          <a href={url} target="_blank" rel="noopener noreferrer" className="title-link">
            {title}
          </a>
        ) : (
          title
        )}
      </h1>
      
      {/* Author and Date */}
      <div className="author-date">
        <span>{author}</span>
        {author && formattedDate && <span> - </span>}
        <span>{formattedDate}</span>
      </div>
      
      {/* Categories */}
      <div className="categories">
        {categories.map((category, index) => (
          <span key={index} className="category">
            {category}
          </span>
        ))}
      </div>
      
      {/* DOI and Publication Info */}
      <div className="doi-info">
        <span>DOI: {doi}</span>
        <span>Published: {published}</span>
        <span className={`source-label ${getSourceClass(source)}`}>
          Source: {source}
        </span>
        <span className="citation-count">Citations: {citation_count}</span>
      </div>
      
      {/* URL/Link to Paper - Keeping the existing button */}
      {url && (
        <div className="paper-url">
          <a href={url} target="_blank" rel="noopener noreferrer">
            Access Full Paper ↗
          </a>
        </div>
      )}
      
      {/* Abstract Section */}
      <div className="abstract">
        <h2>Abstract</h2>
        <p>{abstract}</p>
      </div>
      
      {/* Publisher */}
      {publisher && (
        <div className="publisher">
          © {new Date().getFullYear()} {publisher}
        </div>
      )}
      
      <style jsx>{`
        /* Main card styling */
        .research-paper-card {
          position: relative;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
          background-color: white;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
          transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
          border: 1px solid rgba(0, 0, 0, 0.05);
          transform: translateZ(0);
          backface-visibility: hidden;
        }
        
        .research-paper-card:hover {
          transform: translateY(-5px) translateZ(0);
          box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1), 0 6px 6px rgba(0, 0, 0, 0.06);
          border-color: rgba(2, 132, 199, 0.2);
        }
        
        .research-paper-card:hover .badge {
          background-color: #0369a1;
          box-shadow: 0 3px 5px rgba(0, 0, 0, 0.2);
        }
        
        .research-paper-card:hover .paper-title {
          color: #0284c7;
        }
        
        /* Paper title with right padding to avoid badge overlap */
        .paper-title {
          padding-right: 100px; /* Make room for the badge */
          margin-top: 0;
          margin-bottom: 12px;
          font-size: 1.4rem;
          line-height: 1.3;
          color: #1f2937;
        }
        
        /* Title link styling */
        .title-link {
          color: inherit;
          text-decoration: none;
          transition: all 0.2s ease;
          cursor: pointer;
          position: relative;
          display: inline-block;
        }
        
        .title-link:hover {
          color: #0284c7;
        }
        
        .title-link::after {
          content: '';
          position: absolute;
          width: 100%;
          height: 2px;
          bottom: -2px;
          left: 0;
          background-color: #0284c7;
          transform: scaleX(0);
          transform-origin: bottom right;
          transition: transform 0.3s ease-out;
        }
        
        .title-link:hover::after {
          transform: scaleX(1);
          transform-origin: bottom left;
        }
      
        /* Citation count styling */
        .citation-count {
          font-weight: 600;
          color: #0284c7;
          margin-left: 10px;
          display: inline-flex;
          align-items: center;
        }
        
        .citation-count::before {
          content: '📚';
          margin-right: 4px;
          font-size: 14px;
        }
        
        /* URL link styling */
        .paper-url {
          margin: 15px 0;
        }
        
        .paper-url a {
          display: inline-block;
          color: #0284c7;
          font-weight: 500;
          text-decoration: none;
          padding: 6px 12px;
          border-radius: 4px;
          border: 1px solid #d1e1ff;
          background-color: #f0f5ff;
          transition: all 0.2s ease;
        }
        
        .paper-url a:hover {
          background-color: #0284c7;
          color: white;
          border-color: #0284c7;
          transform: translateY(-2px);
          box-shadow: 0 4px 8px rgba(2, 132, 199, 0.2);
        }
        
        /* Styling for the toggle button in PaperDisplay */
        :global(.toggle-detail-btn) {
          background-color: #f0f5ff;
          border: 1px solid #d1e1ff;
          border-radius: 4px;
          padding: 6px 12px;
          font-size: 0.85rem;
          cursor: pointer;
          transition: all 0.2s ease;
          color: #2563eb;
          font-weight: 500;
        }
        
        :global(.toggle-detail-btn:hover) {
          background-color: #e0eaff;
          border-color: #93b4ff;
        }
        
        :global(.reasoning-header) {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }
        
        :global(.reasoning-text) {
          font-size: 1rem;
          line-height: 1.6;
          color: #333;
          padding: 16px;
          background-color: #f9fafb;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
          transition: all 0.3s ease;
        }
        
        /* Evidence reference link styling */
        :global(.evidence-link) {
          color: #0066ff !important; /* Blue color, forced */
          font-weight: 600;
          cursor: pointer; /* Pointer cursor on hover */
          text-decoration: underline; /* Underline */
          transition: all 0.2s ease;
          display: inline-block;
          padding: 0 2px;
          border-radius: 3px;
          background-color: rgba(0, 102, 255, 0.08);
          margin: 0 1px;
          position: relative;
        }
        
        :global(.evidence-link:hover) {
          color: #0055cc !important; /* Darker blue on hover, forced */
          background-color: rgba(0, 102, 255, 0.15);
          transform: translateY(-1px);
          box-shadow: 0 1px 2px rgba(0, 102, 255, 0.2);
          text-decoration: underline; /* Keep underline */
        }
        
        :global(.evidence-link::after) {
          content: none; /* Remove previous custom underline */
        }
        
        :global(.evidence-note) {
          margin-top: 8px;
          font-size: 0.85rem;
          color: #6b7280;
          font-style: italic;
        }
        
        /* Evidence badge styling */
        .badge {
          position: absolute;
          top: 10px;
          right: 10px;
          background-color: #0284c7;
          color: white;
          padding: 4px 10px;
          border-radius: 4px;
          font-size: 0.8rem;
          font-weight: 600;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          z-index: 1;
          transition: all 0.3s ease;
        }
        
        /* Source highlighting with different colors */
        .source-label {
          font-weight: 500;
          border-radius: 4px;
          padding: 2px 6px;
          margin-right: 4px;
          transition: all 0.2s ease;
        }
        
        .source-crossref {
          background-color: rgba(57, 138, 229, 0.15);
          color: #2563eb;
        }
        
        .source-openalex {
          background-color: rgba(16, 185, 129, 0.15);
          color: #059669;
        }
        
        .source-semantic-scholar {
          background-color: rgba(139, 92, 246, 0.15);
          color: #7c3aed;
        }
        
        .source-default {
          background-color: rgba(107, 114, 128, 0.15);
          color: #4b5563;
        }
        
        .research-paper-card:hover .source-crossref {
          background-color: rgba(57, 138, 229, 0.25);
        }
        
        .research-paper-card:hover .source-openalex {
          background-color: rgba(16, 185, 129, 0.25);
        }
        
        .research-paper-card:hover .source-semantic-scholar {
          background-color: rgba(139, 92, 246, 0.25);
        }
        
        .research-paper-card:hover .source-default {
          background-color: rgba(107, 114, 128, 0.25);
        }
        
        @keyframes badgePulse {
          0% { transform: scale(1); }
          50% { transform: scale(1.05); }
          100% { transform: scale(1); }
        }
        
        .research-paper-card:hover .badge {
          animation: badgePulse 1.5s ease infinite;
        }
      `}</style>
    </div>
  );
};

export default ResearchPaperCard;
