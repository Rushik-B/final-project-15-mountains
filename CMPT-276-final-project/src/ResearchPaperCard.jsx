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

  return (
    <div className="research-paper-card">
      {/* Badge */}
      {badgeText && (
        <div className="badge">
          {badgeText}
        </div>
      )}
      
      {/* Title Section */}
      <h1 className="paper-title">{title}</h1>
      
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
        <span>Source: {source}</span>
        <span className="citation-count">Citations: {citation_count}</span>
      </div>
      
      {/* URL/Link to Paper */}
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
        }
      `}</style>
    </div>
  );
};

export default ResearchPaperCard;
