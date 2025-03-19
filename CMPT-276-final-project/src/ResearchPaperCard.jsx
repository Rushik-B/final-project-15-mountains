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
  url = ''
}) => {
  // Format date to DD/MM/YYYY if provided as a Date object
  const formattedDate = typeof date === 'object' && date instanceof Date
    ? `${date.getDate().toString().padStart(2, '0')}/${(date.getMonth() + 1).toString().padStart(2, '0')}/${date.getFullYear()}`
    : date;

  return (
    <div className="research-paper-card">
      {/* Title Section */}
      <h1>{title}</h1>
      
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
      </div>
      
      {/* URL/Link to Paper */}
      {url && (
        <div className="paper-url">
          <a href={url} target="_blank" rel="noopener noreferrer">
            Access Full Paper ↗
          </a>
        </div>
      )}
      
      {/* Badge */}
      {badgeText && (
        <div className="badge">
          {badgeText}
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
    </div>
  );
};

export default ResearchPaperCard;
