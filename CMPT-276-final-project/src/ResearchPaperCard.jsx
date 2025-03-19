import React from 'react';

const ResearchPaperCard = ({
  title,
  author,
  date,
  abstract,
  categories = [],
  publisher = '',
  badgeText = ''
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
          <span>{badgeText}</span>
        </div>
      )}
      
      {/* Title Section */}
      <h1>{title}</h1>
      
      {/* Author and Date */}
      <div className="author-date">
        <span>{author}</span>
        {author && date && <span>-</span>}
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
