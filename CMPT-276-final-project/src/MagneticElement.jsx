import React, { useRef, useState, useEffect } from 'react';

// MagneticElement: Wrapper component that adds a magnetic pull effect to buttons and links
const MagneticElement = ({ children, strength = 30, className = '', distance = 100 }) => {
  const elementRef = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const handleMouseMove = (e) => {
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      
      // Calculate distance from cursor to element center
      const distanceX = e.clientX - centerX;
      const distanceY = e.clientY - centerY;
      
      // Calculate absolute distance
      const absoluteDistance = Math.sqrt(distanceX ** 2 + distanceY ** 2);
      
      // Only apply magnetic effect if cursor is within the distance threshold
      if (absoluteDistance < distance) {
        // Calculate strength based on distance (stronger when closer)
        const strengthFactor = 1 - (absoluteDistance / distance);
        const moveX = distanceX * strengthFactor * (strength / 100);
        const moveY = distanceY * strengthFactor * (strength / 100);
        
        setPosition({ x: moveX, y: moveY });
        setIsHovered(true);
      } else if (isHovered) {
        setPosition({ x: 0, y: 0 });
        setIsHovered(false);
      }
    };

    const handleMouseLeave = () => {
      setPosition({ x: 0, y: 0 });
      setIsHovered(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      if (element) {
        element.removeEventListener('mouseleave', handleMouseLeave);
      }
    };
  }, [isHovered, strength, distance]);

  return (
    <div 
      ref={elementRef}
      className={`magnetic-element ${className} ${isHovered ? 'magnetic-hovered' : ''}`}
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
        transition: isHovered ? 'transform 0.2s cubic-bezier(0.23, 1, 0.32, 1)' : 'transform 0.5s cubic-bezier(0.23, 1, 0.32, 1)',
      }}
    >
      {children}
      
      <style jsx>{`
        .magnetic-element {
          display: inline-block;
          will-change: transform;
        }
        
        .magnetic-hovered {
          z-index: 2;
        }
      `}</style>
    </div>
  );
};

export default MagneticElement; 