import React, { useState, useEffect } from 'react';

const CustomCursor = () => {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [clicked, setClicked] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const updatePosition = (e) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    const handleMouseDown = () => setClicked(true);
    const handleMouseUp = () => setClicked(false);
    
    const handleMouseEnter = () => setHidden(false);
    const handleMouseLeave = () => setHidden(true);

    // Track interactive elements for hover state
    const addHoverListeners = () => {
      const interactiveElements = document.querySelectorAll(
        'a, button, input, .search-input, .evidence-link, .keyword-chip, .query-button, .toggle-detail-btn, .simplify-button'
      );
      
      interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => setHovered(true));
        el.addEventListener('mouseleave', () => setHovered(false));
      });
    };

    // Add event listeners
    document.addEventListener('mousemove', updatePosition);
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('mouseenter', handleMouseEnter);
    document.addEventListener('mouseleave', handleMouseLeave);
    
    // Initialize hover listeners and refresh them periodically
    addHoverListeners();
    const hoverListenerInterval = setInterval(addHoverListeners, 2000);

    // Remove cursor from cursor-css class element to hide default cursor
    document.body.classList.add('cursor-css');

    return () => {
      document.removeEventListener('mousemove', updatePosition);
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('mouseenter', handleMouseEnter);
      document.removeEventListener('mouseleave', handleMouseLeave);
      clearInterval(hoverListenerInterval);
      document.body.classList.remove('cursor-css');
    };
  }, []);

  return (
    <>
      <div
        className={`cursor-ring ${clicked ? 'cursor-clicked' : ''} ${hovered ? 'cursor-hovered' : ''} ${hidden ? 'cursor-hidden' : ''}`}
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
        }}
      />
      <div
        className={`cursor-dot ${clicked ? 'cursor-clicked' : ''} ${hovered ? 'cursor-hovered' : ''} ${hidden ? 'cursor-hidden' : ''}`}
        style={{
          left: `${position.x}px`,
          top: `${position.y}px`,
        }}
      />

      <style jsx>{`
        .cursor-ring {
          position: fixed;
          width: 30px;
          height: 30px;
          border: 2px solid rgba(37, 99, 235, 0.6);
          border-radius: 50%;
          transform: translate(-50%, -50%);
          pointer-events: none;
          z-index: 9999;
          transition: width 0.2s, height 0.2s, border-color 0.2s, opacity 0.2s;
          transition-timing-function: cubic-bezier(0.23, 1, 0.32, 1);
          mix-blend-mode: difference;
          backdrop-filter: invert(0.2);
        }
        
        .cursor-dot {
          position: fixed;
          width: 8px;
          height: 8px;
          background-color: rgb(37, 99, 235);
          border-radius: 50%;
          transform: translate(-50%, -50%);
          pointer-events: none;
          z-index: 10000;
          transition: width 0.15s, height 0.15s, background-color 0.15s, opacity 0.15s;
          transition-timing-function: cubic-bezier(0.23, 1, 0.32, 1);
        }
        
        .cursor-clicked {
          transform: translate(-50%, -50%) scale(0.8);
          opacity: 0.9;
        }
        
        .cursor-clicked.cursor-ring {
          width: 26px;
          height: 26px;
          border-color: rgba(249, 115, 22, 0.8);
          border-width: 3px;
        }
        
        .cursor-clicked.cursor-dot {
          width: 6px;
          height: 6px;
          background-color: rgb(249, 115, 22);
        }
        
        .cursor-hovered.cursor-ring {
          width: 40px;
          height: 40px;
          border-color: rgba(37, 99, 235, 0.8);
          border-width: 2px;
        }
        
        .cursor-hovered.cursor-dot {
          width: 6px;
          height: 6px;
          background-color: rgb(37, 99, 235);
        }
        
        .cursor-hidden {
          opacity: 0;
        }
        
        /* For touch devices - hide custom cursor */
        @media (pointer: coarse) {
          .cursor-ring, .cursor-dot {
            display: none;
          }
        }
      `}</style>
    </>
  );
};

export default CustomCursor; 