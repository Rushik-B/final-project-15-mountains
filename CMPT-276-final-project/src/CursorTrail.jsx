import React, { useState, useEffect } from 'react';

const CursorTrail = () => {
  const [trail, setTrail] = useState([]);
  const [isMoving, setIsMoving] = useState(false);
  const trailLength = 8; // Number of trail particles
  
  useEffect(() => {
    let timeout;
    let animationFrame;
    
    const updateTrail = (e) => {
      setIsMoving(true);
      clearTimeout(timeout);
      
      // Add new point to the trail
      setTrail((prevTrail) => {
        const newTrail = [
          { x: e.clientX, y: e.clientY, timestamp: Date.now() },
          ...prevTrail,
        ].slice(0, trailLength);
        
        return newTrail;
      });
      
      // Set timeout to detect when movement stops
      timeout = setTimeout(() => {
        setIsMoving(false);
      }, 100);
    };
    
    // Track mouse movement
    window.addEventListener('mousemove', updateTrail);
    
    // Animation for trail fade out
    const animate = () => {
      setTrail((prevTrail) => 
        prevTrail.map((point) => ({
          ...point,
          // Add age property based on timestamp
          age: (Date.now() - point.timestamp) / 1000,
        }))
      );
      
      animationFrame = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      window.removeEventListener('mousemove', updateTrail);
      clearTimeout(timeout);
      cancelAnimationFrame(animationFrame);
    };
  }, []);
  
  return (
    <>
      {trail.map((point, index) => {
        // Calculate opacity based on position in trail (older = more transparent)
        const opacity = isMoving ? Math.max(0, 0.5 - (index * 0.06) - (point.age || 0)) : 0;
        const size = Math.max(4, 8 - index); // Size decreases as index increases
        
        return (
          <div
            key={index}
            className="cursor-trail-particle"
            style={{
              left: `${point.x}px`,
              top: `${point.y}px`,
              width: `${size}px`,
              height: `${size}px`,
              opacity: opacity,
              transform: `translate(-50%, -50%)`,
            }}
          />
        );
      })}
      
      <style jsx>{`
        .cursor-trail-particle {
          position: fixed;
          border-radius: 50%;
          background: rgba(37, 99, 235, 0.5);
          pointer-events: none;
          z-index: 9998;
          transition: opacity 0.2s ease;
          mix-blend-mode: screen;
        }
        
        /* For touch devices - hide trail */
        @media (pointer: coarse) {
          .cursor-trail-particle {
            display: none;
          }
        }
      `}</style>
    </>
  );
};

export default CursorTrail; 