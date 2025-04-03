import React from 'react';
import factifyLogo from './assets/Factify-Logo.jpeg';
import './index.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-logo">
          <img src={factifyLogo} alt="Factify Logo" />
        </div>
        <div className="footer-copyright">
          <p>© 2025 Factify. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
} 