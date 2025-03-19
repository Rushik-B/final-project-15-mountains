#!/usr/bin/env python3
"""
Factify API runner script
Starts the Factify Flask application with proper configuration
"""

import os
from app import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv('PORT', 8080))
    
    # Determine debug status
    debug = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    print(f"Starting Factify API on port {port} (debug={debug})")
    print("Press CTRL+C to quit")
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=port, debug=debug)