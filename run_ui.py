#!/usr/bin/env python
"""
Simple UI Execution File for Medical Superbill Extractor
Run this file to start the Streamlit UI application.

Features:
- Nanonets-OCR for text extraction  
- NuExtract-2.0-8B for structured data extraction
- No fallback models - clean implementation
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit UI application."""
    print("Medical Superbill Extractor - Starting UI...")
    print("=" * 50)
    
    # Get the directory of this script
    current_dir = Path(__file__).parent
    ui_app_path = current_dir / "ui" / "app.py"
    
    # Check if the UI app exists
    if not ui_app_path.exists():
        print(f"Error: UI app not found at {ui_app_path}")
        sys.exit(1)
    
    # Change to the project directory
    os.chdir(current_dir)
    
    # Command to run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(ui_app_path),
        "--theme.base", "light",
        "--theme.primaryColor", "#1f77b4", 
        "--theme.backgroundColor", "#ffffff",
        "--server.port", "8503",
        "--server.address", "localhost"
    ]
    
    print(f"Starting Streamlit on http://localhost:8503")
    print(f"Using models: Nanonets-OCR + NuExtract-2.0-8B")
    print(f"Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
