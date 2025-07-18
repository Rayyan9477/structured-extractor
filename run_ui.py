"""
Streamlit UI Launcher for Medical Superbill Data Extraction System
"""
import os
import sys
from pathlib import Path
import streamlit as st

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.extraction_engine import ExtractionEngine
from ui.app import run_ui_app

def main():
    """Initialize engine and launch the Streamlit UI."""
    try:
        # Initialize the extraction engine
        engine = ExtractionEngine()
        
        # Run the Streamlit app
        run_ui_app(engine)
        
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        print("Please ensure all dependencies are installed from requirements.txt")

if __name__ == "__main__":
    main()
