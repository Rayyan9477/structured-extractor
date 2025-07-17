"""
Streamlit UI Launcher for Medical Superbill Data Extraction System
"""
import os
import sys
from pathlib import Path
import streamlit as st

# Add the project root to the path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR.parent))

# Run the Streamlit app
if __name__ == "__main__":
    print("Starting Medical Superbill Extractor UI...")
    os.system(f"streamlit run {os.path.join(ROOT_DIR, 'ui', 'app.py')}")
