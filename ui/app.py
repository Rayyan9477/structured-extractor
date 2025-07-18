"""
Medical Superbill Data Extraction System - Streamlit UI

A modern and attractive UI for the medical superbill extraction system,
providing easy access to all features in an intuitive interface.
"""
import streamlit as st
import os
import sys
import base64
from pathlib import Path
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
import time
import asyncio

# Add the project root to the path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import components
from ui.components.sidebar import render_sidebar
from ui.components.file_uploader import render_file_uploader
from ui.components.extraction_results import render_extraction_results
from ui.components.batch_processor import render_batch_processor
from ui.components.extraction_config import render_extraction_config
from ui.components.validation_panel import render_validation_panel
from ui.components.export_options import render_export_options

# Import system components
from src.core.data_schema import ExtractionResults
from src.extraction_engine import ExtractionEngine
from src.core.config_manager import ConfigManager


# Page configuration
st.set_page_config(
    page_title="Medical Superbill Extractor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    css_path = os.path.join(ROOT_DIR, "ui", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)


def run_ui_app(engine: ExtractionEngine):
    """
    Main function to run the Streamlit UI.
    
    Args:
        engine: Initialized extraction engine
    """
    # Load CSS
    load_css()
    
    # Background overlay with gradient
    st.markdown('<div class="bg-overlay"></div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'extraction_engine' not in st.session_state:
        st.session_state.extraction_engine = engine
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
        
    # Title and description
    st.markdown("""
    <div class="header-container">
        <h1 class="main-title">Medical Superbill Data Extraction</h1>
        <p class="subtitle">Extract structured data from medical superbills using advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    render_sidebar()
    
    # Main navigation
    selected_tab = option_menu(
        menu_title=None,
        options=["Single File", "Batch Processing", "Configuration", "Documentation"],
        icons=["file-earmark-medical", "files", "gear", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(255, 255, 255, 0.1)", "border-radius": "10px"},
            "icon": {"color": "#6236FF", "font-size": "20px"},
            "nav-link": {"text-align": "center", "margin": "0px", "padding": "10px 20px", "color": "#333", "font-weight": "normal"},
            "nav-link-selected": {"background-color": "#6236FF", "color": "white", "font-weight": "bold"},
        }
    )
    
    if selected_tab == "Single File":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_file = render_file_uploader()
            
            if uploaded_file:
                if st.button("Extract Data", type="primary", use_container_width=True):
                    with st.spinner("Extracting data..."):
                        # Save temp file
                        temp_dir = ROOT_DIR / "temp"
                        temp_dir.mkdir(exist_ok=True)
                        temp_file_path = temp_dir / uploaded_file.name
                        
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        try:
                            # Run extraction
                            results = asyncio.run(st.session_state.extraction_engine.extract_from_file(str(temp_file_path)))
                            st.session_state.extraction_results = results
                            st.success("Extraction complete!")
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")
                        finally:
                            # Clean up
                            if temp_file_path.exists():
                                os.remove(temp_file_path)

        with col2:
            if st.session_state.extraction_results:
                render_extraction_results(st.session_state.extraction_results)
            else:
                st.info("Upload a file and click 'Extract Data' to begin.")
                
    elif selected_tab == "Batch Processing":
        render_batch_processor(st.session_state.extraction_engine)
        
    elif selected_tab == "Configuration":
        render_extraction_config()
    elif selected_tab == "Documentation":
        st.info("Documentation section coming soon.")

if __name__ == "__main__":
    # This part is for direct execution and might need a mock engine
    # For the main execution, run `run_ui.py`
    from src.core.config_manager import ConfigManager
    
    if 'extraction_engine' not in st.session_state:
        # Create a mock or real engine for direct testing
        config = ConfigManager()
        st.session_state.extraction_engine = ExtractionEngine(config)
        
    run_ui_app(st.session_state.extraction_engine)
