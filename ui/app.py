"""
Medical Superbill Data Extraction System - Streamlit UI

A modern and attractive UI for the medical superbill extraction system,
providing easy access to all features in an intuitive interface.
"""
import streamlit as st
import os
import sys
import base64
import re
import uuid
from pathlib import Path
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
import time
import asyncio
import tempfile
from contextlib import contextmanager

# Add the project root to the path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Security constants
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}

def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent path traversal."""
    if not filename:
        return f"upload_{uuid.uuid4().hex[:8]}.tmp"
    
    # Remove path components and dangerous characters
    safe_name = os.path.basename(filename)
    safe_name = re.sub(r'[^\w\-_\.]', '_', safe_name)
    
    # Ensure filename is not empty or hidden
    if not safe_name or safe_name.startswith('.'):
        safe_name = f"upload_{uuid.uuid4().hex[:8]}.tmp"
    
    return safe_name

def get_safe_extension(filename: str) -> str:
    """Get safe file extension for temporary file."""
    ext = Path(filename).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else '.tmp'

@contextmanager
def secure_temp_file(data: bytes, suffix: str = '.tmp'):
    """Create secure temporary file with proper cleanup and validation."""
    # Validate file size
    if len(data) > MAX_UPLOAD_SIZE:
        raise ValueError(f"File too large: {len(data)} bytes (max: {MAX_UPLOAD_SIZE})")
    
    # Basic file content validation
    if len(data) < 10:  # Minimum viable file size
        raise ValueError("File appears to be empty or corrupted")
    
    temp_fd = None
    temp_path = None
    try:
        # Create secure temp file with proper permissions
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix='secure_extract_')
        os.chmod(temp_path, 0o600)  # Restrict permissions to owner only
        
        # Write data
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(data)
            temp_fd = None  # Prevent double close
        
        yield temp_path
    finally:
        # Ensure cleanup
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

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
    # Validate CSS path is within project directory
    try:
        css_path_resolved = Path(css_path).resolve()
        ROOT_DIR_resolved = Path(ROOT_DIR).resolve()
        css_path_resolved.relative_to(ROOT_DIR_resolved)
        
        if css_path_resolved.exists() and css_path_resolved.is_file():
            with open(css_path_resolved, 'r', encoding='utf-8') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except (ValueError, OSError) as e:
        st.error("Security: CSS file path validation failed")


def add_bg_from_local(image_file):
    """Safely load background image with path validation."""
    try:
        image_path = Path(image_file).resolve()
        ROOT_DIR_resolved = Path(ROOT_DIR).resolve()
        
        # Validate image path is within project directory
        image_path.relative_to(ROOT_DIR_resolved)
        
        if image_path.exists() and image_path.is_file():
            with open(image_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode()
            
            bg_img = f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
            }}
            </style>
            """
            st.markdown(bg_img, unsafe_allow_html=True)
    except (ValueError, OSError):
        # Silently fail for security - don't reveal path information
        pass


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
                        try:
                            # Use secure temporary file handling
                            with secure_temp_file(uploaded_file.getbuffer(), get_safe_extension(uploaded_file.name)) as temp_file_path:
                                # Run extraction
                                results = asyncio.run(st.session_state.extraction_engine.extract_from_file(str(temp_file_path)))
                                st.session_state.extraction_results = results
                                st.success("Extraction complete!")
                        except ValueError as e:
                            st.error("File validation failed. Please check file format and size.")
                        except Exception as e:
                            st.error("Processing failed. Please try again with a different file.")
                            # Log full error internally without exposing to user
                            import logging
                            logging.getLogger(__name__).error(f"Extraction failed: {str(e)}", exc_info=True)

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
