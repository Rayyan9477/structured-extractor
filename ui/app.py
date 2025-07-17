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
from src.extraction_engine import ExtractionEngine
from src.core.config_manager import ConfigManager
from src.core.data_schema import ExtractionResult


# Page configuration
st.set_page_config(
    page_title="Medical Superbill Extractor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
def load_css():
    with open(os.path.join(ROOT_DIR, "ui", "assets", "style.css")) as f:
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


def main():
    # Load CSS and background
    load_css()
    
    # Background overlay with gradient
    st.markdown("""
    <div class="bg-overlay"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'extraction_engine' not in st.session_state:
        st.session_state.extraction_engine = None
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'config' not in st.session_state:
        st.session_state.config = ConfigManager()
        
    # Initialize the extraction engine if not already done
    if st.session_state.extraction_engine is None:
        with st.spinner("Initializing extraction engine..."):
            st.session_state.extraction_engine = ExtractionEngine()
    
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
    
    # Content based on selected tab
    if selected_tab == "Single File":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # File uploader component
            uploaded_file = render_file_uploader()
            
            if uploaded_file:
                st.session_state.current_file = uploaded_file
                
                # Extract button
                extract_col, settings_col = st.columns([2, 1])
                with extract_col:
                    if st.button("Extract Data", type="primary", use_container_width=True):
                        with st.spinner("Extracting data from document..."):
                            # Save the uploaded file to a temporary location
                            temp_file_path = os.path.join(ROOT_DIR, "temp", uploaded_file.name)
                            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
                            
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Process the file
                            try:
                                progress_bar = st.progress(0)
                                
                                # Simulate a multi-step process with progress
                                for i in range(5):
                                    # Perform extraction steps
                                    if i == 0:
                                        status_text = st.empty()
                                        status_text.text("Initializing document processing...")
                                    elif i == 1:
                                        status_text.text("Performing OCR on document...")
                                    elif i == 2:
                                        status_text.text("Detecting fields and structures...")
                                    elif i == 3:
                                        status_text.text("Extracting structured data...")
                                    elif i == 4:
                                        status_text.text("Validating extracted information...")
                                    
                                    progress_bar.progress((i + 1) * 20)
                                    time.sleep(0.5)
                                
                                # TODO: Replace with actual extraction logic
                                # extraction_results = await st.session_state.extraction_engine.extract_from_file(temp_file_path)
                                # st.session_state.extraction_results = extraction_results
                                
                                # Simulated results for now
                                st.session_state.extraction_results = {
                                    "success": True,
                                    "patients": [
                                        {
                                            "patient_name": "John Doe",
                                            "patient_dob": "1980-05-15",
                                            "patient_id": "PT12345",
                                            "cpt_codes": [{"code": "99213", "description": "Office visit, est patient"}, 
                                                          {"code": "85027", "description": "Complete CBC"}],
                                            "diagnosis_codes": [{"code": "E11.9", "description": "Type 2 diabetes"}],
                                            "date_of_service": "2025-06-30",
                                            "provider_info": {"name": "Dr. Sarah Smith", "npi": "1234567890"}
                                        }
                                    ]
                                }
                                
                                progress_bar.progress(100)
                                status_text.text("Extraction complete!")
                                time.sleep(0.5)
                                status_text.empty()
                                progress_bar.empty()
                                
                                st.success("Data extraction completed successfully!")
                                
                                # Clean up
                                os.remove(temp_file_path)
                                
                            except Exception as e:
                                st.error(f"Error extracting data: {str(e)}")
                
                # Extraction settings
                with settings_col:
                    anonymize_phi = st.toggle("Anonymize PHI", value=False)
        
        with col2:
            # Results display
            if st.session_state.extraction_results:
                render_extraction_results(st.session_state.extraction_results)
            else:
                st.markdown("""
                <div class="info-container">
                    <div class="info-icon">ℹ️</div>
                    <div class="info-content">
                        <h3>No extraction results yet</h3>
                        <p>Upload a medical superbill document and click 'Extract Data' to start the extraction process.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Features highlight
                st.markdown("""
                <div class="features-container">
                    <h3>Key Features</h3>
                    <div class="feature-grid">
                        <div class="feature-card">
                            <div class="feature-icon">🔍</div>
                            <div class="feature-title">Multi-Model OCR</div>
                            <div class="feature-desc">Combines advanced OCR models for superior text recognition</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">🧠</div>
                            <div class="feature-title">NLP Extraction</div>
                            <div class="feature-desc">Uses AI to understand and extract medical data</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">👥</div>
                            <div class="feature-title">Multi-Patient</div>
                            <div class="feature-desc">Handles multiple patients in a single document</div>
                        </div>
                        <div class="feature-card">
                            <div class="feature-icon">🔐</div>
                            <div class="feature-title">HIPAA Compliant</div>
                            <div class="feature-desc">PHI detection and anonymization</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # Export options and validation panel
        if st.session_state.extraction_results:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                render_export_options(st.session_state.extraction_results)
            
            with col2:
                render_validation_panel(st.session_state.extraction_results)
    
    elif selected_tab == "Batch Processing":
        render_batch_processor()
    
    elif selected_tab == "Configuration":
        render_extraction_config()
    
    elif selected_tab == "Documentation":
        st.markdown("""
        <div class="documentation-container">
            <h2>Medical Superbill Extractor Documentation</h2>
            <p>This application uses advanced AI models to extract structured data from medical superbills.</p>
            
            <h3>How to Use</h3>
            <ol>
                <li><strong>Upload File</strong>: Upload a medical superbill in PDF or image format</li>
                <li><strong>Extract Data</strong>: Click the Extract Data button to process the document</li>
                <li><strong>Review Results</strong>: Examine the extracted information and make any corrections</li>
                <li><strong>Export</strong>: Export the results in your preferred format (JSON, CSV, Excel)</li>
            </ol>
            
            <h3>Extracted Fields</h3>
            <ul>
                <li><strong>Patient Information</strong>: Name, DOB, ID, contact details</li>
                <li><strong>Medical Codes</strong>: CPT codes, ICD-10 diagnosis codes</li>
                <li><strong>Service Information</strong>: Date of service, procedures</li>
                <li><strong>Provider Information</strong>: Name, NPI, contact details</li>
                <li><strong>Financial Information</strong>: Charges, copays, adjustments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
