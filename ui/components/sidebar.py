"""
Sidebar Component for Medical Superbill Extractor UI
"""
import streamlit as st
import os
from pathlib import Path


def render_sidebar():
    """Render the sidebar with app info and settings."""
    with st.sidebar:
        # Logo and app info
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">🏥</div>
            <h2>Medical Superbill Extractor</h2>
            <p>v1.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Environment info
        with st.expander("Environment Info"):
            st.markdown("**Python Version:** 3.10.x")
            st.markdown("**Models:**")
            st.markdown("- MonkeyOCR")
            st.markdown("- Nanonets-OCR")
            st.markdown("- NuExtract-2.0-4B")
        
        # Quick links
        st.markdown("### Quick Links")
        st.markdown("- [Documentation](#)")
        st.markdown("- [GitHub Repository](https://github.com/Rayyan9477/structured-extractor)")
        st.markdown("- [Report an Issue](#)")
        
        # Settings
        st.markdown("### Settings")
        
        # Model selection
        st.selectbox(
            "OCR Model",
            ["Auto (Recommended)", "MonkeyOCR", "Nanonets-OCR", "Tesseract (Fallback)"],
            index=0
        )
        
        # Performance settings
        st.selectbox(
            "Performance Mode",
            ["Balanced", "High Quality", "Fast Processing"],
            index=0
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            # OCR settings
            st.slider("OCR Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
            
            # Extraction settings
            st.toggle("Enable Multi-Patient Detection", value=True)
            st.toggle("Validate CPT/ICD Codes", value=True)
            
            # Processing settings
            st.checkbox("Use GPU Acceleration (if available)", value=True)
            st.checkbox("Enable Memory Optimization", value=True)
            
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
            "© 2025 Medical Superbill Extraction Team"
            "</div>",
            unsafe_allow_html=True
        )
