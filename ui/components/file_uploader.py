"""
File Uploader Component for Medical Superbill Extractor UI
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image


def render_file_uploader():
    """
    Render the file uploader component with drag and drop functionality.
    
    Returns:
        The uploaded file object if a file was uploaded, otherwise None
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Upload Document</div>
    """, unsafe_allow_html=True)
    
    # File types explanation
    file_col, format_col = st.columns([3, 2])
    
    with file_col:
        # Add some descriptive text
        st.markdown("Upload a medical superbill document to extract data.")
        
        # Create the file uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "jpg", "jpeg", "png", "tiff", "tif"],
            label_visibility="collapsed"
        )
    
    with format_col:
        st.markdown("**Supported Formats:**")
        st.markdown("- PDF Documents (.pdf)")
        st.markdown("- Images (.jpg, .png, .tiff)")
    
    # Display preview if file is uploaded
    if uploaded_file is not None:
        # Get file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Show file info
        st.markdown(f"**File:** {uploaded_file.name}")
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        # Display file details in a cleaner format
        details_md = ""
        for key, value in file_details.items():
            details_md += f"**{key}:** {value}  \n"
        st.markdown(details_md)
        
        # Preview for images
        if file_extension in ['.jpg', '.jpeg', '.png']:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Document Preview", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        
        # For PDFs, just show an info message
        elif file_extension == '.pdf':
            st.info("PDF document loaded. Click 'Extract Data' to process.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return uploaded_file
