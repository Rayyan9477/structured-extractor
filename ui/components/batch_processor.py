"""
Batch Processing Component for Medical Superbill Extractor UI
"""
import streamlit as st
import os
from pathlib import Path
import pandas as pd
import time
import asyncio
from src.extraction_engine import ExtractionEngine
from src.core.data_schema import ExtractionResults


def render_batch_processor(engine: ExtractionEngine):
    """
    Render the batch processing component for handling multiple files.
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Batch Processing</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Upload multiple medical superbill documents for batch processing.
    Results will be combined into a single export file.
    """)
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "jpg", "jpeg", "png", "tiff", "tif"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    # Configuration options
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")
        
        # Display file list
        file_df = pd.DataFrame({
            "File Name": [file.name for file in uploaded_files],
            "Type": [Path(file.name).suffix.lower() for file in uploaded_files],
            "Size (KB)": [f"{file.size / 1024:.2f}" for file in uploaded_files]
        })
        
        st.dataframe(file_df, use_container_width=True, hide_index=True)
        
        # Batch processing options
        st.markdown("### Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            anonymize_phi = st.toggle("Anonymize PHI", value=False)
            validate_codes = st.toggle("Validate Medical Codes", value=True)
        
        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["CSV", "JSON", "Excel"],
                index=0
            )
            
            create_summary = st.toggle("Create Summary Report", value=True)
        
        # Process button
        if st.button("Process Batch", type="primary", use_container_width=True):
            with st.spinner("Processing files..."):
                # Setup progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create a placeholder for results summary
                results_placeholder = st.empty()
                
                # Process each file with visual feedback
                processed_results = []
                
                temp_dir = Path("temp_batch")
                temp_dir.mkdir(exist_ok=True)
                
                file_paths = []
                for file in uploaded_files:
                    file_path = temp_dir / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(str(file_path))

                try:
                    batch_results = asyncio.run(engine.extract_batch(file_paths))
                    
                    for i, result in enumerate(batch_results):
                        status_text.text(f"Processed file {i+1}/{len(uploaded_files)}: {Path(result.file_path).name}")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        processed_results.append({
                            "file_name": Path(result.file_path).name,
                            "status": "Success" if result.success else "Failed",
                            "patients_found": result.total_patients,
                            "confidence": f"{result.extraction_confidence:.2%}"
                        })
                    
                    status_text.text("Batch processing complete!")
                    
                    results_df = pd.DataFrame(processed_results)
                    results_placeholder.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    st.success(f"Successfully processed {len(uploaded_files)} files!")
                    
                    # Add export buttons for the entire batch
                    
                finally:
                    # Clean up temp files
                    for path in file_paths:
                        os.remove(path)
                    if temp_dir.exists():
                        os.rmdir(temp_dir)

    else:
        # Show instructions when no files are uploaded
        st.info("Upload multiple files to begin batch processing.")
        
        # Show features
        st.markdown("""
        ### Batch Processing Features:
        - Process multiple superbills in one operation
        - Generate combined reports and summaries
        - Export all results in CSV, JSON, or Excel format
        - Track processing progress with detailed status updates
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)


def _generate_excel_bytes(df):
    """Generate Excel file from DataFrame and return bytes."""
    import io
    from io import BytesIO
    
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')
    buffer.seek(0)
    return buffer.read()
