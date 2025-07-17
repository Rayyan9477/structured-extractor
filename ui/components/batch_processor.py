"""
Batch Processing Component for Medical Superbill Extractor UI
"""
import streamlit as st
import os
from pathlib import Path
import pandas as pd
import time


def render_batch_processor():
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
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Simulate processing steps with progress updates
                    for step in range(5):
                        time.sleep(0.2)  # Simulate processing time
                        sub_progress = (i * 5 + step + 1) / (len(uploaded_files) * 5)
                        progress_bar.progress(sub_progress)
                    
                    # Add simulated result (replace with actual processing)
                    processed_results.append({
                        "file_name": file.name,
                        "status": "Success",
                        "patients_found": 1,
                        "cpt_codes_found": 3,
                        "diagnosis_codes_found": 2
                    })
                
                # Complete the progress
                progress_bar.progress(1.0)
                status_text.text("Batch processing complete!")
                
                # Display results summary
                results_df = pd.DataFrame(processed_results)
                results_placeholder.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Show export options
                st.success(f"Successfully processed {len(uploaded_files)} files!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "Download CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name="batch_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        "Download Excel",
                        data=_generate_excel_bytes(results_df),
                        file_name="batch_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                with col3:
                    st.download_button(
                        "Download JSON",
                        data=results_df.to_json(orient="records"),
                        file_name="batch_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
    
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
