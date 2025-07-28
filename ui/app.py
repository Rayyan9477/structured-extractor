import streamlit as st
import sys
import os
from pathlib import Path
import json
import pandas as pd
import asyncio
import time
import torch
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager
from src.extraction_engine import ExtractionEngine


def initialize_engine():
    """Initialize the extraction engine with optimized sequential loading."""
    if 'extraction_engine' not in st.session_state:
        config = ConfigManager()
        
        # Configure for sequential processing with GPU optimization
        config.update_config("models.sequential_loading", True)
        config.update_config("models.unload_after_use", True)
        config.update_config("processing.use_cuda", True)
        config.update_config("processing.mixed_precision", True)
        
        st.session_state.extraction_engine = ExtractionEngine(config)
        st.session_state.model_info = {
            'ocr_model': 'Nanonets OCR (GPU Optimized)',
            'extraction_model': 'NuExtract 2.0-8B',
            'processing_strategy': 'Sequential Loading',
            'gpu_acceleration': torch.cuda.is_available()
        }


def process_pdf(uploaded_file):
    """Process uploaded PDF file using optimized sequential processing."""
    # Save uploaded file to temporary location
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    start_time = time.time()
    
    try:
        # Show progress information
        with st.spinner("🚀 Initializing models sequentially for optimal performance..."):
            # Process file using the extraction engine with progress tracking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Extract data using the optimized engine
            extraction_result = loop.run_until_complete(
                st.session_state.extraction_engine.extract_from_file(str(temp_path))
            )
        
        processing_time = time.time() - start_time
        
        # Create enhanced result structure
        result = {
            'success': extraction_result.success,
            'processing_time': processing_time,
            'extraction_confidence': extraction_result.extraction_confidence,
            'extraction_result': extraction_result,
            'total_patients': extraction_result.total_patients,
            'metadata': {
                'file_size': len(uploaded_file.getbuffer()) / 1024 / 1024,  # MB
                'processing_strategy': 'Sequential: Nanonets → NuExtract',
                'gpu_used': torch.cuda.is_available(),
                'timestamp': datetime.now().isoformat()
            },
            'error': None
        }
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'extraction_result': None,
            'processing_time': processing_time,
            'metadata': {'error_type': type(e).__name__}
        }
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def display_extraction_results(result):
    """Display extracted medical data with enhanced UI and patient differentiation."""
    if not result['success']:
        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        if 'processing_time' in result:
            st.write(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
        return
    
    extraction_result = result['extraction_result']
    
    # Enhanced document overview with metrics
    st.subheader("📊 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    with col2:
        st.metric("Confidence Score", f"{result['extraction_confidence']:.1%}")
    with col3:
        st.metric("Patients Found", result.get('total_patients', 0))
    with col4:
        gpu_status = "🚀 GPU" if result['metadata']['gpu_used'] else "💻 CPU"
        st.metric("Processing Mode", gpu_status)
    
    # Technical details in expander
    with st.expander("🔧 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File Size:** {result['metadata']['file_size']:.1f} MB")
            st.write(f"**Strategy:** {result['metadata']['processing_strategy']}")
            st.write(f"**GPU Acceleration:** {'✅ Enabled' if result['metadata']['gpu_used'] else '❌ Disabled'}")
        with col2:
            if 'model_info' in st.session_state:
                info = st.session_state.model_info
                st.write(f"**OCR Model:** {info['ocr_model']}")
                st.write(f"**Extraction Model:** {info['extraction_model']}")
                st.write(f"**Loading Strategy:** {info['processing_strategy']}")
    
    # Enhanced patient data display
    st.subheader("👥 Extracted Patient Data")
    
    if extraction_result.patients:
        # Show overall statistics
        total_cpt = sum(len(p.cpt_codes) if p.cpt_codes else 0 for p in extraction_result.patients)
        total_icd = sum(len(p.icd10_codes) if p.icd10_codes else 0 for p in extraction_result.patients)
        
        st.info(f"📋 **Summary:** Found {len(extraction_result.patients)} patients with {total_cpt} CPT codes and {total_icd} ICD-10 codes")
        
        for i, patient in enumerate(extraction_result.patients):
            # Enhanced patient header with color coding
            patient_name = f"{patient.first_name or 'Unknown'} {patient.last_name or 'Patient'}"
            cpt_count = len(patient.cpt_codes) if patient.cpt_codes else 0
            icd_count = len(patient.icd10_codes) if patient.icd10_codes else 0
            
            with st.expander(f"👤 **Patient {i+1}:** {patient_name} | 🟢 {cpt_count} CPT | 🔵 {icd_count} ICD-10", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📝 Demographics")
                    st.write(f"**Full Name:** {patient.first_name or ''} {patient.middle_name or ''} {patient.last_name or ''}")
                    if patient.date_of_birth:
                        if hasattr(patient.date_of_birth, 'strftime'):
                            dob_str = patient.date_of_birth.strftime('%Y-%m-%d')
                        else:
                            dob_str = str(patient.date_of_birth)
                        st.write(f"**Date of Birth:** {dob_str}")
                    if patient.patient_id:
                        st.write(f"**Patient ID:** {patient.patient_id}")
                    
                    # Page information (if available)
                    if hasattr(patient, 'page_number'):
                        st.write(f"**Found on Page:** {patient.page_number}")
                    
                    if patient.financial_info:
                        st.markdown("### 💰 Financial Information")
                        if patient.financial_info.total_charges:
                            st.write(f"**Total Charges:** ${patient.financial_info.total_charges:.2f}")
                        if hasattr(patient.financial_info, 'copay') and patient.financial_info.copay:
                            st.write(f"**Copay:** ${patient.financial_info.copay:.2f}")
                        if hasattr(patient.financial_info, 'deductible') and patient.financial_info.deductible:
                            st.write(f"**Deductible:** ${patient.financial_info.deductible:.2f}")
                
                with col2:
                    if patient.cpt_codes:
                        st.markdown("### 🟢 CPT Codes (Procedures)")
                        cpt_data = []
                        for cpt in patient.cpt_codes:
                            cpt_data.append({
                                'Code': cpt.code,
                                'Description': cpt.description[:50] + "..." if cpt.description and len(cpt.description) > 50 else cpt.description or "-",
                                'Charge': f"${cpt.charge:.2f}" if cpt.charge else "-",
                                'Confidence': f"{cpt.confidence.overall:.1%}" if cpt.confidence and hasattr(cpt.confidence, 'overall') else "-"
                            })
                        st.dataframe(pd.DataFrame(cpt_data), use_container_width=True)
                    
                    if patient.icd10_codes:
                        st.markdown("### 🔵 ICD-10 Codes (Diagnoses)")
                        icd_data = []
                        for icd in patient.icd10_codes:
                            icd_data.append({
                                'Code': icd.code,
                                'Description': icd.description[:50] + "..." if icd.description and len(icd.description) > 50 else icd.description or "-",
                                'Confidence': f"{icd.confidence.overall:.1%}" if icd.confidence and hasattr(icd.confidence, 'overall') else "-"
                            })
                        st.dataframe(pd.DataFrame(icd_data), use_container_width=True)
                    
                    if patient.service_info:
                        st.markdown("### 🏥 Service Information")
                        if patient.service_info.date_of_service:
                            if hasattr(patient.service_info.date_of_service, 'strftime'):
                                service_date = patient.service_info.date_of_service.strftime('%Y-%m-%d')
                            else:
                                service_date = str(patient.service_info.date_of_service)
                            st.write(f"**Date of Service:** {service_date}")
                        if patient.service_info.provider_name:
                            st.write(f"**Provider:** {patient.service_info.provider_name}")
                        if patient.service_info.provider_npi:
                            st.write(f"**NPI:** {patient.service_info.provider_npi}")
                
                # Patient-specific insights
                if cpt_count > 0 or icd_count > 0:
                    st.markdown("---")
                    insight_col1, insight_col2, insight_col3 = st.columns(3)
                    with insight_col1:
                        st.metric("Procedures", cpt_count)
                    with insight_col2:
                        st.metric("Diagnoses", icd_count)
                    with insight_col3:
                        confidence = patient.extraction_confidence if hasattr(patient, 'extraction_confidence') else 0.85
                        st.metric("Confidence", f"{confidence:.1%}")
    else:
        st.warning("⚠️ No patient data found in the document. Please ensure the document contains medical superbill information.")


def main():
    st.set_page_config(
        page_title="Medical Superbill Extractor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Medical Superbill Extraction System")
    st.markdown("**Advanced AI-powered extraction with sequential model loading and GPU optimization**")
    
    initialize_engine()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File upload section with enhanced info
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a medical superbill PDF", 
            type="pdf",
            help="Upload a medical superbill or similar document for AI-powered data extraction"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = len(uploaded_file.getbuffer()) / 1024 / 1024
            st.success(f"✅ **{uploaded_file.name}** loaded ({file_size:.1f} MB)")
            
            if st.button("🚀 Extract Medical Data", type="primary"):
                result = process_pdf(uploaded_file)
                display_extraction_results(result)
                
                # Export options
                if result['success']:
                    st.subheader("📤 Export Options")
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        if st.button("📋 Download JSON"):
                            json_data = json.dumps(result, indent=2, default=str)
                            st.download_button(
                                label="💾 Save JSON File",
                                data=json_data,
                                file_name=f"{uploaded_file.name}_extraction.json",
                                mime="application/json"
                            )
                    
                    with export_col2:
                        if st.button("📊 Download CSV"):
                            # Create CSV data
                            csv_data = []
                            for i, patient in enumerate(result['extraction_result'].patients or []):
                                for cpt in patient.cpt_codes or []:
                                    csv_data.append({
                                        'Patient': f"{patient.first_name or ''} {patient.last_name or ''}".strip(),
                                        'Patient_ID': patient.patient_id or '',
                                        'CPT_Code': cpt.code,
                                        'CPT_Description': cpt.description or '',
                                        'Charge': cpt.charge or 0
                                    })
                            
                            if csv_data:
                                df = pd.DataFrame(csv_data)
                                csv_string = df.to_csv(index=False)
                                st.download_button(
                                    label="💾 Save CSV File",
                                    data=csv_string,
                                    file_name=f"{uploaded_file.name}_patients.csv",
                                    mime="text/csv"
                                )
                
                # Show raw JSON for debugging
                with st.expander("🔍 Raw Extraction Data (Advanced)"):
                    st.json(result, expanded=False)
    
    with col2:
        # Enhanced sidebar with system information
        st.subheader("🤖 System Status")
        
        if 'model_info' in st.session_state:
            info = st.session_state.model_info
            
            # Model Information
            st.markdown("### 🧠 AI Models")
            st.write(f"**OCR:** {info['ocr_model']}")
            st.write(f"**Extraction:** {info['extraction_model']}")
            st.write(f"**Strategy:** {info['processing_strategy']}")
            
            # Performance indicators
            st.markdown("### ⚡ Performance")
            gpu_status = "🚀 GPU Accelerated" if info['gpu_acceleration'] else "💻 CPU Processing"
            st.write(f"**Mode:** {gpu_status}")
            st.write(f"**VRAM Optimized:** ✅ Yes")
            st.write(f"**Sequential Loading:** ✅ Yes")
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name()
                st.write(f"**GPU:** {gpu_name}")
                
                # GPU memory info (if available)
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    st.write(f"**GPU Memory:** {total_memory:.1f} GB")
                except:
                    pass
        
        # Processing capabilities
        st.markdown("### 🎯 Capabilities")
        st.write("✅ Multi-patient detection")
        st.write("✅ CPT code extraction")
        st.write("✅ ICD-10 code extraction")
        st.write("✅ Patient differentiation")
        st.write("✅ Financial data extraction")
        st.write("✅ Service information")
        
        # Tips section
        st.markdown("### 💡 Tips")
        st.info("""
        **For best results:**
        - Use clear, high-quality PDFs
        - Ensure text is readable
        - Multi-page documents supported
        - Processing time varies by complexity
        """)
        
        # Version info
        st.markdown("---")
        st.caption("🔧 **Version:** 2.0 | **Engine:** Sequential AI")
        
        # Performance note
        if torch.cuda.is_available():
            st.success("🚀 **GPU acceleration active** for faster processing")
        else:
            st.warning("💻 **CPU mode** - Consider GPU for faster processing")


if __name__ == "__main__":
    import asyncio
    main()
