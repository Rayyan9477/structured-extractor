"""
Extraction Results Component for Medical Superbill Extractor UI
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from src.core.data_schema import ExtractionResults, PatientData, CPTCode, ICD10Code


def render_extraction_results(results: ExtractionResults):
    """
    Render the extraction results in a well-formatted display.
    
    Args:
        results: The extraction results object
    """
    st.markdown("""
    <div class="results-container">
        <h2 style="color: var(--primary-color); margin-bottom: 1.5rem;">Extraction Results</h2>
    """, unsafe_allow_html=True)
    
    # Display document metadata
    if results.metadata:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pages", results.metadata.page_count)
        with col2:
            st.metric("Processing Time", f"{results.metadata.processing_time:.2f}s")
        with col3:
            st.metric("Overall Confidence", f"{results.overall_confidence:.1%}")
        st.divider()
    
    if not results.patients:
        st.warning("No patient data found in the document.")
        return
    
    # Display total patient count and multi-page information
    col1, col2 = st.columns(2) 
    with col1:
        st.metric("Total Patients", len(results.patients))
    with col2:
        multi_page_patients = sum(1 for p in results.patients if p.spans_multiple_pages)
        if multi_page_patients > 0:
            st.metric("Multi-page Patients", multi_page_patients)
    
    # Show patients in tabs if multiple, otherwise show single patient
    if len(results.patients) > 1:
        patient_tabs = st.tabs([f"Patient {p.patient_index + 1 if hasattr(p, 'patient_index') else i+1}" for i, p in enumerate(results.patients)])
        for i, (tab, patient) in enumerate(zip(patient_tabs, results.patients)):
            with tab:
                _render_patient_data(patient, i)
    else:
        _render_patient_data(results.patients[0], 0)
    
    # Add export button for individual patient
    if st.button(f"Export Patient {patient_idx + 1} Data", key=f"export_patient_{patient_idx}"):
        # Create downloadable JSON
        try:
            if hasattr(patient, 'to_dict'):
                data = patient.to_dict()
            else:
                data = {
                    'first_name': patient.first_name,
                    'last_name': patient.last_name,
                    'date_of_birth': str(patient.date_of_birth) if patient.date_of_birth else None,
                    'patient_id': patient.patient_id,
                    'cpt_codes': [{'code': c.code, 'description': c.description} for c in (patient.cpt_codes or [])],
                    'icd10_codes': [{'code': i.code, 'description': i.description} for i in (patient.icd10_codes or [])]
                }
            
            import json
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"patient_{patient_idx + 1}_data.json",
                mime="application/json"
            )
        except Exception as e:
            st.error(f"Export failed: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)


def _render_patient_data(patient: PatientData, patient_idx: int):
    """Render a single patient's extracted data."""
    
    # Patient Demographics
    st.markdown("<h5>Patient Demographics</h5>", unsafe_allow_html=True)
    
    # Show page information if available
    if hasattr(patient, 'page_number') and patient.page_number:
        if patient.spans_multiple_pages:
            st.info(f"📄 Patient spans multiple pages: {', '.join(map(str, patient.page_numbers))}")
        else:
            st.info(f"📄 Found on page {patient.page_number} of {patient.total_pages}")
    
    cols = st.columns(4)
    cols[0].metric("First Name", patient.first_name or "N/A")
    cols[1].metric("Last Name", patient.last_name or "N/A")
    cols[2].metric("Date of Birth", patient.date_of_birth or "N/A")
    cols[3].metric("Extraction Confidence", f"{patient.extraction_confidence:.1%}" if patient.extraction_confidence else "N/A")

    # Contact Information
    st.markdown("<h5>Contact & Insurance</h5>", unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Patient ID", patient.patient_id or "N/A")
    cols[1].metric("Phone", patient.phone or "N/A")
    cols[2].metric("Insurance Provider", patient.insurance_provider or "N/A")
    cols[3].metric("Insurance ID", patient.insurance_id or "N/A")
    
    if patient.address:
        st.text_area("Address", patient.address, height=60)
    
    # CPT Codes
    if patient.cpt_codes:
        st.markdown("<h5>CPT Codes</h5>", unsafe_allow_html=True)
        cpt_data = [{
            "Code": c.code, 
            "Description": c.description or "N/A", 
            "Charge": f"${c.charge:.2f}" if c.charge else "N/A",
            "Units": str(c.units) if c.units else "1",
            "Confidence": f"{c.confidence:.2%}" if isinstance(c.confidence, (float, int)) else str(c.confidence or 'N/A')
        } for c in patient.cpt_codes]
        st.dataframe(cpt_data, use_container_width=True, hide_index=True)
        
        # Show total charges if available
        total_charges = sum(c.charge or 0 for c in patient.cpt_codes if c.charge)
        if total_charges > 0:
            st.metric("Total CPT Charges", f"${total_charges:.2f}")
    
    # ICD-10 Codes
    if patient.icd10_codes:
        st.markdown("<h5>ICD-10 Codes</h5>", unsafe_allow_html=True)
        icd_data = [{
            "Code": i.code, 
            "Description": i.description or "N/A", 
            "Confidence": f"{i.confidence:.2%}" if isinstance(i.confidence, (float, int)) else str(i.confidence or 'N/A')
        } for i in patient.icd10_codes]
        st.dataframe(icd_data, use_container_width=True, hide_index=True)
        
    # Validation Errors
    if hasattr(patient, 'validation_errors') and patient.validation_errors:
        st.markdown("<h5>⚠️ Validation Issues</h5>", unsafe_allow_html=True)
        for error in patient.validation_errors:
            st.warning(error)
    
    # Field Confidences
    if hasattr(patient, 'confidences') and patient.confidences:
        with st.expander("Field Confidence Scores"):
            for field, conf in patient.confidences.items():
                st.metric(f"{field.replace('_', ' ').title()}", f"{conf.confidence:.1%}" if hasattr(conf, 'confidence') else str(conf))

    with st.expander("View Raw JSON Data"):
        try:
            if hasattr(patient, 'to_dict'):
                st.json(patient.to_dict())
            elif hasattr(patient, 'model_dump'):
                st.json(patient.model_dump())
            else:
                # Fallback for basic patient data
                import json
                patient_dict = {
                    'first_name': patient.first_name,
                    'last_name': patient.last_name,
                    'middle_name': patient.middle_name,
                    'date_of_birth': str(patient.date_of_birth) if patient.date_of_birth else None,
                    'patient_id': patient.patient_id,
                    'phone': patient.phone,
                    'email': patient.email,
                    'address': patient.address,
                    'insurance_provider': patient.insurance_provider,
                    'insurance_id': patient.insurance_id,
                    'extraction_confidence': patient.extraction_confidence,
                    'page_number': getattr(patient, 'page_number', None),
                    'total_pages': getattr(patient, 'total_pages', None),
                    'spans_multiple_pages': getattr(patient, 'spans_multiple_pages', False),
                    'cpt_codes': [{'code': c.code, 'description': c.description, 'charge': c.charge, 'units': c.units} for c in (patient.cpt_codes or [])],
                    'icd10_codes': [{'code': i.code, 'description': i.description} for i in (patient.icd10_codes or [])]
                }
                st.json(patient_dict)
        except Exception as e:
            st.error(f"Error displaying JSON data: {e}")
            st.text(str(patient))
