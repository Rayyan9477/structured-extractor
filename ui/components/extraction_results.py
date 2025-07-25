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
    
    if not results.patients:
        st.warning("No patient data found in the document.")
        return
    
    if len(results.patients) > 1:
        patient_tabs = st.tabs([f"Patient {p.patient_index or i+1}" for i, p in enumerate(results.patients)])
        for i, (tab, patient) in enumerate(zip(patient_tabs, results.patients)):
            with tab:
                _render_patient_data(patient)
    else:
        _render_patient_data(results.patients[0])
    
    st.markdown("</div>", unsafe_allow_html=True)


def _render_patient_data(patient: PatientData):
    """Render a single patient's extracted data."""
    
    # Patient Demographics
    st.markdown("<h5>Patient Demographics</h5>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("First Name", patient.first_name or "N/A")
    cols[1].metric("Last Name", patient.last_name or "N/A")
    cols[2].metric("Date of Birth", patient.date_of_birth or "N/A")

    # Identifiers
    st.markdown("<h5>Identifiers</h5>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Patient ID", patient.patient_id or "N/A")
    cols[1].metric("Phone", patient.phone or "N/A")
    
    # CPT Codes
    if patient.cpt_codes:
        st.markdown("<h5>CPT Codes</h5>", unsafe_allow_html=True)
        cpt_data = [{
            "Code": c.code, 
            "Description": c.description or "N/A", 
            "Confidence": f"{c.confidence:.2%}" if isinstance(c.confidence, (float, int)) else str(c.confidence or 'N/A')
        } for c in patient.cpt_codes]
        st.dataframe(cpt_data, use_container_width=True, hide_index=True)
    
    # ICD-10 Codes
    if patient.icd10_codes:
        st.markdown("<h5>ICD-10 Codes</h5>", unsafe_allow_html=True)
        icd_data = [{
            "Code": i.code, 
            "Description": i.description or "N/A", 
            "Confidence": f"{i.confidence:.2%}" if isinstance(i.confidence, (float, int)) else str(i.confidence or 'N/A')
        } for i in patient.icd10_codes]
        st.dataframe(icd_data, use_container_width=True, hide_index=True)
        
    # Financial Info
    if patient.financial_info:
        st.markdown("<h5>Financial Information</h5>", unsafe_allow_html=True)
        fin = patient.financial_info
        st.metric("Total Charges", f"${fin.total_charges:,.2f}" if fin.total_charges else "N/A")

    with st.expander("View Raw JSON Data"):
        try:
            if hasattr(patient, 'model_dump'):
                st.json(patient.model_dump())
            else:
                # Fallback for basic patient data
                import json
                patient_dict = {
                    'first_name': patient.first_name,
                    'last_name': patient.last_name,
                    'date_of_birth': str(patient.date_of_birth) if patient.date_of_birth else None,
                    'patient_id': patient.patient_id,
                    'phone': patient.phone,
                    'cpt_codes': [{'code': c.code, 'description': c.description} for c in (patient.cpt_codes or [])],
                    'icd10_codes': [{'code': i.code, 'description': i.description} for i in (patient.icd10_codes or [])]
                }
                st.json(patient_dict)
        except Exception as e:
            st.error(f"Error displaying JSON data: {e}")
            st.text(str(patient))
