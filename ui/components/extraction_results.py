"""
Extraction Results Component for Medical Superbill Extractor UI
"""
import streamlit as st
import pandas as pd
from datetime import datetime


def render_extraction_results(results):
    """
    Render the extraction results in a well-formatted display.
    
    Args:
        results: The extraction results dictionary/object
    """
    st.markdown("""
    <div class="results-container">
        <h2 style="color: var(--primary-color); margin-bottom: 1.5rem;">Extraction Results</h2>
    """, unsafe_allow_html=True)
    
    # Check if extraction was successful
    if not results.get("success", False):
        st.error("Extraction failed. Please check the document and try again.")
        if "error" in results:
            st.error(f"Error: {results['error']}")
        return
    
    # Patients data
    patients = results.get("patients", [])
    
    if not patients:
        st.warning("No patient data found in the document.")
        return
    
    # Tabs for multiple patients if needed
    if len(patients) > 1:
        patient_tabs = st.tabs([f"Patient {i+1}" for i in range(len(patients))])
        
        for i, (tab, patient) in enumerate(zip(patient_tabs, patients)):
            with tab:
                _render_patient_data(patient, i+1)
    else:
        _render_patient_data(patients[0])
    
    st.markdown("</div>", unsafe_allow_html=True)


def _render_patient_data(patient, patient_num=None):
    """
    Render a single patient's extracted data.
    
    Args:
        patient: The patient data dictionary
        patient_num: Optional patient number for multi-patient documents
    """
    # Patient info section
    st.markdown("""
    <div class="result-section">
        <div class="result-section-title">Patient Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if patient.get("patient_name"):
            st.markdown(f"**Name:** {patient['patient_name']}")
        if patient.get("patient_id"):
            st.markdown(f"**ID:** {patient['patient_id']}")
        if patient.get("patient_dob"):
            st.markdown(f"**DOB:** {patient['patient_dob']}")
    
    with col2:
        if patient.get("patient_address"):
            st.markdown(f"**Address:** {patient['patient_address']}")
        if patient.get("patient_phone"):
            st.markdown(f"**Phone:** {patient['patient_phone']}")
        if patient.get("patient_email"):
            st.markdown(f"**Email:** {patient['patient_email']}")
    
    # Service info section
    st.markdown("""
    <div class="result-section">
        <div class="result-section-title">Service Information</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if patient.get("date_of_service"):
            st.markdown(f"**Date of Service:** {patient['date_of_service']}")
        if patient.get("claim_date"):
            st.markdown(f"**Claim Date:** {patient['claim_date']}")
    
    with col2:
        if patient.get("provider_info") and isinstance(patient["provider_info"], dict):
            provider = patient["provider_info"]
            if provider.get("name"):
                st.markdown(f"**Provider:** {provider['name']}")
            if provider.get("npi"):
                st.markdown(f"**NPI:** {provider['npi']}")
    
    # CPT Codes
    if patient.get("cpt_codes") and len(patient["cpt_codes"]) > 0:
        st.markdown("""
        <div class="result-section">
            <div class="result-section-title">CPT Codes</div>
        </div>
        """, unsafe_allow_html=True)
        
        cpt_data = []
        for cpt in patient["cpt_codes"]:
            if isinstance(cpt, dict):
                cpt_data.append({
                    "Code": cpt.get("code", ""),
                    "Description": cpt.get("description", ""),
                    "Amount": cpt.get("amount", "")
                })
        
        if cpt_data:
            df = pd.DataFrame(cpt_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Diagnosis Codes
    if patient.get("diagnosis_codes") and len(patient["diagnosis_codes"]) > 0:
        st.markdown("""
        <div class="result-section">
            <div class="result-section-title">Diagnosis Codes (ICD-10)</div>
        </div>
        """, unsafe_allow_html=True)
        
        icd_data = []
        for icd in patient["diagnosis_codes"]:
            if isinstance(icd, dict):
                icd_data.append({
                    "Code": icd.get("code", ""),
                    "Description": icd.get("description", "")
                })
        
        if icd_data:
            df = pd.DataFrame(icd_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Financial Information
    if any(key in patient for key in ["charges", "copay", "deductible", "insurance_info"]):
        st.markdown("""
        <div class="result-section">
            <div class="result-section-title">Financial Information</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if patient.get("charges"):
                st.markdown(f"**Total Charges:** ${patient['charges']}")
            if patient.get("copay"):
                st.markdown(f"**Copay:** ${patient['copay']}")
            if patient.get("deductible"):
                st.markdown(f"**Deductible:** ${patient['deductible']}")
        
        with col2:
            if patient.get("insurance_info") and isinstance(patient["insurance_info"], dict):
                insurance = patient["insurance_info"]
                if insurance.get("company"):
                    st.markdown(f"**Insurance:** {insurance['company']}")
                if insurance.get("policy_number"):
                    st.markdown(f"**Policy #:** {insurance['policy_number']}")
    
    # Confidence scores if available
    if patient.get("confidence_scores") and isinstance(patient["confidence_scores"], dict):
        with st.expander("Extraction Confidence Scores"):
            for field, score in patient["confidence_scores"].items():
                st.progress(float(score), text=f"{field}: {score:.2f}")
    
    # Show JSON data
    with st.expander("Raw Extracted Data (JSON)"):
        st.json(patient)
