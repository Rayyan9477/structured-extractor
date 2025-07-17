"""
Validation Panel Component for Medical Superbill Extractor UI
"""
import streamlit as st
import pandas as pd


def render_validation_panel(results):
    """
    Render validation panel for extracted data.
    
    Args:
        results: The extraction results to validate
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Validation Results</div>
    """, unsafe_allow_html=True)
    
    # Validation overview
    patients = results.get("patients", [])
    
    if not patients:
        st.warning("No patient data to validate.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Simulate validation results
    validation_results = _simulate_validation(patients)
    
    # Display validation summary
    if validation_results["total_issues"] == 0:
        st.success("All data passed validation checks!")
    else:
        st.warning(f"{validation_results['total_issues']} validation issues found")
    
    # Display validation metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "CPT Code Validation",
            f"{validation_results['cpt_valid']}/{validation_results['cpt_total']}",
            delta="Valid",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "ICD-10 Code Validation",
            f"{validation_results['icd_valid']}/{validation_results['icd_total']}",
            delta="Valid",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "Date Validation",
            f"{validation_results['date_valid']}/{validation_results['date_total']}",
            delta="Valid",
            delta_color="normal"
        )
    
    # Display validation issues if any
    if validation_results["issues"]:
        st.markdown("### Issues Found")
        
        for issue in validation_results["issues"]:
            alert_type = issue["severity"]
            message = issue["message"]
            
            if alert_type == "warning":
                st.warning(message)
            elif alert_type == "error":
                st.error(message)
            else:
                st.info(message)
                
        # Show suggested fixes
        st.markdown("### Suggested Fixes")
        
        fix_data = []
        for issue in validation_results["issues"]:
            if "fix" in issue:
                fix_data.append({
                    "Field": issue["field"],
                    "Current Value": issue["value"],
                    "Suggested Value": issue["fix"],
                    "Confidence": issue.get("confidence", "Medium")
                })
        
        if fix_data:
            fix_df = pd.DataFrame(fix_data)
            
            # Display fixes with checkboxes
            edited_df = st.data_editor(
                fix_df,
                column_config={
                    "Apply": st.column_config.CheckboxColumn(
                        "Apply",
                        help="Select to apply this fix",
                        default=True
                    )
                },
                disabled=["Field", "Current Value", "Suggested Value", "Confidence"],
                hide_index=True,
                use_container_width=True
            )
            
            # Button to apply selected fixes
            if st.button("Apply Selected Fixes"):
                st.success("Fixes applied successfully!")
                
                # TODO: Implement actual fix application
                # This would update the results and re-validate
        else:
            st.info("No automatic fixes available for these issues.")
    
    # Validation options
    with st.expander("Validation Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Validate CPT Codes", value=True)
            st.checkbox("Validate ICD-10 Codes", value=True)
            st.checkbox("Validate Dates", value=True)
            
        with col2:
            st.checkbox("Validate Patient Info", value=True)
            st.checkbox("Validate Provider Info", value=True)
            st.select_slider(
                "Validation Strictness",
                options=["Lenient", "Standard", "Strict"],
                value="Standard"
            )
    
    # Manual edit button
    st.button("Edit Extraction Results Manually", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def _simulate_validation(patients):
    """Simulate validation results for demonstration."""
    cpt_total = sum(len(patient.get("cpt_codes", [])) for patient in patients)
    icd_total = sum(len(patient.get("diagnosis_codes", [])) for patient in patients)
    date_total = len(patients) * 2  # Assuming date_of_service and claim_date per patient
    
    # Initialize with mostly valid
    cpt_valid = cpt_total
    icd_valid = icd_total
    date_valid = date_total
    
    # Create some sample issues
    issues = []
    
    # Add a sample CPT code issue
    if cpt_total > 0:
        cpt_valid -= 1
        issues.append({
            "severity": "warning",
            "field": "CPT Code",
            "value": "9921X",
            "message": "Invalid CPT code '9921X' - not a valid 5-digit format",
            "fix": "99213",
            "confidence": "High"
        })
    
    # Add a sample ICD-10 code issue
    if icd_total > 0:
        icd_valid -= 1
        issues.append({
            "severity": "error",
            "field": "ICD-10 Code",
            "value": "E.11.9",
            "message": "Invalid ICD-10 format 'E.11.9' - should be E11.9",
            "fix": "E11.9",
            "confidence": "High"
        })
    
    # Add a sample date issue
    if date_total > 0:
        date_valid -= 1
        issues.append({
            "severity": "warning",
            "field": "Date of Service",
            "value": "2025-13-01",
            "message": "Invalid date '2025-13-01' - month value out of range",
            "fix": "2025-12-01",
            "confidence": "Medium"
        })
    
    return {
        "cpt_total": cpt_total,
        "cpt_valid": cpt_valid,
        "icd_total": icd_total,
        "icd_valid": icd_valid,
        "date_total": date_total,
        "date_valid": date_valid,
        "total_issues": len(issues),
        "issues": issues
    }
