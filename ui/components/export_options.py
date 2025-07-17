"""
Export Options Component for Medical Superbill Extractor UI
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime


def render_export_options(results):
    """
    Render export options for extraction results.
    
    Args:
        results: The extraction results to export
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Export Options</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "Excel"],
            index=0
        )
    
    with col2:
        if export_format == "CSV" or export_format == "Excel":
            flatten = st.checkbox("Flatten Nested Data", value=True)
        else:
            include_metadata = st.checkbox("Include Metadata", value=True)
    
    # Additional options based on format
    if export_format == "Excel":
        include_summary = st.checkbox("Include Summary Sheet", value=True)
        include_charts = st.checkbox("Include Charts & Visualizations", value=True)
    
    # Convert results to appropriate format
    if export_format == "JSON":
        # Get the data to export
        export_data = _prepare_json_data(results, include_metadata)
        
        # Create a download button
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"superbill_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
    elif export_format == "CSV":
        # Get the data to export
        export_data = _prepare_csv_data(results, flatten)
        
        # Create a download button
        st.download_button(
            label="Download CSV",
            data=export_data.to_csv(index=False),
            file_name=f"superbill_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    elif export_format == "Excel":
        # Create a download button
        st.download_button(
            label="Download Excel",
            data=_prepare_excel_data(results, flatten, include_summary, include_charts),
            file_name=f"superbill_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


def _prepare_json_data(results, include_metadata=True):
    """Prepare results data for JSON export."""
    if include_metadata:
        # Include full data with metadata
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "extraction_status": "success" if results.get("success", False) else "failed",
            "patients": results.get("patients", []),
            "metadata": {
                "version": "1.0.0",
                "extraction_time": results.get("processing_time", 0)
            }
        }
    else:
        # Just include the patient data
        export_data = {
            "patients": results.get("patients", [])
        }
    
    return export_data


def _prepare_csv_data(results, flatten=True):
    """Prepare results data for CSV export."""
    patients = results.get("patients", [])
    
    if not patients:
        return pd.DataFrame()
    
    if flatten:
        # Create a flattened structure
        flattened_data = []
        
        for patient in patients:
            patient_base = {
                "patient_name": patient.get("patient_name", ""),
                "patient_id": patient.get("patient_id", ""),
                "patient_dob": patient.get("patient_dob", ""),
                "date_of_service": patient.get("date_of_service", ""),
                "claim_date": patient.get("claim_date", "")
            }
            
            # Provider info
            if patient.get("provider_info") and isinstance(patient["provider_info"], dict):
                provider = patient["provider_info"]
                patient_base["provider_name"] = provider.get("name", "")
                patient_base["provider_npi"] = provider.get("npi", "")
            
            # CPT codes
            if patient.get("cpt_codes") and len(patient["cpt_codes"]) > 0:
                for i, cpt in enumerate(patient["cpt_codes"]):
                    if isinstance(cpt, dict):
                        row = patient_base.copy()
                        row["cpt_code"] = cpt.get("code", "")
                        row["cpt_description"] = cpt.get("description", "")
                        row["cpt_amount"] = cpt.get("amount", "")
                        
                        # Add diagnosis codes
                        if patient.get("diagnosis_codes") and len(patient["diagnosis_codes"]) > 0:
                            for j, icd in enumerate(patient["diagnosis_codes"]):
                                if j < 4 and isinstance(icd, dict):  # Limit to 4 diagnosis codes per row
                                    row[f"diagnosis_code_{j+1}"] = icd.get("code", "")
                                    row[f"diagnosis_desc_{j+1}"] = icd.get("description", "")
                        
                        flattened_data.append(row)
            else:
                # No CPT codes, just add patient with diagnosis
                row = patient_base.copy()
                
                # Add diagnosis codes
                if patient.get("diagnosis_codes") and len(patient["diagnosis_codes"]) > 0:
                    for j, icd in enumerate(patient["diagnosis_codes"]):
                        if j < 4 and isinstance(icd, dict):  # Limit to 4 diagnosis codes per row
                            row[f"diagnosis_code_{j+1}"] = icd.get("code", "")
                            row[f"diagnosis_desc_{j+1}"] = icd.get("description", "")
                
                flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)
    else:
        # Create a simpler structure
        patient_data = []
        
        for patient in patients:
            patient_row = {
                "patient_name": patient.get("patient_name", ""),
                "patient_id": patient.get("patient_id", ""),
                "patient_dob": patient.get("patient_dob", ""),
                "date_of_service": patient.get("date_of_service", ""),
                "claim_date": patient.get("claim_date", ""),
                "cpt_codes": ", ".join([cpt.get("code", "") for cpt in patient.get("cpt_codes", []) if isinstance(cpt, dict)]),
                "diagnosis_codes": ", ".join([icd.get("code", "") for icd in patient.get("diagnosis_codes", []) if isinstance(icd, dict)])
            }
            
            patient_data.append(patient_row)
        
        return pd.DataFrame(patient_data)


def _prepare_excel_data(results, flatten=True, include_summary=True, include_charts=True):
    """Prepare results data for Excel export."""
    import io
    from io import BytesIO
    import pandas as pd
    import xlsxwriter
    
    buffer = BytesIO()
    
    # Get the patient data
    patient_df = _prepare_csv_data(results, flatten)
    
    # Create an Excel writer object
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write the main data
        patient_df.to_excel(writer, sheet_name='Patient Data', index=False)
        
        # Get the workbook and add formatting
        workbook = writer.book
        worksheet = writer.sheets['Patient Data']
        
        # Add formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#6236FF',
            'color': 'white',
            'border': 1,
            'align': 'center'
        })
        
        # Apply formatting to header
        for col_num, value in enumerate(patient_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, max(len(value) + 2, 12))
        
        # Add a summary sheet if requested
        if include_summary and not patient_df.empty:
            patients = results.get("patients", [])
            
            # Create summary data
            summary_data = {
                "Total Patients": [len(patients)],
                "Total CPT Codes": [sum(len(patient.get("cpt_codes", [])) for patient in patients)],
                "Total Diagnosis Codes": [sum(len(patient.get("diagnosis_codes", [])) for patient in patients)],
                "Extraction Date": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            
            # Create summary dataframe
            summary_df = pd.DataFrame(summary_data)
            
            # Write summary to a new sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format summary sheet
            summary_sheet = writer.sheets['Summary']
            
            for col_num, value in enumerate(summary_df.columns.values):
                summary_sheet.write(0, col_num, value, header_format)
                summary_sheet.set_column(col_num, col_num, max(len(value) + 2, 15))
        
        # Add charts if requested
        if include_charts and not patient_df.empty and 'cpt_codes' in patient_df.columns:
            chart_sheet = workbook.add_worksheet('Charts')
            
            # Create simple chart data (this is simulated)
            chart_data = {
                "CPT Code": ["99213", "85027", "99214", "99211", "Others"],
                "Count": [3, 2, 2, 1, 5]
            }
            
            chart_df = pd.DataFrame(chart_data)
            
            # Write chart data
            chart_df.to_excel(writer, sheet_name='Charts', index=False)
            
            # Add a chart
            chart = workbook.add_chart({'type': 'pie'})
            
            # Configure the chart
            chart.add_series({
                'name': 'CPT Code Distribution',
                'categories': ['Charts', 1, 0, 5, 0],
                'values': ['Charts', 1, 1, 5, 1],
            })
            
            # Insert the chart into the worksheet
            chart_sheet.insert_chart('D2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
    
    # Reset buffer position
    buffer.seek(0)
    
    return buffer.getvalue()
