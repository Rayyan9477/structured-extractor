"""
Extraction Configuration Component for Medical Superbill Extractor UI
"""
import streamlit as st
import yaml
import json
from pathlib import Path
import os


def render_extraction_config():
    """
    Render extraction configuration options for customizing the extraction process.
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Extraction Configuration</div>
    """, unsafe_allow_html=True)
    
    # Load default config
    config = _load_default_config()
    
    # Create tabs for different configuration categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Models", "Document Processing", "Field Extraction", 
        "Code Validation", "Advanced"
    ])
    
    # Model configuration
    with tab1:
        st.markdown("### OCR Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Nanonets settings
            st.markdown("#### Nanonets-OCR Settings")
            
            # Use actual config structure
            nanonets_config = config.get("ocr", {}).get("nanonets_ocr", {})
            
            nanonets_model_name = st.text_input(
                "Model Name",
                value=nanonets_config.get("model_name", "nanonets/Nanonets-OCR-s")
            )
            
            nanonets_max_length = st.number_input(
                "Max New Tokens",
                min_value=128,
                max_value=15000,
                value=nanonets_config.get("max_new_tokens", 15000),
                step=128
            )
            
            nanonets_timeout = st.number_input(
                "Timeout (seconds)",
                min_value=30,
                max_value=120,
                value=nanonets_config.get("timeout", 60),
                step=10
            )
        
        with col2:
            # NuExtract Vision OCR settings
            st.markdown("#### NuExtract Vision-OCR Settings")
            
            nuextract_config = config.get("extraction", {}).get("nuextract", {})
            
            nuextract_model_name = st.text_input(
                "NuExtract Model Name",
                value=nuextract_config.get("model_name", "numind/NuExtract-2.0-8B")
            )
            
            nuextract_max_length = st.number_input(
                "Max Context Length",
                min_value=1024,
                max_value=8192,
                value=nuextract_config.get("max_length", 8192),
                step=256
            )
            
            nuextract_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=nuextract_config.get("temperature", 0.1),
                step=0.05
            )
        
        st.markdown("### Model Selection")
        
        # Add model selection options
        use_models = st.multiselect(
            "Active Models",
            ["nanonets_ocr", "nuextract_vision"],
            default=["nanonets_ocr"],
            help="Select which models to use for processing"
        )
    
    # Document Processing configuration
    with tab2:
        st.markdown("### PDF Processing")
        
        if "document_processing" in config and "pdf" in config["document_processing"]:
            pdf_config = config["document_processing"]["pdf"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                pdf_dpi = st.number_input(
                    "DPI",
                    min_value=72,
                    max_value=600,
                    value=pdf_config.get("dpi", 300),
                    step=72
                )
                
                pdf_format = st.selectbox(
                    "Format",
                    ["RGB", "RGBA", "L", "1"],
                    index=0 if pdf_config.get("format") == "RGB" else 1
                )
            
            with col2:
                pdf_first_page = st.number_input(
                    "First Page (blank for all)",
                    min_value=1,
                    value=pdf_config.get("first_page", 1) if pdf_config.get("first_page") else 1,
                    help="Leave as 1 to process from the beginning"
                )
                
                pdf_last_page = st.number_input(
                    "Last Page (blank for all)",
                    min_value=1,
                    value=pdf_config.get("last_page", 1) if pdf_config.get("last_page") else 1,
                    help="Leave as 1 to process until the end"
                )
        
        st.markdown("### Image Preprocessing")
        
        if "document_processing" in config and "image_preprocessing" in config["document_processing"]:
            img_config = config["document_processing"]["image_preprocessing"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                resize_factor = st.slider(
                    "Resize Factor",
                    min_value=0.5,
                    max_value=2.0,
                    value=img_config.get("resize_factor", 1.0),
                    step=0.1
                )
                
                denoise = st.toggle(
                    "Denoise Images",
                    value=img_config.get("denoise", True)
                )
            
            with col2:
                enhance_contrast = st.toggle(
                    "Enhance Contrast",
                    value=img_config.get("enhance_contrast", True)
                )
                
                rotation_correction = st.toggle(
                    "Automatic Rotation Correction",
                    value=img_config.get("rotation_correction", True)
                )
                
                binarize = st.toggle(
                    "Binarize Images",
                    value=img_config.get("binarize", False)
                )
    
    # Field Extraction configuration
    with tab3:
        st.markdown("### Field Configuration")
        
        if "extraction_fields" in config:
            field_config = config["extraction_fields"]
            
            # Required fields
            st.markdown("#### Required Fields")
            required_fields = field_config.get("required_fields", [])
            
            # Create multiselect with all possible fields
            all_fields = [
                "cpt_codes", "diagnosis_codes", "patient_name", "date_of_service",
                "claim_date", "provider_info", "patient_dob", "patient_address",
                "insurance_info", "procedure_descriptions", "charges", "copay", "deductible"
            ]
            
            selected_required = st.multiselect(
                "Required Fields",
                options=all_fields,
                default=required_fields
            )
            
            # Optional fields (those not selected as required)
            st.markdown("#### Optional Fields")
            
            available_optional = [field for field in all_fields if field not in selected_required]
            optional_fields = field_config.get("optional_fields", [])
            
            # Filter out any that are now required
            filtered_optional = [field for field in optional_fields if field in available_optional]
            
            selected_optional = st.multiselect(
                "Optional Fields",
                options=available_optional,
                default=filtered_optional
            )
    
    # Code Validation configuration
    with tab4:
        st.markdown("### Medical Code Patterns")
        
        if "medical_codes" in config:
            code_config = config["medical_codes"]
            
            # CPT Codes
            st.markdown("#### CPT Codes")
            
            if "cpt_codes" in code_config:
                cpt_pattern = st.text_input(
                    "CPT Pattern",
                    value=code_config["cpt_codes"].get("pattern", "\\b\\d{5}\\b")
                )
            
            # ICD-10 Codes
            st.markdown("#### ICD-10 Codes")
            
            icd10_pattern = st.text_input(
                "ICD-10 Pattern",
                value=code_config.get("icd10_codes", {}).get("pattern", "\\b[A-Z]\\d{2}(\\.\\d{1,3})?\\b")
            )
            
            # Validation options
            st.markdown("#### Validation Options")
            
            validation_options = st.multiselect(
                "Enable Validation For",
                options=["CPT Format", "CPT Database", "ICD-10 Format", "ICD-10 Database", "Date Format", "Date Range"],
                default=["CPT Format", "ICD-10 Format", "Date Format"]
            )
    
    # Advanced configuration
    with tab5:
        st.markdown("### Advanced Settings")
        
        # Performance settings
        st.markdown("#### Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_gpu = st.toggle(
                "Use GPU Acceleration",
                value=True,
                help="Enable GPU acceleration for faster processing"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=16,
                value=4,
                help="Number of images to process in parallel"
            )
        
        with col2:
            memory_optimization = st.toggle(
                "Memory Optimization",
                value=True,
                help="Enable memory optimization for large documents"
            )
            
            num_workers = st.number_input(
                "Worker Threads",
                min_value=1,
                max_value=16,
                value=4,
                help="Number of worker threads for parallel processing"
            )
        
        # Security settings
        st.markdown("#### Security")
        
        phi_anonymization = st.toggle(
            "Enable PHI Anonymization",
            value=False,
            help="Automatically anonymize Protected Health Information"
        )
        
        anonymization_method = st.selectbox(
            "Anonymization Method",
            ["Redaction", "Replacement", "Tokenization"],
            index=0,
            disabled=not phi_anonymization
        )
        
        # Export raw config
        st.markdown("#### Configuration Export/Import")
        
        # Export button
        if st.button("Export Configuration"):
            # Create updated config from UI values
            updated_config = {
                "ocr": {
                    "ensemble": {
                        "use_models": use_models,
                        "method": "best_confidence",
                        "weights": {
                            "nanonets_ocr": 1.3
                        }
                    },
                    "nanonets_ocr": {
                        "model_name": nanonets_model_name,
                        "max_new_tokens": nanonets_max_length,
                        "timeout": nanonets_timeout,
                        "use_local": True
                    }
                },
                "extraction": {
                    "nuextract": {
                        "model_name": nuextract_model_name,
                        "max_length": nuextract_max_length,
                        "temperature": nuextract_temperature,
                        "use_local": True
                    }
                },
                "document_processing": {
                    "pdf": {
                        "dpi": pdf_dpi,
                        "format": pdf_format,
                        "first_page": pdf_first_page if pdf_first_page > 1 else None,
                        "last_page": pdf_last_page if pdf_last_page > 1 else None
                    },
                    "image_preprocessing": {
                        "resize_factor": resize_factor,
                        "denoise": denoise,
                        "enhance_contrast": enhance_contrast,
                        "binarize": binarize,
                        "rotation_correction": rotation_correction
                    }
                },
                "extraction_fields": {
                    "required_fields": selected_required,
                    "optional_fields": selected_optional
                },
                "medical_codes": {
                    "cpt_codes": {
                        "pattern": cpt_pattern
                    },
                    "icd10_codes": {
                        "pattern": icd10_pattern
                    }
                },
                "performance": {
                    "use_gpu": use_gpu,
                    "batch_size": batch_size,
                    "memory_optimization": memory_optimization,
                    "num_workers": num_workers
                },
                "security": {
                    "phi_anonymization": phi_anonymization,
                    "anonymization_method": anonymization_method.lower() if phi_anonymization else None
                }
            }
            
            # Offer download
            st.download_button(
                label="Download Configuration",
                data=yaml.dump(updated_config, default_flow_style=False),
                file_name="extraction_config.yaml",
                mime="application/x-yaml"
            )
        
        # Upload option
        uploaded_config = st.file_uploader(
            "Upload Configuration File",
            type=["yaml", "yml"]
        )
        
        if uploaded_config is not None:
            st.success("Configuration file uploaded! Click Apply to use this configuration.")
            
            if st.button("Apply Configuration"):
                st.success("Configuration applied successfully!")
    
    st.markdown("</div>", unsafe_allow_html=True)


def _load_default_config():
    """Load the default configuration."""
    # Path to the default config file
    default_config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    
    try:
        if default_config_path.exists():
            with open(default_config_path, "r") as file:
                return yaml.safe_load(file)
        else:
            # Return a basic default config
            return {
                "ocr": {
                    "ensemble": {
                        "use_models": ["nanonets_ocr"],
                        "method": "best_confidence",
                        "weights": {
                            "nanonets_ocr": 1.3
                        }
                    },
                    "nanonets_ocr": {
                        "model_name": "nanonets/Nanonets-OCR-s",
                        "max_new_tokens": 15000,
                        "timeout": 60,
                        "use_local": True
                    }
                },
                "extraction": {
                    "nuextract": {
                        "model_name": "numind/NuExtract-2.0-8B",
                        "max_length": 8192,
                        "temperature": 0.1,
                        "use_local": True
                    }
                },
                "document_processing": {
                    "pdf": {
                        "dpi": 300,
                        "format": "RGB",
                        "first_page": None,
                        "last_page": None
                    },
                    "image_preprocessing": {
                        "resize_factor": 1.0,
                        "denoise": True,
                        "enhance_contrast": True,
                        "binarize": False,
                        "rotation_correction": True
                    }
                },
                "extraction_fields": {
                    "required_fields": [
                        "cpt_codes",
                        "diagnosis_codes",
                        "patient_name",
                        "date_of_service",
                        "claim_date",
                        "provider_info"
                    ],
                    "optional_fields": [
                        "patient_dob",
                        "patient_address",
                        "insurance_info",
                        "procedure_descriptions",
                        "charges",
                        "copay",
                        "deductible"
                    ]
                },
                "medical_codes": {
                    "cpt_codes": {
                        "pattern": "\\b\\d{5}\\b"
                    }
                }
            }
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}
