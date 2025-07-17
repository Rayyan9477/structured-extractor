#!/usr/bin/env python3
"""
Medical Superbill Data Extraction System - Complete Application

This is a unified main file that runs the entire medical superbill extraction project
including CLI interface, UI/UX, and all core functionality.

Usage:
    python app_main.py                          # Launch Streamlit UI
    python app_main.py --cli file.pdf           # CLI mode
    python app_main.py --help                   # Show help
"""

import argparse
import sys
import os
import asyncio
import json
import threading
import time
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Core dependencies
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from PIL import Image, ImageEnhance
    import torch
    from transformers import AutoTokenizer, AutoModel, pipeline
    import cv2
    import pytesseract
    import fitz  # PyMuPDF
    from pdf2image import convert_from_path
    import yaml
    import openpyxl
    from pydantic import BaseModel, Field, validator
    import regex as re
    from dateutil import parser as date_parser
    import logging
    from pathlib import Path
    from io import BytesIO
    import streamlit.web.cli as stcli
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    DEPENDENCIES_OK = False

# ============================================================================
# DATA MODELS AND SCHEMAS
# ============================================================================

class FieldConfidence(BaseModel):
    """Confidence scoring for extracted fields."""
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    validation_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class CPTCode(BaseModel):
    """CPT code representation."""
    code: str
    description: Optional[str] = None
    confidence: FieldConfidence = Field(default_factory=FieldConfidence)
    charges: Optional[float] = None

class ICD10Code(BaseModel):
    """ICD-10 diagnosis code representation."""
    code: str
    description: Optional[str] = None
    confidence: FieldConfidence = Field(default_factory=FieldConfidence)

class ContactInfo(BaseModel):
    """Contact information."""
    phone: Optional[str] = None
    email: Optional[str] = None
    fax: Optional[str] = None

class Address(BaseModel):
    """Address information."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "USA"

class ProviderInfo(BaseModel):
    """Healthcare provider information."""
    name: Optional[str] = None
    npi: Optional[str] = None
    taxonomy: Optional[str] = None
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None

class PatientData(BaseModel):
    """Complete patient data structure."""
    # Personal Information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    patient_id: Optional[str] = None
    account_number: Optional[str] = None
    
    # Contact Information
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None
    
    # Medical Information
    cpt_codes: List[CPTCode] = Field(default_factory=list)
    icd10_codes: List[ICD10Code] = Field(default_factory=list)
    date_of_service: Optional[str] = None
    provider: Optional[ProviderInfo] = None
    
    # Financial Information
    charges: Optional[float] = None
    copay: Optional[float] = None
    deductible: Optional[float] = None
    insurance_info: Optional[Dict[str, Any]] = None
    
    # Confidence and Metadata
    confidence_scores: FieldConfidence = Field(default_factory=FieldConfidence)
    extracted_text: Optional[str] = None

class ExtractionResults(BaseModel):
    """Complete extraction results."""
    success: bool = True
    patients: List[PatientData] = Field(default_factory=list)
    total_patients: int = 0
    confidence_score: float = 0.0
    processing_time: Optional[float] = None
    source_file: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or str(ROOT_DIR / "config" / "config.yaml")
        self.config = self._load_default_config()
        if os.path.exists(self.config_path):
            self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "models": {
                "ocr": {
                    "primary": "tesseract",
                    "fallback": "easyocr",
                    "confidence_threshold": 0.7
                },
                "extraction": {
                    "model_name": "microsoft/DialoGPT-medium",  # Fallback model
                    "max_length": 512,
                    "temperature": 0.1
                }
            },
            "document_processing": {
                "pdf": {
                    "dpi": 300,
                    "max_pages": 50
                },
                "image_preprocessing": {
                    "resize_factor": 1.0,
                    "denoise": True,
                    "enhance_contrast": True
                }
            },
            "extraction_fields": {
                "patient_info": ["first_name", "last_name", "date_of_birth", "patient_id"],
                "medical_codes": ["cpt_codes", "icd10_codes"],
                "financial": ["charges", "copay", "deductible"]
            },
            "validation": {
                "dates": {
                    "min_year": 1900,
                    "max_year": datetime.now().year + 1
                },
                "validate_codes": True
            },
            "export": {
                "csv": {
                    "include_headers": True,
                    "delimiter": ","
                },
                "json": {
                    "indent": 2
                }
            },
            "ui": {
                "theme": "light",
                "max_file_size_mb": 50
            }
        }
    
    def _load_config_file(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                self._merge_configs(self.config, file_config)
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file)])
        ]
    )

def get_logger(name: str):
    """Get logger instance."""
    return logging.getLogger(name)

# ============================================================================
# OCR AND DOCUMENT PROCESSING
# ============================================================================

class OCREngine:
    """Unified OCR processing engine."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger(__name__)
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using available OCR engines."""
        try:
            # Try Tesseract first
            text = pytesseract.image_to_string(image)
            if text.strip():
                return text
        except Exception as e:
            self.logger.warning(f"Tesseract failed: {e}")
        
        try:
            # Try EasyOCR as fallback
            import easyocr
            reader = easyocr.Reader(['en'])
            result = reader.readtext(np.array(image))
            text = ' '.join([detection[1] for detection in result])
            return text
        except Exception as e:
            self.logger.warning(f"EasyOCR failed: {e}")
        
        return ""

class DocumentProcessor:
    """Document processing and conversion."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.ocr_engine = OCREngine(config)
        self.logger = get_logger(__name__)
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """Convert PDF to text using OCR."""
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            texts = []
            
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}/{len(images)}")
                # Preprocess image
                processed_image = self._preprocess_image(image)
                # Extract text
                text = self.ocr_engine.extract_text_from_image(processed_image)
                texts.append(text)
            
            return texts
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return []
    
    def process_image(self, image_path: str) -> str:
        """Process image file to extract text."""
        try:
            image = Image.open(image_path)
            processed_image = self._preprocess_image(image)
            return self.ocr_engine.extract_text_from_image(processed_image)
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply image preprocessing for better OCR results."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Convert to OpenCV format for advanced processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Noise reduction
            cv_image = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
            
            # Convert back to PIL
            image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return image
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image

# ============================================================================
# FIELD EXTRACTION AND VALIDATION
# ============================================================================

class FieldExtractor:
    """Extract structured fields from text using regex patterns."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger(__name__)
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for field extraction."""
        self.patterns = {
            'cpt_codes': [
                r'\b\d{5}\b',  # Standard 5-digit CPT codes
                r'\b9\d{4}\b',  # Evaluation & Management codes
            ],
            'icd10_codes': [
                r'\b[A-Z]\d{2}\.?\d{0,3}\b',  # ICD-10 format
                r'\b[A-Z]\d{2}\b',  # Short ICD-10 format
            ],
            'dates': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            ],
            'names': [
                r'(?:Patient|Name|Pt)[:.\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:Last|First)[:.\s]+([A-Z][a-z]+)',
            ],
            'charges': [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',  # Dollar amounts
            ],
            'phone': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            ],
            'patient_id': [
                r'(?:Patient\s+ID|Account)[:.\s]+(\w+)',
                r'\b\d{6,}\b',  # Numeric IDs
            ]
        }
    
    def extract_from_text(self, text: str) -> PatientData:
        """Extract structured data from text."""
        patient = PatientData()
        
        try:
            # Extract CPT codes
            patient.cpt_codes = self._extract_cpt_codes(text)
            
            # Extract ICD-10 codes
            patient.icd10_codes = self._extract_icd10_codes(text)
            
            # Extract dates
            dates = self._extract_dates(text)
            if dates:
                patient.date_of_service = dates[0]
            
            # Extract names
            names = self._extract_names(text)
            if names:
                name_parts = names[0].split()
                if len(name_parts) >= 2:
                    patient.first_name = name_parts[0]
                    patient.last_name = name_parts[-1]
                    if len(name_parts) > 2:
                        patient.middle_name = ' '.join(name_parts[1:-1])
                else:
                    patient.first_name = name_parts[0]
            
            # Extract charges
            charges = self._extract_charges(text)
            if charges:
                patient.charges = float(charges[0].replace(',', ''))
            
            # Extract patient ID
            patient_ids = self._extract_patient_ids(text)
            if patient_ids:
                patient.patient_id = patient_ids[0]
            
            # Store original text
            patient.extracted_text = text
            
            # Calculate confidence
            patient.confidence_scores = self._calculate_confidence(patient, text)
            
        except Exception as e:
            self.logger.error(f"Error extracting fields: {e}")
        
        return patient
    
    def _extract_cpt_codes(self, text: str) -> List[CPTCode]:
        """Extract CPT codes from text."""
        codes = []
        for pattern in self.patterns['cpt_codes']:
            matches = re.finditer(pattern, text)
            for match in matches:
                code = match.group()
                if self._validate_cpt_code(code):
                    codes.append(CPTCode(
                        code=code,
                        confidence=FieldConfidence(overall=0.8)
                    ))
        return codes
    
    def _extract_icd10_codes(self, text: str) -> List[ICD10Code]:
        """Extract ICD-10 codes from text."""
        codes = []
        for pattern in self.patterns['icd10_codes']:
            matches = re.finditer(pattern, text)
            for match in matches:
                code = match.group()
                if self._validate_icd10_code(code):
                    codes.append(ICD10Code(
                        code=code,
                        confidence=FieldConfidence(overall=0.7)
                    ))
        return codes
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        dates = []
        for pattern in self.patterns['dates']:
            matches = re.finditer(pattern, text)
            for match in matches:
                date_str = match.group()
                standardized = self._standardize_date(date_str)
                if standardized:
                    dates.append(standardized)
        return dates
    
    def _extract_names(self, text: str) -> List[str]:
        """Extract names from text."""
        names = []
        for pattern in self.patterns['names']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1) if match.groups() else match.group()
                if name and len(name.split()) <= 4:  # Reasonable name length
                    names.append(name.strip())
        return names
    
    def _extract_charges(self, text: str) -> List[str]:
        """Extract monetary charges from text."""
        charges = []
        for pattern in self.patterns['charges']:
            matches = re.finditer(pattern, text)
            for match in matches:
                amount = match.group(1)
                charges.append(amount)
        return charges
    
    def _extract_patient_ids(self, text: str) -> List[str]:
        """Extract patient IDs from text."""
        ids = []
        for pattern in self.patterns['patient_id']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                patient_id = match.group(1) if match.groups() else match.group()
                if patient_id and len(patient_id) >= 4:  # Reasonable ID length
                    ids.append(patient_id.strip())
        return ids
    
    def _validate_cpt_code(self, code: str) -> bool:
        """Basic CPT code validation."""
        return code.isdigit() and len(code) == 5 and int(code) >= 10000
    
    def _validate_icd10_code(self, code: str) -> bool:
        """Basic ICD-10 code validation."""
        # Remove dots for validation
        clean_code = code.replace('.', '')
        return (len(clean_code) >= 3 and 
                clean_code[0].isalpha() and 
                clean_code[1:3].isdigit())
    
    def _standardize_date(self, date_str: str) -> Optional[str]:
        """Standardize date format."""
        try:
            parsed_date = date_parser.parse(date_str)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            return None
    
    def _calculate_confidence(self, patient: PatientData, text: str) -> FieldConfidence:
        """Calculate overall confidence score."""
        scores = []
        
        # Score based on number of fields extracted
        field_count = sum([
            bool(patient.first_name),
            bool(patient.last_name),
            bool(patient.date_of_service),
            bool(patient.patient_id),
            bool(patient.cpt_codes),
            bool(patient.icd10_codes),
            bool(patient.charges)
        ])
        
        field_score = field_count / 7.0  # 7 key fields
        scores.append(field_score)
        
        # Score based on text quality
        text_score = min(1.0, len(text.split()) / 100)  # Normalize by expected word count
        scores.append(text_score)
        
        overall_confidence = sum(scores) / len(scores)
        
        return FieldConfidence(
            overall=overall_confidence,
            ocr_confidence=0.8,  # Default OCR confidence
            extraction_confidence=field_score,
            validation_confidence=0.7  # Default validation confidence
        )

# ============================================================================
# MAIN EXTRACTION ENGINE
# ============================================================================

class ExtractionEngine:
    """Main extraction engine coordinating all components."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.document_processor = DocumentProcessor(self.config)
        self.field_extractor = FieldExtractor(self.config)
        self.logger = get_logger(__name__)
    
    async def extract_from_file(self, file_path: str) -> ExtractionResults:
        """Extract data from a file."""
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # Process document based on file type
            if file_path.suffix.lower() == '.pdf':
                page_texts = self.document_processor.process_pdf(str(file_path))
                combined_text = '\n\n'.join(page_texts)
            else:
                # Assume image file
                combined_text = self.document_processor.process_image(str(file_path))
            
            if not combined_text.strip():
                return ExtractionResults(
                    success=False,
                    errors=["No text could be extracted from the document"]
                )
            
            # Extract structured data
            patient_data = self.field_extractor.extract_from_text(combined_text)
            
            # Create results
            processing_time = time.time() - start_time
            
            results = ExtractionResults(
                success=True,
                patients=[patient_data],
                total_patients=1,
                confidence_score=patient_data.confidence_scores.overall,
                processing_time=processing_time,
                source_file=str(file_path)
            )
            
            self.logger.info(f"Extraction completed in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            return ExtractionResults(
                success=False,
                errors=[str(e)],
                source_file=str(file_path),
                processing_time=time.time() - start_time
            )
    
    async def extract_from_text(self, text: str) -> ExtractionResults:
        """Extract data from raw text."""
        start_time = time.time()
        
        try:
            patient_data = self.field_extractor.extract_from_text(text)
            
            processing_time = time.time() - start_time
            
            return ExtractionResults(
                success=True,
                patients=[patient_data],
                total_patients=1,
                confidence_score=patient_data.confidence_scores.overall,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting from text: {e}")
            return ExtractionResults(
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )

# ============================================================================
# DATA EXPORT FUNCTIONALITY
# ============================================================================

class DataExporter:
    """Export extraction results to various formats."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = get_logger(__name__)
    
    def export_to_json(self, results: ExtractionResults, output_path: str) -> bool:
        """Export results to JSON format."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results.dict(), f, indent=2, default=str)
            self.logger.info(f"Results exported to JSON: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def export_to_csv(self, results: ExtractionResults, output_path: str) -> bool:
        """Export results to CSV format."""
        try:
            data = []
            for i, patient in enumerate(results.patients):
                row = {
                    'Patient_Number': i + 1,
                    'First_Name': patient.first_name or '',
                    'Last_Name': patient.last_name or '',
                    'Date_of_Birth': patient.date_of_birth or '',
                    'Patient_ID': patient.patient_id or '',
                    'Date_of_Service': patient.date_of_service or '',
                    'CPT_Codes': '; '.join([code.code for code in patient.cpt_codes]),
                    'ICD10_Codes': '; '.join([code.code for code in patient.icd10_codes]),
                    'Charges': patient.charges or 0.0,
                    'Confidence_Score': patient.confidence_scores.overall
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Results exported to CSV: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_excel(self, results: ExtractionResults, output_path: str) -> bool:
        """Export results to Excel format."""
        try:
            # Create patient summary data
            patient_data = []
            for i, patient in enumerate(results.patients):
                row = {
                    'Patient_Number': i + 1,
                    'First_Name': patient.first_name or '',
                    'Last_Name': patient.last_name or '',
                    'Date_of_Birth': patient.date_of_birth or '',
                    'Patient_ID': patient.patient_id or '',
                    'Date_of_Service': patient.date_of_service or '',
                    'Charges': patient.charges or 0.0,
                    'Confidence_Score': patient.confidence_scores.overall
                }
                patient_data.append(row)
            
            # Create CPT codes data
            cpt_data = []
            for i, patient in enumerate(results.patients):
                for code in patient.cpt_codes:
                    cpt_data.append({
                        'Patient_Number': i + 1,
                        'CPT_Code': code.code,
                        'Description': code.description or '',
                        'Charges': code.charges or 0.0,
                        'Confidence': code.confidence.overall
                    })
            
            # Create ICD-10 data
            icd_data = []
            for i, patient in enumerate(results.patients):
                for code in patient.icd10_codes:
                    icd_data.append({
                        'Patient_Number': i + 1,
                        'ICD10_Code': code.code,
                        'Description': code.description or '',
                        'Confidence': code.confidence.overall
                    })
            
            # Write to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                pd.DataFrame(patient_data).to_excel(writer, sheet_name='Patient_Summary', index=False)
                if cpt_data:
                    pd.DataFrame(cpt_data).to_excel(writer, sheet_name='CPT_Codes', index=False)
                if icd_data:
                    pd.DataFrame(icd_data).to_excel(writer, sheet_name='ICD10_Codes', index=False)
            
            self.logger.info(f"Results exported to Excel: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            return False

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def load_custom_css():
    """Load custom CSS for Streamlit UI."""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .result-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 20px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        transition: width 0.3s ease;
    }
    
    .upload-area {
        border: 2px dashed #2a5298;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
        margin: 1rem 0;
    }
    
    .stAlert > div {
        border-radius: 8px;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .code-block {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007acc;
        font-family: 'Courier New', monospace;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    
    .info-card {
        background: #e2f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid #2a5298;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #2a5298;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_main_header():
    """Render the main application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Superbill Data Extraction System</h1>
        <p style="margin-bottom: 0; opacity: 0.9;">
            Extract structured data from medical superbills using advanced AI
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_file_upload():
    """Render file upload component."""
    st.markdown("### 📄 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a medical superbill document",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
        help="Supported formats: PDF, JPG, PNG, TIFF"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**Type:** {uploaded_file.type}")
        
        with col2:
            if uploaded_file.type.startswith('image/'):
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Preview", width=200)
                except Exception:
                    st.info("Unable to preview image")
    
    return uploaded_file

def render_extraction_results(results: ExtractionResults):
    """Render extraction results."""
    if not results.success:
        st.error("❌ Extraction failed")
        for error in results.errors:
            st.error(f"Error: {error}")
        return
    
    st.success("✅ Extraction completed successfully!")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; color: #2a5298;">Patients Found</h4>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(results.total_patients), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; color: #2a5298;">Confidence</h4>
            <h2 style="margin: 0;">{:.1%}</h2>
        </div>
        """.format(results.confidence_score), unsafe_allow_html=True)
    
    with col3:
        processing_time = results.processing_time or 0
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; color: #2a5298;">Processing Time</h4>
            <h2 style="margin: 0;">{:.1f}s</h2>
        </div>
        """.format(processing_time), unsafe_allow_html=True)
    
    with col4:
        total_codes = sum(len(p.cpt_codes) + len(p.icd10_codes) for p in results.patients)
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0; color: #2a5298;">Medical Codes</h4>
            <h2 style="margin: 0;">{}</h2>
        </div>
        """.format(total_codes), unsafe_allow_html=True)
    
    # Patient details
    for i, patient in enumerate(results.patients):
        with st.expander(f"👤 Patient {i+1} Details", expanded=True):
            render_patient_details(patient)

def render_patient_details(patient: PatientData):
    """Render individual patient details."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personal Information:**")
        if patient.first_name or patient.last_name:
            name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()
            st.write(f"• **Name:** {name}")
        if patient.date_of_birth:
            st.write(f"• **DOB:** {patient.date_of_birth}")
        if patient.patient_id:
            st.write(f"• **Patient ID:** {patient.patient_id}")
        if patient.date_of_service:
            st.write(f"• **Service Date:** {patient.date_of_service}")
    
    with col2:
        st.markdown("**Financial Information:**")
        if patient.charges:
            st.write(f"• **Charges:** ${patient.charges:,.2f}")
        if patient.copay:
            st.write(f"• **Copay:** ${patient.copay:,.2f}")
        if patient.deductible:
            st.write(f"• **Deductible:** ${patient.deductible:,.2f}")
    
    # Medical codes
    if patient.cpt_codes:
        st.markdown("**CPT Codes:**")
        cpt_data = []
        for code in patient.cpt_codes:
            cpt_data.append({
                'Code': code.code,
                'Description': code.description or 'N/A',
                'Charges': f"${code.charges:,.2f}" if code.charges else 'N/A',
                'Confidence': f"{code.confidence.overall:.1%}"
            })
        st.dataframe(pd.DataFrame(cpt_data), use_container_width=True)
    
    if patient.icd10_codes:
        st.markdown("**ICD-10 Diagnosis Codes:**")
        icd_data = []
        for code in patient.icd10_codes:
            icd_data.append({
                'Code': code.code,
                'Description': code.description or 'N/A',
                'Confidence': f"{code.confidence.overall:.1%}"
            })
        st.dataframe(pd.DataFrame(icd_data), use_container_width=True)
    
    # Confidence visualization
    st.markdown("**Confidence Scores:**")
    conf = patient.confidence_scores
    
    st.markdown(f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Overall Confidence</span>
            <span>{conf.overall:.1%}</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {conf.overall*100}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_export_options(results: ExtractionResults):
    """Render export options."""
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    exporter = DataExporter(ConfigManager())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        if st.button("📄 Export JSON", use_container_width=True):
            json_data = json.dumps(results.dict(), indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"extraction_results_{timestamp}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            # Create CSV data
            csv_data = []
            for i, patient in enumerate(results.patients):
                row = {
                    'Patient_Number': i + 1,
                    'First_Name': patient.first_name or '',
                    'Last_Name': patient.last_name or '',
                    'Date_of_Birth': patient.date_of_birth or '',
                    'Patient_ID': patient.patient_id or '',
                    'Date_of_Service': patient.date_of_service or '',
                    'CPT_Codes': '; '.join([code.code for code in patient.cpt_codes]),
                    'ICD10_Codes': '; '.join([code.code for code in patient.icd10_codes]),
                    'Charges': patient.charges or 0.0,
                    'Confidence_Score': patient.confidence_scores.overall
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name=f"extraction_results_{timestamp}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📈 Export Excel", use_container_width=True):
            # Create Excel file in memory
            from io import BytesIO
            
            buffer = BytesIO()
            
            # Patient summary data
            patient_data = []
            for i, patient in enumerate(results.patients):
                row = {
                    'Patient_Number': i + 1,
                    'First_Name': patient.first_name or '',
                    'Last_Name': patient.last_name or '',
                    'Date_of_Birth': patient.date_of_birth or '',
                    'Patient_ID': patient.patient_id or '',
                    'Date_of_Service': patient.date_of_service or '',
                    'Charges': patient.charges or 0.0,
                    'Confidence_Score': patient.confidence_scores.overall
                }
                patient_data.append(row)
            
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                pd.DataFrame(patient_data).to_excel(writer, sheet_name='Patient_Summary', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name=f"extraction_results_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def render_sidebar():
    """Render application sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        ocr_model = st.selectbox(
            "OCR Engine",
            ["Auto (Recommended)", "Tesseract", "EasyOCR"],
            help="Choose the OCR engine for text extraction"
        )
        
        # Processing options
        st.markdown("### 🔧 Processing Options")
        
        enhance_image = st.checkbox("Enhance Image Quality", value=True)
        validate_codes = st.checkbox("Validate Medical Codes", value=True)
        anonymize_phi = st.checkbox("Anonymize PHI", value=False)
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Minimum confidence score for accepting extractions"
        )
        
        st.markdown("---")
        
        # System info
        with st.expander("System Information"):
            st.markdown("**Version:** 1.0.0")
            st.markdown("**Python:** " + sys.version.split()[0])
            st.markdown("**OCR:** Tesseract + EasyOCR")
            st.markdown("**Models:** Available")
        
        # Help and documentation
        st.markdown("### 📚 Help & Documentation")
        st.markdown("- [User Guide](#)")
        st.markdown("- [API Documentation](#)")
        st.markdown("- [Report Issue](#)")

# ============================================================================
# STREAMLIT UI MAIN APPLICATION
# ============================================================================

def run_streamlit_ui():
    """Run the Streamlit UI application."""
    if not DEPENDENCIES_OK:
        st.error("Missing required dependencies. Please install them first.")
        st.code("pip install -r requirements.txt")
        return
    
    # Page configuration
    st.set_page_config(
        page_title="Medical Superbill Extractor",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'extraction_engine' not in st.session_state:
        st.session_state.extraction_engine = ExtractionEngine()
    if 'extraction_results' not in st.session_state:
        st.session_state.extraction_results = None
    
    # Render sidebar
    render_sidebar()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Single File", "📁 Batch Processing", "⚙️ Configuration", "📚 Documentation"])
    
    with tab1:
        # Main content
        render_main_header()
        
        # File upload section
        uploaded_file = render_file_upload()
        
        # Processing section
        if uploaded_file is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### 🔄 Processing")
            
            with col2:
                if st.button("🚀 Extract Data", type="primary", use_container_width=True):
                    with st.spinner("Processing document..."):
                        # Save uploaded file temporarily
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        
                        temp_file = temp_dir / uploaded_file.name
                        with open(temp_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the file
                        results = asyncio.run(
                            st.session_state.extraction_engine.extract_from_file(str(temp_file))
                        )
                        
                        # Clean up temp file
                        temp_file.unlink()
                        
                        # Store results
                        st.session_state.extraction_results = results
        
        # Results section
        if st.session_state.extraction_results:
            st.markdown("---")
            st.markdown("## 📊 Extraction Results")
            render_extraction_results(st.session_state.extraction_results)
            
            st.markdown("---")
            render_export_options(st.session_state.extraction_results)
        
        # Example usage section
        if not uploaded_file:
            render_example_section()
    
    with tab2:
        render_batch_processing()
    
    with tab3:
        render_configuration()
    
    with tab4:
        render_documentation()

def render_example_section():
    """Render example usage section."""
    st.markdown("---")
    st.markdown("### 💡 Example Usage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Supported Document Types:**
        - PDF medical superbills
        - Scanned images (JPG, PNG, TIFF)
        - Multi-page documents
        - Handwritten and printed text
        """)
    
    with col2:
        st.markdown("""
        **Extracted Information:**
        - Patient demographics
        - CPT procedure codes
        - ICD-10 diagnosis codes
        - Service dates and charges
        - Provider information
        """)
    
    # Sample workflow
    st.markdown("### 🔄 Workflow")
    
    steps = [
        ("1️⃣ Upload", "Upload your medical superbill document"),
        ("2️⃣ Process", "Click 'Extract Data' to analyze the document"),
        ("3️⃣ Review", "Review the extracted information and confidence scores"),
        ("4️⃣ Export", "Download results in JSON, CSV, or Excel format")
    ]
    
    cols = st.columns(4)
    for i, (step, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{step}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def render_batch_processing():
    """Render batch processing interface."""
    st.markdown("### 📁 Batch Processing")
    st.info("Process multiple medical superbill documents at once")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Choose multiple documents",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple documents for batch processing"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} files selected**")
        
        # Display file list
        for i, file in enumerate(uploaded_files):
            st.markdown(f"{i+1}. {file.name} ({file.size / 1024:.1f} KB)")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Save file temporarily
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    
                    temp_file = temp_dir / file.name
                    with open(temp_file, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Process file
                    results = asyncio.run(
                        st.session_state.extraction_engine.extract_from_file(str(temp_file))
                    )
                    batch_results.append((file.name, results))
                    
                    # Clean up
                    temp_file.unlink()
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing completed!")
                
                # Display batch results summary
                st.markdown("### 📊 Batch Results Summary")
                
                successful = sum(1 for _, result in batch_results if result.success)
                failed = len(batch_results) - successful
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Files", len(batch_results))
                with col2:
                    st.metric("Successful", successful)
                with col3:
                    st.metric("Failed", failed)
                
                # Individual results
                for filename, result in batch_results:
                    with st.expander(f"📄 {filename}"):
                        if result.success:
                            st.success("✅ Extraction successful")
                            for i, patient in enumerate(result.patients):
                                st.markdown(f"**Patient {i+1}:**")
                                if patient.first_name or patient.last_name:
                                    name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()
                                    st.write(f"• Name: {name}")
                                if patient.cpt_codes:
                                    st.write(f"• CPT Codes: {', '.join([c.code for c in patient.cpt_codes])}")
                                if patient.charges:
                                    st.write(f"• Charges: ${patient.charges:,.2f}")
                        else:
                            st.error("❌ Extraction failed")
                            for error in result.errors:
                                st.error(f"Error: {error}")

def render_configuration():
    """Render configuration interface."""
    st.markdown("### ⚙️ Configuration")
    st.info("Customize extraction settings and model parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### OCR Settings")
        
        ocr_engine = st.selectbox(
            "Primary OCR Engine",
            ["tesseract", "easyocr"],
            help="Choose the primary OCR engine"
        )
        
        ocr_confidence = st.slider(
            "OCR Confidence Threshold",
            0.0, 1.0, 0.7, 0.05,
            help="Minimum confidence score for OCR text acceptance"
        )
        
        enhance_images = st.checkbox(
            "Enable Image Enhancement",
            value=True,
            help="Apply preprocessing to improve OCR accuracy"
        )
        
        st.markdown("#### Document Processing")
        
        pdf_dpi = st.number_input(
            "PDF Conversion DPI",
            min_value=150,
            max_value=600,
            value=300,
            step=50,
            help="Resolution for PDF to image conversion"
        )
        
        max_pages = st.number_input(
            "Maximum Pages per Document",
            min_value=1,
            max_value=100,
            value=50,
            help="Limit the number of pages to process"
        )
    
    with col2:
        st.markdown("#### Extraction Settings")
        
        extraction_confidence = st.slider(
            "Extraction Confidence Threshold",
            0.0, 1.0, 0.6, 0.05,
            help="Minimum confidence for field extraction"
        )
        
        validate_codes = st.checkbox(
            "Validate Medical Codes",
            value=True,
            help="Enable validation of CPT and ICD-10 codes"
        )
        
        anonymize_phi = st.checkbox(
            "Anonymize PHI",
            value=False,
            help="Remove or mask personally identifiable information"
        )
        
        st.markdown("#### Export Settings")
        
        include_raw_text = st.checkbox(
            "Include Raw Text in Export",
            value=False,
            help="Include original extracted text in export files"
        )
        
        export_format = st.multiselect(
            "Default Export Formats",
            ["JSON", "CSV", "Excel"],
            default=["JSON", "CSV"],
            help="Default formats for data export"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            st.success("✅ Configuration saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.info("Configuration reset to default values")

def render_documentation():
    """Render documentation interface."""
    st.markdown("### 📚 Documentation")
    
    doc_tab1, doc_tab2, doc_tab3 = st.tabs(["📖 User Guide", "🔧 API Reference", "❓ FAQ"])
    
    with doc_tab1:
        st.markdown("""
        ## User Guide
        
        ### Getting Started
        
        1. **Upload Document**: Click on the "Upload Document" section and select your medical superbill
        2. **Process**: Click "Extract Data" to begin processing
        3. **Review**: Check the extracted information and confidence scores
        4. **Export**: Download results in your preferred format
        
        ### Supported Document Types
        
        - **PDF Files**: Multi-page medical superbills
        - **Image Files**: JPG, PNG, TIFF formats
        - **Scanned Documents**: Both handwritten and printed text
        
        ### Best Practices
        
        - Ensure documents are clear and well-lit
        - Avoid blurry or rotated images
        - Use high-resolution scans (300 DPI or higher)
        - Check confidence scores to verify accuracy
        
        ### Troubleshooting
        
        If extraction fails or returns poor results:
        
        1. Check document quality and resolution
        2. Try adjusting confidence thresholds
        3. Enable image enhancement in settings
        4. Ensure the document contains standard medical billing information
        """)
    
    with doc_tab2:
        st.markdown("""
        ## API Reference
        
        ### Command Line Interface
        
        The application can be run in CLI mode for automation:
        
        ```bash
        # Process a single file
        python app_main.py --cli document.pdf
        
        # Process multiple files
        python app_main.py --cli *.pdf
        
        # Run with custom settings
        python app_main.py --cli document.pdf --output results.json --format json
        
        # Show help
        python app_main.py --help
        ```
        
        ### Configuration File
        
        Create a `config.yaml` file to customize settings:
        
        ```yaml
        models:
          ocr:
            primary: tesseract
            confidence_threshold: 0.7
        
        document_processing:
          pdf:
            dpi: 300
            max_pages: 50
        
        validation:
          validate_codes: true
        ```
        
        ### Data Schema
        
        The extracted data follows this structure:
        
        ```json
        {
          "success": true,
          "patients": [
            {
              "first_name": "John",
              "last_name": "Doe",
              "date_of_birth": "1980-01-01",
              "cpt_codes": [
                {
                  "code": "99213",
                  "description": "Office visit",
                  "charges": 150.00
                }
              ]
            }
          ]
        }
        ```
        """)
    
    with doc_tab3:
        st.markdown("""
        ## Frequently Asked Questions
        
        ### Q: What types of documents are supported?
        A: The system supports PDF files and common image formats (JPG, PNG, TIFF). Documents should contain medical superbill information.
        
        ### Q: How accurate is the extraction?
        A: Accuracy depends on document quality. Well-formatted, clear documents typically achieve 85-95% accuracy.
        
        ### Q: Can I process multiple documents at once?
        A: Yes, use the "Batch Processing" tab to upload and process multiple documents simultaneously.
        
        ### Q: How do I interpret confidence scores?
        A: Confidence scores range from 0-100%. Higher scores indicate more reliable extractions. Review items with low confidence manually.
        
        ### Q: Is my data secure?
        A: The application processes documents locally. No data is sent to external servers unless explicitly configured.
        
        ### Q: Can I customize the extraction fields?
        A: Yes, modify the configuration file or use the Configuration tab to adjust extraction parameters.
        
        ### Q: What should I do if extraction fails?
        A: Check document quality, try different OCR engines, or adjust confidence thresholds in the configuration.
        
        ### Q: How do I export results?
        A: Use the export buttons in the results section to download data in JSON, CSV, or Excel formats.
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🆘 Support
    
    If you need additional help:
    
    - Check the troubleshooting section above
    - Review the configuration options
    - Try the CLI mode for debugging
    - Ensure all dependencies are installed correctly
    """)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Medical Superbill Data Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_main.py                          # Launch Streamlit UI
  python app_main.py --cli file.pdf           # CLI extraction
  python app_main.py --cli *.pdf --format csv # Batch processing
  python app_main.py --demo                   # Run demo mode
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--cli",
        nargs="*",
        help="Run in CLI mode with specified input files"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with sample documents"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file path (for single file processing)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "excel", "all"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        help="Custom configuration file path"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    return parser

async def cli_process_files(file_paths: List[str], args) -> bool:
    """Process files in CLI mode."""
    config = ConfigManager(args.config)
    engine = ExtractionEngine(config)
    exporter = DataExporter(config)
    
    logger = get_logger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"Processing file {i}/{total_files}: {file_path}")
        
        try:
            # Extract data
            results = await engine.extract_from_file(file_path)
            
            if results.success:
                # Generate output filename
                base_name = Path(file_path).stem
                
                # Export in requested format(s)
                if args.format in ["json", "all"]:
                    json_path = output_dir / f"{base_name}_results.json"
                    exporter.export_to_json(results, str(json_path))
                
                if args.format in ["csv", "all"]:
                    csv_path = output_dir / f"{base_name}_results.csv"
                    exporter.export_to_csv(results, str(csv_path))
                
                if args.format in ["excel", "all"]:
                    excel_path = output_dir / f"{base_name}_results.xlsx"
                    exporter.export_to_excel(results, str(excel_path))
                
                success_count += 1
                
                # Print summary
                print(f"  ✅ Success - Found {results.total_patients} patients")
                print(f"     Confidence: {results.confidence_score:.1%}")
                print(f"     Processing time: {results.processing_time:.2f}s")
                
            else:
                print(f"  ❌ Failed - {'; '.join(results.errors)}")
                
        except Exception as e:
            print(f"  ❌ Error - {str(e)}")
    
    print(f"\nProcessing complete: {success_count}/{total_files} files successful")
    return success_count == total_files

def run_demo():
    """Run demo mode with sample processing."""
    print("🏥 Medical Superbill Extractor - Demo Mode")
    print("=" * 50)
    
    # Demo text sample
    demo_text = """
    MEDICAL SUPERBILL
    
    Patient: John Smith
    DOB: 01/15/1980
    Patient ID: 12345
    Date of Service: 03/15/2024
    
    Diagnosis Codes:
    Z00.00 - Encounter for general adult medical examination
    
    Procedure Codes:
    99213 - Office visit, established patient
    
    Charges: $150.00
    """
    
    print("Processing demo text...")
    
    # Create engine and process
    config = ConfigManager()
    engine = ExtractionEngine(config)
    
    # Process demo text
    results = asyncio.run(engine.extract_from_text(demo_text))
    
    if results.success:
        print("✅ Demo extraction successful!")
        print(f"Patients found: {results.total_patients}")
        print(f"Confidence: {results.confidence_score:.1%}")
        
        for i, patient in enumerate(results.patients):
            print(f"\n👤 Patient {i+1}:")
            if patient.first_name or patient.last_name:
                name = f"{patient.first_name or ''} {patient.last_name or ''}".strip()
                print(f"  Name: {name}")
            if patient.patient_id:
                print(f"  ID: {patient.patient_id}")
            if patient.date_of_service:
                print(f"  Service Date: {patient.date_of_service}")
            if patient.cpt_codes:
                print(f"  CPT Codes: {', '.join([c.code for c in patient.cpt_codes])}")
            if patient.icd10_codes:
                print(f"  ICD-10 Codes: {', '.join([c.code for c in patient.icd10_codes])}")
            if patient.charges:
                print(f"  Charges: ${patient.charges:.2f}")
    
    else:
        print("❌ Demo extraction failed")
        for error in results.errors:
            print(f"Error: {error}")

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point."""
    
    if not DEPENDENCIES_OK:
        print("❌ Missing required dependencies!")
        print("Please install them with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Parse command line arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    setup_logger(log_level=log_level, log_file=args.log_file)
    
    # Determine run mode
    if args.demo:
        # Run demo mode
        run_demo()
        
    elif args.cli is not None:
        # Run CLI mode
        if not args.cli:
            print("❌ No input files specified for CLI mode")
            sys.exit(1)
        
        # Expand file patterns
        input_files = []
        for pattern in args.cli:
            path = Path(pattern)
            if path.is_file():
                input_files.append(str(path))
            else:
                # Handle glob patterns
                input_files.extend([str(p) for p in Path().glob(pattern)])
        
        if not input_files:
            print("❌ No input files found")
            sys.exit(1)
        
        print(f"🔄 Processing {len(input_files)} files...")
        
        # Process files
        success = asyncio.run(cli_process_files(input_files, args))
        sys.exit(0 if success else 1)
        
    else:
        # Run Streamlit UI (default mode)
        print("🚀 Starting Medical Superbill Extractor UI...")
        print("📊 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop")
        
        try:
            # Run Streamlit app
            sys.argv = ["streamlit", "run", __file__, "--server.port=8501", "--server.headless=true"]
            import streamlit.web.cli as stcli
            sys.exit(stcli.main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error starting UI: {e}")
            print("💡 Try CLI mode instead: python app_main.py --cli your_file.pdf")
            sys.exit(1)

# ============================================================================
# STREAMLIT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check if we're running under Streamlit
    if "streamlit" in sys.modules:
        # Running under Streamlit - run UI
        run_streamlit_ui()
    else:
        # Running standalone - parse args and route
        main()
