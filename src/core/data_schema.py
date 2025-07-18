"""
Data Schema Definitions for Medical Superbill Extraction

Defines the structure and validation rules for extracted medical data.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import date, datetime
from enum import Enum
import re


class OCRResult(BaseModel):
    """Represents the output of an OCR engine for a single page or image."""
    text: str
    confidence: float
    model_name: str
    processing_time: Optional[float] = None
    bounding_boxes: Optional[List[Dict[str, Any]]] = None


class FieldConfidence(BaseModel):
    """Confidence scoring for extracted fields."""
    value: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    source: str = Field(..., description="Source of extraction (OCR model, NLP model, etc.)")
    method: str = Field(..., description="Extraction method used")


class PHIType(str, Enum):
    """Types of Protected Health Information."""
    NAME = "name"
    SSN = "ssn"
    DOB = "date_of_birth"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    ACCOUNT_NUMBER = "account_number"
    OTHER = "other"


class CPTCode(BaseModel):
    """CPT (Current Procedural Terminology) code structure."""
    code: str = Field(..., pattern=r"^\d{5}$", description="5-digit CPT code")
    description: Optional[str] = Field(None, description="Procedure description")
    modifier: Optional[str] = Field(None, description="CPT modifier codes")
    units: Optional[int] = Field(None, ge=1, description="Number of units performed")
    charge: Optional[float] = Field(None, ge=0, description="Charge amount for procedure")
    confidence: Optional[FieldConfidence] = None
    
    @validator('code')
    def validate_cpt_code(cls, v):
        """Validate CPT code format."""
        if not re.match(r'^\d{5}$', v):
            raise ValueError('CPT code must be exactly 5 digits')
        return v


class ICD10Code(BaseModel):
    """ICD-10 diagnosis code structure."""
    code: str = Field(..., description="ICD-10 diagnosis code")
    description: Optional[str] = Field(None, description="Diagnosis description")
    is_primary: bool = Field(False, description="Whether this is the primary diagnosis")
    confidence: Optional[FieldConfidence] = None
    
    @validator('code')
    def validate_icd10_code(cls, v):
        """Validate ICD-10 code format."""
        # Basic ICD-10 pattern: Letter + 2 digits + optional decimal + 1-3 digits
        pattern = r'^[A-Z]\d{2}(\.[\dA-Z]{1,3})?$'
        if not re.match(pattern, v):
            raise ValueError('Invalid ICD-10 code format')
        return v


class InsuranceInfo(BaseModel):
    """Patient insurance information."""
    insurance_company: Optional[str] = None
    policy_number: Optional[str] = None
    group_number: Optional[str] = None
    subscriber_id: Optional[str] = None
    relationship_to_subscriber: Optional[str] = None
    confidence: Optional[FieldConfidence] = None


class Address(BaseModel):
    """Address structure."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = "USA"
    confidence: Optional[FieldConfidence] = None
    
    @validator('zip_code')
    def validate_zip_code(cls, v):
        """Validate US ZIP code format."""
        if v and not re.match(r'^\d{5}(-\d{4})?$', v):
            raise ValueError('Invalid ZIP code format')
        return v


class ContactInfo(BaseModel):
    """Contact information structure."""
    phone: Optional[str] = None
    email: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None
    confidence: Optional[FieldConfidence] = None
    
    @validator('phone', 'emergency_phone')
    def validate_phone(cls, v):
        """Validate phone number format."""
        if v:
            # Remove formatting and validate 10-digit US phone
            clean_phone = re.sub(r'[^\d]', '', v)
            if len(clean_phone) != 10:
                raise ValueError('Phone number must be 10 digits')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if v and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v


class ProviderInfo(BaseModel):
    """Healthcare provider information."""
    name: Optional[str] = None
    npi_number: Optional[str] = None
    practice_name: Optional[str] = None
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None
    tax_id: Optional[str] = None
    referring_provider: Optional[str] = None
    referring_npi: Optional[str] = None
    signature_present: bool = False
    confidence: Optional[FieldConfidence] = None
    
    @validator('npi_number', 'referring_npi')
    def validate_npi(cls, v):
        """Validate NPI number format."""
        if v and not re.match(r'^\d{10}$', v):
            raise ValueError('NPI must be exactly 10 digits')
        return v


class ServiceInfo(BaseModel):
    """Service/visit information."""
    date_of_service: Optional[date] = None
    claim_date: Optional[date] = None
    place_of_service: Optional[str] = None
    visit_type: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_minutes: Optional[int] = Field(None, ge=0)
    chief_complaint: Optional[str] = None
    # Additional fields used by exporters / down-stream code
    provider_name: Optional[str] = None
    provider_npi: Optional[str] = None
    facility_name: Optional[str] = None
    confidence: Optional[FieldConfidence] = None


class FinancialInfo(BaseModel):
    """Financial and billing information."""
    total_charges: Optional[float] = Field(None, ge=0)
    amount_paid: Optional[float] = Field(None, ge=0)
    outstanding_balance: Optional[float] = Field(None, ge=0)
    copay: Optional[float] = Field(None, ge=0)
    deductible: Optional[float] = Field(None, ge=0)
    # Additional fields referenced in exporters
    insurance_payment: Optional[float] = Field(None, ge=0)
    patient_payment: Optional[float] = Field(None, ge=0)
    balance_due: Optional[float] = Field(None, ge=0)
    payment_method: Optional[str] = None
    confidence: Optional[FieldConfidence] = None


class PHIItem(BaseModel):
    """Protected Health Information item."""
    phi_type: PHIType
    value: str
    location: Optional[Dict[str, Any]] = None  # Bounding box or text position
    confidence: Optional[FieldConfidence] = None
    anonymized: bool = False
    replacement_value: Optional[str] = None


class PatientData(BaseModel):
    """Complete patient data structure."""
    # Patient Demographics
    patient_id: Optional[str] = None
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[date] = None
    gender: Optional[str] = None
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None
    # Quick access fields extracted directly (flattened)
    phone: Optional[str] = None
 
    # Insurance Information
    insurance: Optional[InsuranceInfo] = None
    
    # Medical Codes and Procedures
    cpt_codes: List[CPTCode] = Field(default_factory=list)
    icd10_codes: List[ICD10Code] = Field(default_factory=list)
    
    # Service Information
    service_info: Optional[ServiceInfo] = None
    
    # Provider Information
    provider: Optional[ProviderInfo] = None
    
    # Financial Information
    financial_info: Optional[FinancialInfo] = None
    
    # PHI Detection
    phi_detected: List[PHIItem] = Field(default_factory=list)
    
    # Metadata
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_errors: List[str] = Field(default_factory=list)
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Raw extracted text sections
    raw_text_sections: Dict[str, str] = Field(default_factory=dict)
    # Index of patient in multi-patient documents (assigned during processing)
    patient_index: Optional[int] = None


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    document_id: str
    source_file: str
    file_type: str
    page_count: int
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    ocr_models_used: List[str] = Field(default_factory=list)
    nlp_models_used: List[str] = Field(default_factory=list)
    extraction_settings: Dict[str, Any] = Field(default_factory=dict)
    

class SuperbillDocument(BaseModel):
    """Complete superbill document structure."""
    metadata: DocumentMetadata
    patients: List[PatientData] = Field(default_factory=list)
    document_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_errors: List[str] = Field(default_factory=list)
    
    # Quality metrics
    total_patients_detected: int = 0
    total_cpt_codes: int = 0
    total_icd10_codes: int = 0
    phi_items_detected: int = 0
    
    def update_metrics(self):
        """Update document-level metrics."""
        self.total_patients_detected = len(self.patients)
        self.total_cpt_codes = sum(len(p.cpt_codes) for p in self.patients)
        self.total_icd10_codes = sum(len(p.icd10_codes) for p in self.patients)
        self.phi_items_detected = sum(len(p.phi_detected) for p in self.patients)


class ExtractionResult(BaseModel):
    """Final extraction result for export."""
    success: bool
    document: Optional[SuperbillDocument] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    
    # Export formats
    csv_data: Optional[List[Dict[str, Any]]] = None
    json_data: Optional[Dict[str, Any]] = None


class ExtractionResults(BaseModel):
    """Represents the complete results of an extraction process."""
    success: bool
    file_path: str
    extraction_timestamp: datetime
    total_patients: int
    patients: List['PatientData'] = []
    extraction_confidence: float = 0.0
    metadata: Dict[str, Any] = {}


# Validation patterns and constants
VALIDATION_PATTERNS = {
    'cpt_code': r'^\d{5}$',
    'icd10_code': r'^[A-Z]\d{2}(\.[\dA-Z]{1,3})?$',
    'npi_number': r'^\d{10}$',
    'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
    'phone': r'^\d{3}[-.]?\d{3}[-.]?\d{4}$',
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'zip_code': r'^\d{5}(-\d{4})?$'
}

# Common field mappings for flexible extraction
FIELD_MAPPINGS = {
    'patient_name': ['patient_name', 'name', 'patient', 'pt_name'],
    'date_of_birth': ['dob', 'date_of_birth', 'birth_date', 'birthdate'],
    'date_of_service': ['dos', 'date_of_service', 'service_date', 'visit_date'],
    'claim_date': ['claim_date', 'billing_date', 'statement_date'],
    'cpt_code': ['cpt', 'cpt_code', 'procedure_code', 'proc_code'],
    'icd10_code': ['icd10', 'icd_10', 'diagnosis_code', 'dx_code', 'dx']
}
