"""
Data Schema for Structured Extraction System

Defines data structures and types for the extraction system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import date
from enum import Enum


class PHIType(Enum):
    """Types of PHI (Protected Health Information)."""
    SSN = "ssn"
    NAME = "name"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    DOB = "dob"
    MRN = "mrn"
    ACCOUNT = "account"
    OTHER = "other"


@dataclass
class PHIItem:
    """Protected Health Information item."""
    phi_type: PHIType
    value: str
    confidence: float
    position: Optional[tuple] = None


@dataclass
class FieldConfidence:
    """Confidence score for a field detection."""
    field_name: str
    confidence: float
    model_name: str


@dataclass
class OCRResult:
    """OCR extraction result with confidence score."""
    text: str
    confidence: float
    model_name: str = "default"
    processing_time: float = 0.0


@dataclass
class StructuredData:
    """Structured data extraction result."""
    data: Dict[str, Any]
    confidence: float
    model_name: str = "default"


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    file_name: str
    file_path: str
    file_size: int
    page_count: int
    processing_time: float
    extraction_date: str


@dataclass
class ExtractionResults:
    """Complete extraction results."""
    metadata: Optional[DocumentMetadata] = None
    text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    ocr_confidence: float = 0.0
    extraction_confidence: float = 0.0
    overall_confidence: float = 0.0
    # Additional fields for compatibility
    success: bool = True
    file_path: str = ""
    extraction_timestamp: Optional[str] = None
    total_patients: int = 0
    patients: List[Any] = field(default_factory=list)  # Will be List[PatientData] but avoiding circular reference
    error_message: Optional[str] = None


@dataclass
class PatientData:
    """Patient information."""
    id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_id: Optional[str] = None
    # Page-related fields for multi-page processing
    page_number: Optional[int] = None  # The page where this patient was found
    source_pages: List[int] = field(default_factory=list)  # All pages that contain this patient's data
    source_page_text: Optional[str] = None  # Truncated source text for debugging
    insurance_provider: Optional[str] = None
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)


@dataclass
class CPTCode:
    """CPT code with description and charge."""
    code: str
    description: Optional[str] = None
    charge: Optional[float] = None
    units: int = 1
    date: Optional[date] = None
    confidence: float = 0.0


@dataclass
class ICD10Code:
    """ICD-10 diagnosis code."""
    code: str
    description: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ServiceInfo:
    """Medical service information."""
    cpt_codes: List[CPTCode] = field(default_factory=list)
    icd10_codes: List[ICD10Code] = field(default_factory=list)
    service_date: Optional[date] = None
    total_charge: Optional[float] = None
    provider_name: Optional[str] = None
    facility_name: Optional[str] = None
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)


@dataclass
class Address:
    """Address information."""
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = "USA"


@dataclass
class ContactInfo:
    """Contact information."""
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Address] = None


@dataclass
class ProviderInfo:
    """Healthcare provider information."""
    name: Optional[str] = None
    npi: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    tax_id: Optional[str] = None
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)


@dataclass
class FinancialInfo:
    """Financial information from the document."""
    total_charge: Optional[float] = None
    insurance_paid: Optional[float] = None
    patient_paid: Optional[float] = None
    adjustments: Optional[float] = None
    balance_due: Optional[float] = None
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)


@dataclass
class SuperbillDocument:
    """Complete superbill document."""
    document_id: str
    patients: List[PatientData] = field(default_factory=list)
    services: List[ServiceInfo] = field(default_factory=list)
    provider: Optional[ProviderInfo] = None
    financial: Optional[FinancialInfo] = None
    document_date: Optional[date] = None
    confidence: float = 0.0
