"""
Data Schema for Structured Extraction System

Defines data structures and types for the extraction system.
"""

from __future__ import annotations  # Enable forward references
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum


class PHIType(Enum):
    """Types of PHI (Protected Health Information)."""
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    SSN = "ssn"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    INSURANCE_ID = "insurance_id"
    PATIENT_ID = "patient_id"


@dataclass
class PHIItem:
    """Protected Health Information item."""
    phi_type: PHIType
    value: str
    confidence: float = 0.0
    position: Optional[Tuple[int, int]] = None  # Start, end positions in text
    context: Optional[str] = None  # Surrounding text for validation


# --- Move CPTCode and ICD10Code definitions before PatientData ---
@dataclass
class CPTCode:
    """
    CPT (Current Procedural Terminology) code with metadata.
    """
    code: str
    description: Optional[str] = None
    charge: Optional[float] = None
    units: int = 1
    date: Optional[str] = None  # Use string literal for date
    confidence: float = 0.0

@dataclass
class ICD10Code:
    """
    ICD-10 diagnosis code with metadata.
    
    Attributes:
        code: The ICD-10 code (letter + numbers + optional decimal portion)
        description: Optional description of the diagnosis
        confidence: Confidence score of the extraction (0.0 to 1.0)
    """
    code: str
    description: Optional[str] = None
    confidence: float = 0.0


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
    additional_data: Optional[Dict[str, Any]] = field(default_factory=dict)


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
    # Use string literal for forward reference to avoid circular dependency
    patients: List['PatientData'] = field(default_factory=list)
    error_message: Optional[str] = None




@dataclass
class PatientData:
    """
    Patient information with multi-page support.
    """
    # Required fields (no defaults)
    first_name: str
    last_name: str
    
    # Optional fields with defaults
    middle_name: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO format YYYY-MM-DD
    patient_id: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    insurance_id: Optional[str] = None
    insurance_provider: Optional[str] = None
    date_of_service: Optional[str] = None  # ISO format YYYY-MM-DD
    
    # Fields with default values
    cpt_codes: List['CPTCode'] = field(default_factory=list)
    icd10_codes: List['ICD10Code'] = field(default_factory=list)
    page_number: int = 0
    total_pages: int = 0
    spans_multiple_pages: bool = False
    page_numbers: List[int] = field(default_factory=list)
    text_segment: Optional[str] = None
    extraction_confidence: float = 0.0
    patient_index: int = 0
    validation_errors: List[str] = field(default_factory=list)
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)
    financial_info: Optional['FinancialInfo'] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "middle_name": self.middle_name,
            "date_of_birth": str(self.date_of_birth) if self.date_of_birth else None,
            "patient_id": self.patient_id,
            "gender": self.gender,
            "address": self.address,
            "phone": self.phone,
            "email": self.email,
            "insurance_id": self.insurance_id,
            "insurance_provider": self.insurance_provider,
            "date_of_service": str(self.date_of_service) if self.date_of_service else None,
            "cpt_codes": [{"code": code.code, "description": code.description, "charge": code.charge} 
                         for code in self.cpt_codes] if self.cpt_codes else [],
            "icd10_codes": [{"code": code.code, "description": code.description} 
                         for code in self.icd10_codes] if self.icd10_codes else [],
            "extraction_confidence": self.extraction_confidence,
            "page_number": self.page_number,
            "total_pages": self.total_pages,
        }
        if self.spans_multiple_pages:
            result["spans_multiple_pages"] = True
            result["page_numbers"] = self.page_numbers
        if self.validation_errors:
            result["validation_errors"] = self.validation_errors
        if self.confidences:
            result["field_confidences"] = {
                field: {
                    "score": conf.confidence,
                    "model": conf.model_name if conf.model_name else "default"
                } for field, conf in self.confidences.items()
            }
        if self.financial_info:
            result["financial_info"] = {
                "total_charges": self.financial_info.total_charges,
                "insurance_payment": self.financial_info.insurance_payment,
                "patient_payment": self.financial_info.patient_payment,
                "balance_due": self.financial_info.balance_due,
                "copay": self.financial_info.copay,
                "deductible": self.financial_info.deductible
            }
        return result





@dataclass
class ServiceInfo:
    """Medical service information."""
    cpt_codes: List[CPTCode] = field(default_factory=list)
    icd10_codes: List[ICD10Code] = field(default_factory=list)
    service_date: Optional[str] = None  # Use ISO format YYYY-MM-DD
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
    total_charges: Optional[float] = None
    insurance_payment: Optional[float] = None
    patient_payment: Optional[float] = None
    insurance_paid: Optional[float] = None
    patient_paid: Optional[float] = None
    total_charge: Optional[float] = None
    adjustments: Optional[float] = None
    balance_due: Optional[float] = None
    copay: Optional[float] = None
    deductible: Optional[float] = None
    confidences: Dict[str, FieldConfidence] = field(default_factory=dict)


@dataclass
class SuperbillDocument:
    """Complete superbill document."""
    document_id: str
    patients: List[PatientData] = field(default_factory=list)
    services: List[ServiceInfo] = field(default_factory=list)
    provider: Optional[ProviderInfo] = None
    financial: Optional[FinancialInfo] = None
    document_date: Optional[str] = None  # ISO format YYYY-MM-DD
    confidence: float = 0.0
