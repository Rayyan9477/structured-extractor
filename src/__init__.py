"""
Medical Superbill Data Extraction System

A comprehensive Python project designed to automate the extraction of structured data
from medical superbills using advanced OCR and NLP models.
"""

from .core.config_manager import ConfigManager
from .processors.document_processor import DocumentProcessor
from .processors.ocr_engine import OCREngine
from .extractors.field_detector import FieldDetectionEngine
from .extractors.nuextract_engine import NuExtractEngine
from .extractors.multi_patient_handler import MultiPatientHandler
from .validators.date_validator import DateValidator
from .validators.data_validator import DataValidator
from .exporters.data_exporter import DataExporter
from .extraction_engine import ExtractionEngine

__version__ = "1.0.0"
__author__ = "Medical Superbill Extraction Team"
__description__ = "Automated medical superbill data extraction system"

__all__ = [
    # Core components
    'ConfigManager',
    
    # Processing components
    'DocumentProcessor',
    'OCREngine',
    
    # Extraction components
    'FieldDetectionEngine',
    'NuExtractEngine',
    'MultiPatientHandler',
    
    # Validation components
    'DataValidator',
    'DateValidator',
    
    # Export components
    'DataExporter',
    
    # Main engine
    'ExtractionEngine',
]
