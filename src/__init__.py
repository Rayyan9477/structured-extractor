"""
Medical Superbill Data Extraction System

A comprehensive Python project designed to automate the extraction of structured data
from medical superbills using advanced OCR and NLP models.
"""

from .core import ConfigManager, Logger, DataSchema
from .processors import DocumentProcessor, OCREngine
from .models import ModelManager
from .extractors import (
    FieldDetector, 
    NuExtractEngine, 
    MultiPatientHandler,
    FieldType,
    DetectionResult,
    BoundaryType,
    PatientBoundary,
    PatientSegment
)
from .validators import (
    DataValidator,
    CPTCodeValidator,
    ICD10CodeValidator,
    DateValidator,
    PHIAnonymizer
)
from .exporters import (
    DataExporter,
    CSVExporter,
    JSONExporter,
    ExcelExporter
)
from .extraction_engine import ExtractionEngine, extract_from_file, extract_from_text, extract_batch

__version__ = "1.0.0"
__author__ = "Medical Superbill Extraction Team"
__description__ = "Automated medical superbill data extraction system"

__all__ = [
    # Core components
    'ConfigManager',
    'Logger', 
    'DataSchema',
    
    # Processing components
    'DocumentProcessor',
    'OCREngine',
    'ModelManager',
    
    # Extraction components
    'FieldDetector',
    'NuExtractEngine',
    'MultiPatientHandler',
    'FieldType',
    'DetectionResult',
    'BoundaryType',
    'PatientBoundary',
    'PatientSegment',
    
    # Validation components
    'DataValidator',
    'CPTCodeValidator',
    'ICD10CodeValidator',
    'DateValidator',
    'PHIAnonymizer',
    
    # Export components
    'DataExporter',
    'CSVExporter',
    'JSONExporter',
    'ExcelExporter',
    
    # Main engine
    'ExtractionEngine',
    'extract_from_file',
    'extract_from_text', 
    'extract_batch'
]
