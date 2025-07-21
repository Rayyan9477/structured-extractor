"""
Unified Structured Extraction System

This package provides a modular system for extracting structured data from documents
using multiple OCR models and NuExtract.

Main components:
- unified_extraction_system: Main extraction system that orchestrates all components
- processors: Document processing and OCR engines
- extractors: Structured data extraction tools
- exporters: Export formatted results
"""

__version__ = "1.0.0"

from src.unified_extraction_system import (
    UnifiedExtractionSystem,
    extract_from_file,
    extract_from_text,
    batch_extract
)

# Direct imports for convenience
__all__ = [
    'UnifiedExtractionSystem',
    'extract_from_file',
    'extract_from_text',
    'batch_extract',
]
