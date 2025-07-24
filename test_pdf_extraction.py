#!/usr/bin/env python
"""
Test PDF extraction with the fixed system
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.extraction_engine import ExtractionEngine
from src.core.logger import get_logger

logger = get_logger(__name__)

async def test_pdf_extraction():
    """Test PDF extraction without fallbacks."""
    print("Testing PDF Extraction with Fixed System")
    print("=" * 50)
    
    # Use available PDF file for testing
    pdf_path = "superbills/Olivares.OV.04.10.2025-04.29.2025-done.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF file not found at {pdf_path}")
        print("Available PDF files:")
        for pdf_file in Path("superbills").glob("*.pdf"):
            print(f"  - {pdf_file}")
        return False
    
    try:
        # Initialize extraction engine
        print("Initializing extraction engine...")
        engine = ExtractionEngine()
        
        # Test PDF extraction
        print(f"Processing PDF: {pdf_path}")
        results = await engine.extract_from_file(pdf_path)
        
        print(f"\nExtraction Results:")
        print(f"- Success: {results.success}")
        print(f"- Total patients: {results.total_patients}")
        print(f"- Confidence: {results.extraction_confidence:.2%}")
        print(f"- Processing time: {results.processing_time:.2f}s")
        
        if results.patients:
            for i, patient in enumerate(results.patients):
                print(f"\nPatient {i+1}:")
                print(f"- Name: {patient.first_name} {patient.last_name}")
                print(f"- DOB: {patient.date_of_birth}")
                print(f"- Patient ID: {patient.patient_id}")
                print(f"- CPT Codes: {len(patient.cpt_codes or [])}")
                print(f"- ICD-10 Codes: {len(patient.icd10_codes or [])}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_pdf_extraction())