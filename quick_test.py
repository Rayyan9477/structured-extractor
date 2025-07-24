"""
Quick test to verify model integration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.extraction_engine import ExtractionEngine
from src.core.logger import get_logger

logger = get_logger(__name__)

async def test_models():
    """Test the extraction system with sample text."""
    print("Testing Medical Extraction System")
    print("=" * 50)
    
    # Sample medical text
    sample_text = """
    MEDICAL SUPERBILL
    
    Patient Name: John Smith
    Date of Birth: 01/15/1980
    Patient ID: PAT001
    Date of Service: 12/15/2024
    Provider: Dr. Sarah Johnson, MD
    
    Services Provided:
    99213 - Office Visit (Established Patient) - $150.00
    87430 - Strep Test - $25.00
    93000 - Electrocardiogram - $75.00
    
    Diagnosis Codes:
    J06.9 - Acute upper respiratory infection
    I10 - Essential hypertension
    
    Total Charges: $250.00
    """
    
    try:
        # Initialize engine
        print("Initializing extraction engine...")
        engine = ExtractionEngine()
        
        # Test text extraction
        print("Extracting data from sample text...")
        results = await engine.extract_from_text(sample_text)
        
        print(f"Extraction Results:")
        print(f"- Success: {results.success}")
        print(f"- Total patients: {results.total_patients}")
        print(f"- Confidence: {results.extraction_confidence:.2%}")
        
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
    asyncio.run(test_models())