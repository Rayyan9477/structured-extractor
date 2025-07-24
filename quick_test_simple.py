"""
Simple test to verify text extraction without OCR
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.extractors.nuextract_engine import NuExtractEngine
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger

logger = get_logger(__name__)

async def test_nuextract():
    """Test NuExtract directly."""
    print("Testing NuExtract Engine")
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
        # Initialize NuExtract engine
        print("Initializing NuExtract engine...")
        config = ConfigManager()
        engine = NuExtractEngine(config)
        
        # Test text extraction
        print("Extracting data from sample text...")
        patient_data = await engine.extract_patient_data(sample_text)
        
        if patient_data:
            print(f"SUCCESS! Extracted patient data:")
            print(f"- Name: {patient_data.first_name} {patient_data.last_name}")
            print(f"- DOB: {patient_data.date_of_birth}")
            print(f"- Patient ID: {patient_data.patient_id}")
            print(f"- CPT Codes: {len(patient_data.cpt_codes or [])}")
            print(f"- ICD-10 Codes: {len(patient_data.icd10_codes or [])}")
            
            if patient_data.cpt_codes:
                print("CPT Codes found:")
                for cpt in patient_data.cpt_codes:
                    print(f"  - {cpt.code}: {cpt.description}")
                    
            if patient_data.icd10_codes:
                print("ICD-10 Codes found:")
                for icd in patient_data.icd10_codes:
                    print(f"  - {icd.code}: {icd.description}")
        else:
            print("No patient data extracted")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_nuextract())