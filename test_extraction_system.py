"""
Test script to verify the extraction system is working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.extraction_engine import ExtractionEngine
from src.core.logger import get_logger

logger = get_logger(__name__)

async def test_extraction_system():
    """Test the extraction system with sample text."""
    print("🧪 Testing Medical Extraction System")
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
        print("📋 Initializing extraction engine...")
        engine = ExtractionEngine()
        
        # Test text extraction
        print("🔍 Extracting data from sample text...")
        results = await engine.extract_from_text(sample_text, "test_sample")
        
        # Print results
        print(f"\n✅ Extraction completed!")
        print(f"   Total patients: {results.total_patients}")
        print(f"   Confidence: {results.extraction_confidence:.2f}")
        
        if results.patients:
            patient = results.patients[0]
            print(f"\n👤 Patient Information:")
            print(f"   Name: {patient.first_name} {patient.last_name}")
            print(f"   DOB: {patient.date_of_birth}")
            print(f"   Service Date: {patient.date_of_service}")
            
            if patient.cpt_codes:
                print(f"\n🏥 CPT Codes:")
                for cpt in patient.cpt_codes:
                    print(f"   {cpt.code}: {cpt.description}")
            
            if patient.icd10_codes:
                print(f"\n🏥 ICD-10 Codes:")
                for icd in patient.icd10_codes:
                    print(f"   {icd.code}: {icd.description}")
            
            if patient.financial_info:
                print(f"\n💰 Financial Info:")
                print(f"   Total Charges: ${patient.financial_info.total_charges or 0:.2f}")
        
        print(f"\n✨ Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction_system())
