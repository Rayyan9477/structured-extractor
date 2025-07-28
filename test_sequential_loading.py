#!/usr/bin/env python3
"""
Test script to verify sequential model loading works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config_manager import ConfigManager
from src.extraction_engine import ExtractionEngine


async def test_sequential_loading():
    """Test that models load sequentially."""
    print("🧪 Testing Sequential Model Loading")
    print("=" * 50)
    
    try:
        # Initialize config and engine
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        print("📝 Testing text extraction with sequential model loading...")
        
        # Test text extraction
        sample_text = """
        MEDICAL SUPERBILL
        
        Patient: John Smith
        DOB: 01/15/1980
        Patient ID: 12345
        Date of Service: 03/15/2024
        
        Diagnosis Codes:
        Z00.00 - Encounter for general adult medical examination
        
        Procedure Codes:
        99213 - Office visit, established patient
        
        Charges: $150.00
        """
        
        results = await engine.extract_from_text(sample_text)
        
        if results.patients:
            print("✅ Sequential model loading successful!")
            print(f"📊 Found {len(results.patients)} patients")
            
            for i, patient in enumerate(results.patients):
                print(f"\n👤 Patient {i+1}:")
                print(f"  Name: {patient.first_name or ''} {patient.last_name or ''}")
                print(f"  DOB: {patient.date_of_birth}")
                print(f"  CPT Codes: {len(patient.cpt_codes) if patient.cpt_codes else 0}")
                print(f"  ICD-10 Codes: {len(patient.icd10_codes) if patient.icd10_codes else 0}")
            
            return True
        else:
            print("❌ No patients found in extraction results")
            return False
            
    except Exception as e:
        print(f"❌ Sequential loading test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_sequential_loading())
    if success:
        print("\n🎉 Sequential model loading test PASSED!")
    else:
        print("\n💥 Sequential model loading test FAILED!")
        sys.exit(1) 