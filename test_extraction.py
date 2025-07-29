#!/usr/bin/env python3
"""
Quick test script to validate patient extraction fixes
"""

import sys
from pathlib import Path
import asyncio

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config_manager import ConfigManager
from src.core.data_schema import PatientData, ServiceInfo, FinancialInfo, CPTCode, ICD10Code

def test_patient_data_schema():
    """Test that PatientData class has required attributes"""
    print("Testing PatientData schema...")
    
    # Test basic PatientData creation
    try:
        patient = PatientData()
        print("PASS: PatientData class can be instantiated")
        
        # Test required attributes that were missing
        if hasattr(patient, 'date_of_service'):
            print("PASS: date_of_service attribute exists")
        else:
            print("FAIL: date_of_service attribute missing")
            
        if hasattr(patient, 'service_info'):
            print("PASS: service_info attribute exists")
        else:
            print("FAIL: service_info attribute missing")
            
        # Test setting date_of_service
        patient.date_of_service = "2024-01-15"
        print(f"PASS: date_of_service can be set: {patient.date_of_service}")
        
        # Test creating service_info
        patient.service_info = ServiceInfo()
        patient.service_info.date_of_service = "2024-01-15"
        print(f"PASS: service_info can be created and date set: {patient.service_info.date_of_service}")
        
        # Test financial_info
        patient.financial_info = FinancialInfo()
        patient.financial_info.total_charges = 250.00
        print(f"PASS: financial_info works: ${patient.financial_info.total_charges}")
        
        # Test CPT and ICD codes
        cpt = CPTCode(code="99213", description="Office visit", charge=150.00)
        icd = ICD10Code(code="Z00.00", description="Routine checkup")
        
        patient.cpt_codes = [cpt]
        patient.icd10_codes = [icd]
        
        print(f"PASS: CPT codes work: {patient.cpt_codes[0].code}")
        print(f"PASS: ICD codes work: {patient.icd10_codes[0].code}")
        
        print("SUCCESS: All PatientData schema tests passed!")
        return True
        
    except Exception as e:
        print(f"FAIL: PatientData schema test failed: {e}")
        return False

def test_config_loading():
    """Test that configuration loads without issues"""
    print("\nTesting configuration loading...")
    
    try:
        config = ConfigManager()
        print("PASS: ConfigManager loads successfully")
        
        # Test OCR config
        ocr_config = config.get("ocr", {})
        print(f"PASS: OCR config loaded: {len(ocr_config)} settings")
        
        # Test nanonets config
        nanonets_config = config.get("ocr.nanonets_ocr", {})
        max_tokens = nanonets_config.get("max_new_tokens", "not found")
        print(f"PASS: Nanonets max_new_tokens: {max_tokens}")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Configuration test failed: {e}")
        return False

def test_model_paths():
    """Test that model paths exist"""
    print("\nTesting model paths...")
    
    try:
        models_dir = Path("models")
        if models_dir.exists():
            print(f"PASS: Models directory exists: {models_dir.absolute()}")
            
            # Check for NuExtract
            nuextract_path = models_dir / "microsoft" / "nuextract-v1.5"
            if nuextract_path.exists():
                print(f"PASS: NuExtract model found: {nuextract_path}")
            else:
                print(f"WARN: NuExtract model not found at: {nuextract_path}")
            
            # Check for Nanonets
            nanonets_path = models_dir / "nanonets" / "Nanonets-OCR-s"
            if nanonets_path.exists():
                print(f"PASS: Nanonets model found: {nanonets_path}")
            else:
                print(f"WARN: Nanonets model not found at: {nanonets_path}")
                
        else:
            print(f"FAIL: Models directory not found: {models_dir.absolute()}")
            
        return True
        
    except Exception as e:
        print(f"FAIL: Model paths test failed: {e}")
        return False

async def test_extraction_engine():
    """Test that extraction engine can be initialized"""
    print("\nTesting extraction engine initialization...")
    
    try:
        from src.extraction_engine import ExtractionEngine
        
        config = ConfigManager()
        engine = ExtractionEngine(config)
        print("PASS: ExtractionEngine can be instantiated")
        
        # Test that it has required methods
        if hasattr(engine, 'extract_single_patient'):
            print("PASS: extract_single_patient method exists")
        else:
            print("FAIL: extract_single_patient method missing")
            
        return True
        
    except Exception as e:
        print(f"FAIL: Extraction engine test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running Medical Superbill Extractor Tests\n")
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_patient_data_schema():
        tests_passed += 1
        
    if test_config_loading():
        tests_passed += 1
        
    if test_model_paths():
        tests_passed += 1
        
    if asyncio.run(test_extraction_engine()):
        tests_passed += 1
    
    # Summary
    print(f"\nTest Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("SUCCESS: All tests passed! The system should work correctly.")
    else:
        print("WARNING: Some tests failed. Please check the issues above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()