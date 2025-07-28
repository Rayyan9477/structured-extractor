#!/usr/bin/env python3
"""
Integration test for the Medical Superbill Extraction System

This script tests the core functionality without requiring actual model downloads
or processing real files. It verifies that all components can be initialized and
work together correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def test_core_components():
    """Test that all core components can be imported and initialized."""
    print("Testing core component imports...")
    
    try:
        from src.extraction_engine import ExtractionEngine
        from src.core.config_manager import ConfigManager
        from src.core.data_schema import PatientData, CPTCode, ICD10Code, ExtractionResults
        print("OK Core imports successful")
        
        # Test configuration
        config = ConfigManager()
        print("OK Configuration manager initialized")
        
        # Test engine initialization
        engine = ExtractionEngine(config)
        print("OK Extraction engine initialized")
        
        # Test data models
        patient = PatientData(
            first_name="John",
            last_name="Doe",
            date_of_birth="1980-01-15",
            patient_id="123456"
        )
        print("OK PatientData model working")
        
        cpt = CPTCode(code="99213", description="Office visit", confidence=0.85)
        icd = ICD10Code(code="M54.5", description="Low back pain", confidence=0.90)
        print("OK Medical code models working")
        
        # Test extraction results structure
        results = ExtractionResults(
            patients=[patient],
            total_patients=1,
            overall_confidence=0.85,
            success=True
        )
        print("OK ExtractionResults model working")
        
        return True
        
    except Exception as e:
        print(f"FAIL Core component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_components():
    """Test that UI components can be imported."""
    print("\nTesting UI component imports...")
    
    try:
        from ui.app import run_ui_app
        from ui.components.sidebar import render_sidebar
        from ui.components.file_uploader import render_file_uploader
        from ui.components.extraction_results import render_extraction_results
        print("OK UI component imports successful")
        
        import streamlit as st
        print("OK Streamlit available")
        
        return True
        
    except Exception as e:
        print(f"FAIL UI component test failed: {e}")
        return False

def test_data_flow():
    """Test the data flow through the system."""
    print("\nTesting data flow...")
    
    try:
        from src.core.data_schema import PatientData, CPTCode, ICD10Code
        
        # Create a patient with medical codes
        patient = PatientData(
            first_name="Jane",
            last_name="Smith",
            date_of_birth="1975-03-22",
            cpt_codes=[
                CPTCode(code="99214", description="Detailed office visit", charge=150.00, confidence=0.88),
                CPTCode(code="90471", description="Immunization admin", charge=25.00, confidence=0.92)
            ],
            icd10_codes=[
                ICD10Code(code="Z23", description="Encounter for immunization", confidence=0.85)
            ],
            extraction_confidence=0.87,
            page_number=1,
            total_pages=1
        )
        
        # Test patient data serialization
        patient_dict = patient.to_dict()
        print("OK Patient data serialization working")
        
        # Verify required fields are present
        assert patient_dict['first_name'] == "Jane"
        assert patient_dict['last_name'] == "Smith"
        assert len(patient_dict['cpt_codes']) == 2
        assert len(patient_dict['icd10_codes']) == 1
        print("OK Data validation working")
        
        return True
        
    except Exception as e:
        print(f"FAIL Data flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_processing_features():
    """Test that PDF processing components are properly configured."""
    print("\nTesting PDF processing features...")
    
    try:
        from src.processors.document_processor import DocumentProcessor
        from src.core.config_manager import ConfigManager
        
        config = ConfigManager()
        processor = DocumentProcessor(config)
        print("OK Document processor initialized")
        
        # Check if VLM optimization methods are available
        assert hasattr(processor, '_optimize_for_vlm'), "VLM optimization method missing"
        assert hasattr(processor, '_calculate_optimal_batch_size'), "Batch size calculation method missing"
        print("OK VLM optimization features available")
        
        return True
        
    except Exception as e:
        print(f"FAIL PDF processing test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Medical Superbill Extraction System - Integration Test")
    print("=" * 60)
    
    tests = [
        test_core_components,
        test_ui_components, 
        test_data_flow,
        test_pdf_processing_features
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The system is ready for use.")
        return 0
    else:
        print("FAILURE: Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)