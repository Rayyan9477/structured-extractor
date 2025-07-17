#!/usr/bin/env python3
"""
Test script for Medical Superbill Extractor
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import torch
        print("  ✅ PyTorch")
        
        import transformers
        print("  ✅ Transformers")
        
        import streamlit
        print("  ✅ Streamlit")
        
        import pandas
        print("  ✅ Pandas")
        
        import PIL
        print("  ✅ PIL/Pillow")
        
        import cv2
        print("  ✅ OpenCV")
        
        import pytesseract
        print("  ✅ Tesseract")
        
        import yaml
        print("  ✅ PyYAML")
        
        print("  ✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_app_main():
    """Test the main application file."""
    print("\n🔍 Testing main application...")
    
    try:
        from app_main import ExtractionEngine, ConfigManager, DataExporter
        print("  ✅ Main components imported successfully")
        
        # Test configuration
        config = ConfigManager()
        print("  ✅ Configuration manager initialized")
        
        # Test extraction engine
        engine = ExtractionEngine(config)
        print("  ✅ Extraction engine initialized")
        
        # Test exporter
        exporter = DataExporter(config)
        print("  ✅ Data exporter initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_demo_text():
    """Test extraction with demo text."""
    print("\n🔍 Testing demo extraction...")
    
    try:
        import asyncio
        from app_main import ExtractionEngine, ConfigManager
        
        demo_text = """
        MEDICAL SUPERBILL
        
        Patient: Jane Doe
        DOB: 05/20/1985
        Patient ID: 67890
        Date of Service: 03/20/2024
        
        Diagnosis Codes:
        Z00.00 - General examination
        
        Procedure Codes:
        99213 - Office visit
        
        Charges: $200.00
        """
        
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        # Run extraction
        results = asyncio.run(engine.extract_from_text(demo_text))
        
        if results.success:
            print("  ✅ Demo extraction successful!")
            print(f"  📊 Patients found: {results.total_patients}")
            print(f"  📈 Confidence: {results.confidence_score:.1%}")
            
            if results.patients:
                patient = results.patients[0]
                print(f"  👤 Patient: {patient.first_name} {patient.last_name}")
                print(f"  🆔 ID: {patient.patient_id}")
                print(f"  📅 Service Date: {patient.date_of_service}")
                print(f"  💰 Charges: ${patient.charges}")
                print(f"  🏥 CPT Codes: {[c.code for c in patient.cpt_codes]}")
                print(f"  📋 ICD-10 Codes: {[c.code for c in patient.icd10_codes]}")
        else:
            print("  ❌ Demo extraction failed")
            for error in results.errors:
                print(f"    Error: {error}")
        
        return results.success
        
    except Exception as e:
        print(f"  ❌ Error during demo: {e}")
        return False

def main():
    """Run all tests."""
    print("🏥 Medical Superbill Extractor - System Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return False
    
    # Test main application
    app_ok = test_app_main()
    
    if not app_ok:
        print("\n❌ Application tests failed.")
        return False
    
    # Test demo extraction
    demo_ok = test_demo_text()
    
    if not demo_ok:
        print("\n❌ Demo extraction failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All tests passed! The application is ready to use.")
    print("\n📚 Usage options:")
    print("  python app_main.py                    # Launch Streamlit UI")
    print("  python app_main.py --cli file.pdf     # CLI mode")
    print("  python app_main.py --demo             # Demo mode")
    print("  python app_main.py --help             # Show help")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
