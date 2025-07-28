#!/usr/bin/env python3
"""
Test script for Model Loading Fixes

This script tests the fixes implemented for Nanonets and NuExtract model loading.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger, setup_logger
from src.processors.nanonets_ocr_engine import NanonetsOCREngine
from src.extractors.nuextract_engine import NuExtractEngine
from src.processors.resource_manager import ResourceManager
from src.processors.ocr_engine import UnifiedOCREngine


async def test_nanonets_loading():
    """Test Nanonets OCR model loading."""
    print("🔍 Testing Nanonets OCR model loading...")
    
    try:
        config = ConfigManager()
        engine = NanonetsOCREngine(config)
        
        # Test model validation
        print("  📋 Validating model files...")
        if engine._validate_model_files():
            print("  ✅ Model files validation passed")
        else:
            print("  ❌ Model files validation failed")
            return False
        
        # Test model loading
        print("  📥 Loading Nanonets model...")
        await engine.load_models()
        
        if engine.models_loaded:
            print("  ✅ Nanonets model loaded successfully")
            return True
        else:
            print("  ❌ Nanonets model loading failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Nanonets test failed: {e}")
        return False


async def test_nuextract_loading():
    """Test NuExtract model loading."""
    print("🔍 Testing NuExtract model loading...")
    
    try:
        config = ConfigManager()
        engine = NuExtractEngine(config)
        
        # Test model validation
        print("  📋 Validating model files...")
        model_path = Path(config.get_model_path(engine.model_name))
        if engine._validate_model_files(model_path):
            print("  ✅ Model files validation passed")
        else:
            print("  ❌ Model files validation failed")
            return False
        
        # Test model loading
        print("  📥 Loading NuExtract model...")
        await engine.load_model()
        
        if engine.model is not None and engine.processor is not None:
            print("  ✅ NuExtract model loaded successfully")
            return True
        else:
            print("  ❌ NuExtract model loading failed")
            return False
            
    except Exception as e:
        print(f"  ❌ NuExtract test failed: {e}")
        return False


async def test_resource_manager():
    """Test resource manager functionality."""
    print("🔍 Testing resource manager...")
    
    try:
        config = ConfigManager()
        resource_manager = ResourceManager(config)
        
        # Test device selection
        print("  🔧 Testing device selection...")
        device = resource_manager._select_device()
        print(f"  ✅ Selected device: {device}")
        
        # Test memory monitoring
        print("  💾 Testing memory monitoring...")
        memory_usage = resource_manager._get_memory_usage(device)
        total_memory = resource_manager._get_total_memory(device)
        print(f"  ✅ Memory usage: {memory_usage / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Resource manager test failed: {e}")
        return False


async def test_unified_ocr_engine():
    """Test unified OCR engine."""
    print("🔍 Testing unified OCR engine...")
    
    try:
        config = ConfigManager()
        engine = UnifiedOCREngine(config)
        
        # Test engine initialization
        print("  🔧 Testing engine initialization...")
        available_engines = engine.get_available_engines()
        print(f"  ✅ Available engines: {available_engines}")
        
        # Test model loading
        print("  📥 Loading models...")
        await engine.load_models()
        
        # Check loaded engines
        loaded_engines = engine.get_available_engines()
        print(f"  ✅ Loaded engines: {loaded_engines}")
        
        if len(loaded_engines) > 0:
            print("  ✅ Unified OCR engine test passed")
            return True
        else:
            print("  ❌ No engines loaded")
            return False
            
    except Exception as e:
        print(f"  ❌ Unified OCR engine test failed: {e}")
        return False


async def test_end_to_end():
    """Test end-to-end processing with sample text."""
    print("🔍 Testing end-to-end processing...")
    
    try:
        config = ConfigManager()
        
        # Test NuExtract with sample text
        nuextract_engine = NuExtractEngine(config)
        await nuextract_engine.load_model()
        
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
        
        print("  📝 Testing text extraction...")
        result = await nuextract_engine.extract_structured_data(sample_text)
        
        if result and "patients" in result:
            print("  ✅ Text extraction successful")
            print(f"  📊 Found {len(result['patients'])} patients")
            return True
        else:
            print("  ❌ Text extraction failed")
            return False
            
    except Exception as e:
        print(f"  ❌ End-to-end test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🏥 Medical Superbill Extractor - Model Loading Test")
    print("=" * 60)
    
    # Setup logging
    setup_logger("INFO")
    
    tests = [
        ("Nanonets OCR Loading", test_nanonets_loading),
        ("NuExtract Loading", test_nuextract_loading),
        ("Resource Manager", test_resource_manager),
        ("Unified OCR Engine", test_unified_ocr_engine),
        ("End-to-End Processing", test_end_to_end),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model loading fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the logs for details.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 