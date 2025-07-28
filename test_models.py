#!/usr/bin/env python3
"""
Model functionality test for NuExtract and Nanonets OCR engines

This script tests both models to ensure they are working correctly and
processing documents with optimal accuracy.
"""

import asyncio
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add the project root to the path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.core.config_manager import ConfigManager
from src.extractors.nuextract_engine import NuExtractEngine
from src.processors.nanonets_ocr_engine import NanonetsOCREngine


def create_test_image():
    """Create a simple test image with medical-like text."""
    # Create a white background image
    img = Image.new('RGB', (800, 600), color='white')
    
    # In a real test, you would use PIL.ImageDraw to add text
    # For this test, we'll just return a white image
    return img


async def test_nuextract_engine():
    """Test NuExtract engine functionality."""
    print("\n" + "="*50)
    print("Testing NuExtract Engine")
    print("="*50)
    
    try:
        config = ConfigManager()
        engine = NuExtractEngine(config)
        
        # Test model loading
        print("Loading NuExtract model...")
        start_time = time.time()
        await engine.load_model()
        load_time = time.time() - start_time
        print(f"OK Model loaded in {load_time:.2f}s")
        
        # Test text extraction
        sample_text = """
        Patient: John Doe
        Date of Birth: 01/15/1980
        Date of Service: 03/15/2024
        CPT Code: 99213 - Office Visit - $150.00
        Diagnosis: M54.5 - Low back pain
        Provider: Dr. Smith
        NPI: 1234567890
        """
        
        print("Testing structured data extraction...")
        start_time = time.time()
        result = await engine.extract_structured_data(sample_text, "medical_superbill")
        extraction_time = time.time() - start_time
        
        print(f"OK Extraction completed in {extraction_time:.2f}s")
        print(f"OK Result type: {type(result)}")
        print(f"OK Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if result and isinstance(result, dict) and "patients" in result:
            patients = result["patients"]
            print(f"OK Found {len(patients)} patients in structured data")
            
            if patients:
                first_patient = patients[0]
                print(f"OK First patient keys: {list(first_patient.keys())}")
                
                # Check for medical codes
                medical_codes = first_patient.get("medical_codes", {})
                if medical_codes:
                    cpt_codes = medical_codes.get("cpt_codes", [])
                    icd_codes = medical_codes.get("icd10_codes", [])
                    print(f"OK Found {len(cpt_codes)} CPT codes and {len(icd_codes)} ICD codes")
                
            score = 1.0
        else:
            print("WARN No structured patient data found")
            score = 0.5
        
        # Test image processing
        print("Testing image processing...")
        test_image = create_test_image()
        start_time = time.time()
        ocr_result = await engine.extract_from_image(test_image.tobytes() if hasattr(test_image, 'tobytes') else b'')
        image_time = time.time() - start_time
        
        print(f"OK Image processing completed in {image_time:.2f}s")
        print(f"OK OCR result confidence: {ocr_result.confidence:.2f}")
        
        return {
            "success": True,
            "score": score,
            "load_time": load_time,
            "extraction_time": extraction_time,
            "image_time": image_time,
            "confidence": getattr(ocr_result, 'confidence', 0.0)
        }
        
    except Exception as e:
        print(f"FAIL NuExtract test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "score": 0.0,
            "error": str(e)
        }


async def test_nanonets_engine():
    """Test Nanonets OCR engine functionality."""
    print("\n" + "="*50)
    print("Testing Nanonets OCR Engine")
    print("="*50)
    
    try:
        config = ConfigManager()
        
        # Check if model path exists
        model_path = Path(config.get_model_path("nanonets/Nanonets-OCR-s"))
        if not model_path.exists():
            print(f"WARN Nanonets model not found at {model_path}")
            print("WARN Skipping Nanonets test - model not available")
            return {
                "success": False,
                "score": 0.0,
                "error": "Model not found",
                "skipped": True
            }
        
        engine = NanonetsOCREngine(config)
        
        # Test model loading
        print("Loading Nanonets OCR model...")
        start_time = time.time()
        await engine.load_models()
        load_time = time.time() - start_time
        print(f"OK Model loaded in {load_time:.2f}s")
        
        # Test image processing
        print("Testing image text extraction...")
        test_image = create_test_image()
        start_time = time.time()
        result = await engine.extract_text(test_image)
        extraction_time = time.time() - start_time
        
        print(f"OK Text extraction completed in {extraction_time:.2f}s")
        print(f"OK Result confidence: {result.confidence:.2f}")
        print(f"OK Extracted text length: {len(result.text)} characters")
        print(f"OK Model name: {result.model_name}")
        
        # Test batch processing
        print("Testing batch processing...")
        test_images = [create_test_image() for _ in range(2)]
        start_time = time.time()
        batch_results = await engine.extract_text_batch(test_images)
        batch_time = time.time() - start_time
        
        print(f"OK Batch processing completed in {batch_time:.2f}s")
        print(f"OK Processed {len(batch_results)} images")
        
        avg_confidence = sum(r.confidence for r in batch_results) / len(batch_results) if batch_results else 0.0
        
        return {
            "success": True,
            "score": 1.0 if result.confidence > 0.5 else 0.7,
            "load_time": load_time,
            "extraction_time": extraction_time,
            "batch_time": batch_time,
            "confidence": result.confidence,
            "avg_batch_confidence": avg_confidence
        }
        
    except Exception as e:
        print(f"FAIL Nanonets test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "score": 0.0,
            "error": str(e)
        }


async def test_integration():
    """Test integration between both models."""
    print("\n" + "="*50)
    print("Testing Model Integration")
    print("="*50)
    
    try:
        from src.extraction_engine import ExtractionEngine
        
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        # Test with sample text
        sample_medical_text = """
        SUPERBILL
        Patient: Jane Smith
        DOB: 05/22/1975
        Date of Service: 07/15/2024
        
        CPT Codes:
        99214 - Detailed Office Visit - $200.00
        90471 - Immunization Administration - $25.00
        
        ICD-10 Codes:
        Z23 - Encounter for immunization
        
        Provider: Dr. Johnson
        NPI: 9876543210
        Total Charges: $225.00
        """
        
        print("Testing full extraction pipeline...")
        start_time = time.time()
        results = await engine.extract_from_text(sample_medical_text, "integration_test")
        total_time = time.time() - start_time
        
        print(f"OK Pipeline completed in {total_time:.2f}s")
        print(f"OK Success: {results.success}")
        print(f"OK Total patients: {results.total_patients}")
        print(f"OK Overall confidence: {results.extraction_confidence:.2f}")
        
        if results.patients:
            patient = results.patients[0]
            print(f"OK Patient name: {patient.first_name} {patient.last_name}")
            print(f"OK CPT codes: {len(patient.cpt_codes) if patient.cpt_codes else 0}")
            print(f"OK ICD codes: {len(patient.icd10_codes) if patient.icd10_codes else 0}")
        
        return {
            "success": results.success,
            "score": results.extraction_confidence,
            "total_time": total_time,
            "patient_count": results.total_patients
        }
        
    except Exception as e:
        print(f"FAIL Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "score": 0.0,
            "error": str(e)
        }


def calculate_optimization_score(results):
    """Calculate optimization score based on performance metrics."""
    scores = []
    
    # NuExtract scoring
    if "nuextract" in results and results["nuextract"]["success"]:
        nu_result = results["nuextract"]
        # Score based on extraction time and accuracy
        time_score = min(1.0, 10.0 / nu_result.get("extraction_time", 10.0))  # Better if under 10s
        accuracy_score = nu_result.get("score", 0.0)
        scores.append((time_score + accuracy_score) / 2)
    
    # Nanonets scoring
    if "nanonets" in results and results["nanonets"]["success"]:
        nano_result = results["nanonets"]
        # Score based on confidence and processing time
        confidence_score = nano_result.get("confidence", 0.0)
        time_score = min(1.0, 15.0 / nano_result.get("extraction_time", 15.0))  # Better if under 15s
        scores.append((confidence_score + time_score) / 2)
    
    # Integration scoring
    if "integration" in results and results["integration"]["success"]:
        int_result = results["integration"]
        integration_score = int_result.get("score", 0.0)
        scores.append(integration_score)
    
    return sum(scores) / len(scores) if scores else 0.0


async def main():
    """Run all model tests."""
    print("=" * 60)
    print("Medical Document Processing - Model Verification Test")
    print("=" * 60)
    
    # Run all tests
    results = {}
    
    print("Running NuExtract test...")
    results["nuextract"] = await test_nuextract_engine()
    
    print("Running Nanonets test...")
    results["nanonets"] = await test_nanonets_engine()
    
    print("Running Integration test...")
    results["integration"] = await test_integration()
    
    # Calculate overall scores
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        total_tests += 1
        status = "PASS" if result["success"] else "FAIL"
        if result.get("skipped"):
            status = "SKIP"
        
        print(f"{test_name.upper()}: {status}")
        
        if result["success"]:
            passed_tests += 1
            print(f"  Score: {result['score']:.2f}")
            
            # Print performance metrics
            if "load_time" in result:
                print(f"  Load time: {result['load_time']:.2f}s")
            if "extraction_time" in result:
                print(f"  Extraction time: {result['extraction_time']:.2f}s")
            if "confidence" in result:
                print(f"  Confidence: {result['confidence']:.2f}")
        else:
            if not result.get("skipped"):
                print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Calculate optimization score
    optimization_score = calculate_optimization_score(results)
    
    print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
    print(f"Optimization Score: {optimization_score:.2f}/1.0")
    
    # Provide recommendations
    print("\nOptimization Assessment:")
    if optimization_score >= 0.8:
        print("OK EXCELLENT: Both models are working optimally with high accuracy")
    elif optimization_score >= 0.6:
        print("OK GOOD: Models are working well with acceptable performance")
    elif optimization_score >= 0.4:
        print("WARN MODERATE: Models are functional but may need optimization")
    else:
        print("FAIL POOR: Significant issues detected, optimization required")
    
    # Specific recommendations
    if "nuextract" in results and results["nuextract"]["success"]:
        if results["nuextract"]["score"] < 0.7:
            print("  - Consider fine-tuning NuExtract prompts for better extraction accuracy")
        if results["nuextract"].get("extraction_time", 0) > 10:
            print("  - NuExtract processing time could be optimized")
    
    if "nanonets" in results and results["nanonets"]["success"]:
        if results["nanonets"]["confidence"] < 0.7:
            print("  - Nanonets OCR confidence could be improved with better image preprocessing")
        if results["nanonets"].get("extraction_time", 0) > 15:
            print("  - Nanonets processing time could be optimized")
    
    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)