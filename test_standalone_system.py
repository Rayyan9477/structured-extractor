"""
Test Script for Medical Superbill Extraction System

Tests the complete pipeline with standalone local models.
"""

import asyncio
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.extraction_engine import ExtractionPipeline
from src.processors.ocr_engine import UnifiedOCREngine
from src.extractors.nuextract_engine import NuExtractEngine


async def test_individual_models():
    """Test each model individually."""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL MODELS")
    print("="*60)
    
    config = ConfigManager()
    logger = get_logger(__name__)
    
    # Create test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    )
    
    # Test OCR Engine
    print("\n🔍 Testing OCR Engine...")
    try:
        ocr_engine = UnifiedOCREngine(config)
        await ocr_engine.load_models()
        
        available_engines = ocr_engine.get_available_engines()
        print(f"   Available engines: {available_engines}")
        
        result = await ocr_engine.extract_text(test_image)
        print(f"   ✓ OCR completed - Model: {result.model_name}")
        print(f"   ✓ Text length: {len(result.text)} chars")
        print(f"   ✓ Confidence: {result.confidence:.2f}")
        print(f"   ✓ Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"   ✗ OCR test failed: {e}")
        return False
    
    # Test NuExtract Engine
    print("\n🎯 Testing NuExtract Engine...")
    try:
        nuextract_engine = NuExtractEngine(config)
        await nuextract_engine.load_model()
        
        test_text = """
        Patient: John Doe
        Date of Birth: 1980-01-15
        Date of Service: 2024-03-15
        CPT Code: 99213 - Office Visit
        ICD-10: M54.5 - Low back pain
        Total Charge: $150.00
        """
        
        result = await nuextract_engine.extract_structured_data(
            test_text, 
            template_name="medical_superbill"
        )
        
        print(f"   ✓ Extraction completed")
        print(f"   ✓ Extracted fields: {len(result)}")
        print(f"   ✓ Sample result: {str(result)[:200]}...")
        
    except Exception as e:
        print(f"   ✗ NuExtract test failed: {e}")
        return False
    
    return True


async def test_full_pipeline():
    """Test the complete extraction pipeline."""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    config = ConfigManager()
    logger = get_logger(__name__)
    
    # Test with a sample medical document image
    print("\n📄 Testing with sample document...")
    
    # Look for a test image
    test_image_paths = [
        "sample_medical_document.png",
        "test_medical_document.png",
        "sample_handwritten_note.png"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if Path(path).exists():
            test_image_path = path
            break
    
    if not test_image_path:
        print("   ⚠️  No test image found, creating synthetic test...")
        # Create a synthetic test image with text
        test_image = Image.new('RGB', (800, 600), 'white')
        # In a real scenario, you'd add text to this image
        
        pipeline = ExtractionPipeline(config)
        
        try:
            # Test OCR on synthetic image
            images = [test_image]
            results = await pipeline.ocr_engine.extract_text_batch(images)
            
            print(f"   ✓ OCR processed {len(results)} images")
            for i, result in enumerate(results):
                print(f"   ✓ Image {i+1}: {len(result.text)} chars, confidence: {result.confidence:.2f}")
            
            # Test structured extraction
            if results and results[0].text:
                structured_data = await pipeline.nuextract_engine.extract_structured_data(
                    results[0].text,
                    template_name="medical_superbill"
                )
                print(f"   ✓ Structured extraction: {len(structured_data)} fields")
            
        except Exception as e:
            print(f"   ✗ Pipeline test failed: {e}")
            return False
    else:
        print(f"   📎 Using test image: {test_image_path}")
        
        pipeline = ExtractionPipeline(config)
        
        try:
            results = await pipeline.extract_from_file(test_image_path)
            
            print(f"   ✓ Extraction completed")
            print(f"   ✓ Found {len(results.patients)} patients")
            
            for i, patient in enumerate(results.patients):
                print(f"   ✓ Patient {i+1}: {patient.patient_info.first_name if patient.patient_info else 'Unknown'}")
            
        except Exception as e:
            print(f"   ✗ Full pipeline test failed: {e}")
            return False
    
    return True


async def run_performance_test():
    """Run basic performance tests."""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)
    
    config = ConfigManager()
    
    # Create multiple test images
    test_images = []
    for i in range(3):
        test_image = Image.fromarray(
            np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        )
        test_images.append(test_image)
    
    print(f"\n⏱️  Testing batch processing with {len(test_images)} images...")
    
    try:
        ocr_engine = UnifiedOCREngine(config)
        await ocr_engine.load_models()
        
        import time
        start_time = time.time()
        
        results = await ocr_engine.extract_text_batch(test_images)
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_images)
        
        print(f"   ✓ Processed {len(results)} images")
        print(f"   ✓ Total time: {total_time:.2f}s")
        print(f"   ✓ Average time per image: {avg_time:.2f}s")
        print(f"   ✓ Throughput: {len(test_images) / total_time:.2f} images/second")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Performance test failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    import torch
    import psutil
    
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f}GB")
    
    memory = psutil.virtual_memory()
    print(f"RAM: {memory.available / 1e9:.1f}GB available / {memory.total / 1e9:.1f}GB total")
    print(f"CPU Cores: {psutil.cpu_count()}")


async def main():
    """Run all tests."""
    print("🏥 MEDICAL SUPERBILL EXTRACTION SYSTEM - TESTING")
    
    # Print system info
    print_system_info()
    
    # Run diagnostics first
    print("\n🔧 Running model diagnostics...")
    try:
        from diagnose_models import ModelDiagnostics
        diagnostics = ModelDiagnostics()
        diag_results = await diagnostics.run_full_diagnostics()
        
        # Check if models are ready
        model_loading = diag_results["model_loading"]
        models_loaded = all(info["loaded"] for info in model_loading.values())
        
        if not models_loaded:
            print("\n❌ Not all models loaded successfully. Please check diagnostics.")
            diagnostics.print_summary(diag_results)
            return 1
        else:
            print("\n✅ All models loaded successfully!")
            
    except Exception as e:
        print(f"\n⚠️  Could not run diagnostics: {e}")
    
    # Test individual models
    individual_success = await test_individual_models()
    
    # Test full pipeline
    pipeline_success = await test_full_pipeline()
    
    # Performance test
    perf_success = await run_performance_test()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Individual Models", individual_success),
        ("Full Pipeline", pipeline_success),
        ("Performance", perf_success)
    ]
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in tests)
    
    if all_passed:
        print("\n🎉 All tests passed! The system is ready for use.")
        print("\nTo run the extraction system:")
        print("  python main.py <path_to_medical_document>")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
