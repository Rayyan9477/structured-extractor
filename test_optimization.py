#!/usr/bin/env python3
"""
Optimization and Performance Test for Medical Document Processing

This script analyzes the optimization strategies and performance characteristics
of both NuExtract and Nanonets models.
"""

import asyncio
import sys
import time
import json
import gc
import psutil
import os
from pathlib import Path
from statistics import mean, median

# Add the project root to the path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from src.core.config_manager import ConfigManager
from src.extraction_engine import ExtractionEngine


def get_system_info():
    """Get system information for optimization analysis."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total // (1024**3),  # GB
        "memory_available": psutil.virtual_memory().available // (1024**3),  # GB
        "gpu_available": False,
        "gpu_memory": 0
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
    except ImportError:
        pass
    
    return info


async def test_memory_usage():
    """Test memory usage and optimization."""
    print("\n" + "="*50)
    print("Memory Usage and Optimization Test")
    print("="*50)
    
    process = psutil.Process(os.getpid())
    
    # Baseline memory usage
    baseline_memory = process.memory_info().rss / (1024**2)  # MB
    print(f"Baseline memory usage: {baseline_memory:.1f} MB")
    
    try:
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        # Memory after engine initialization
        init_memory = process.memory_info().rss / (1024**2)
        print(f"Memory after engine init: {init_memory:.1f} MB (+{init_memory - baseline_memory:.1f} MB)")
        
        # Load models and measure memory
        print("Loading models...")
        start_time = time.time()
        await engine.pipeline._initialize_models()
        load_time = time.time() - start_time
        
        models_memory = process.memory_info().rss / (1024**2)
        print(f"Memory after model loading: {models_memory:.1f} MB (+{models_memory - init_memory:.1f} MB)")
        print(f"Model loading time: {load_time:.2f}s")
        
        # Test processing with different text sizes
        test_texts = {
            "small": "Patient: John Doe\nDOB: 01/15/1980\nCPT: 99213",
            "medium": """
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
            """ * 3,  # Medium text
            "large": """
            MEDICAL SUPERBILL - COMPREHENSIVE VISIT
            
            Patient Information:
            Name: Robert Williams III
            Date of Birth: 03/15/1965
            Patient ID: MW12345678
            Address: 123 Main Street, Anytown, ST 12345
            Phone: (555) 123-4567
            Email: robert.williams@email.com
            
            Insurance Information:
            Primary Insurance: BlueCross BlueShield
            Policy Number: BC987654321
            Group Number: 12345
            Subscriber ID: MW12345678
            
            Provider Information:
            Provider: Dr. Sarah Johnson, MD
            NPI: 1234567890
            Practice: Comprehensive Medical Center
            Address: 456 Healthcare Blvd, Medical City, ST 54321
            Phone: (555) 987-6543
            Tax ID: 12-3456789
            
            Service Information:
            Date of Service: 12/15/2024
            Place of Service: Office
            
            CPT Codes and Procedures:
            99214 - Office/outpatient visit, established patient, detailed - $185.00
            93000 - Electrocardiogram, routine ECG - $45.00
            85025 - Blood count; complete CBC - $25.00
            80053 - Comprehensive metabolic panel - $35.00
            90471 - Immunization administration - $15.00
            90658 - Influenza vaccine - $30.00
            
            ICD-10 Diagnosis Codes:
            Z00.00 - Encounter for general adult medical examination
            I10 - Essential hypertension
            E11.9 - Type 2 diabetes mellitus without complications
            Z23 - Encounter for immunization
            
            Financial Summary:
            Total Charges: $335.00
            Insurance Payment: $268.00
            Patient Copay: $25.00
            Patient Deductible: $42.00
            Balance Due: $0.00
            
            Additional Notes:
            Patient presents for routine annual physical examination.
            Blood pressure elevated, discussed lifestyle modifications.
            Diabetes well controlled with current medications.
            Annual influenza vaccination administered.
            Follow-up appointment scheduled in 6 months.
            """ * 2  # Large text
        }
        
        processing_times = {}
        memory_peaks = {}
        
        for size, text in test_texts.items():
            print(f"\nTesting {size} text ({len(text)} characters)...")
            
            # Clear memory before test
            gc.collect()
            
            pre_memory = process.memory_info().rss / (1024**2)
            start_time = time.time()
            
            try:
                result = await engine.extract_from_text(text, f"test_{size}")
                processing_time = time.time() - start_time
                processing_times[size] = processing_time
                
                post_memory = process.memory_info().rss / (1024**2)
                memory_peaks[size] = post_memory - pre_memory
                
                print(f"  Processing time: {processing_time:.2f}s")
                print(f"  Memory delta: +{memory_peaks[size]:.1f} MB")
                print(f"  Success: {result.success}")
                print(f"  Patients found: {result.total_patients}")
                print(f"  Confidence: {result.extraction_confidence:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                processing_times[size] = float('inf')
                memory_peaks[size] = 0
        
        return {
            "baseline_memory": baseline_memory,
            "models_memory": models_memory - baseline_memory,
            "model_load_time": load_time,
            "processing_times": processing_times,
            "memory_peaks": memory_peaks
        }
        
    except Exception as e:
        print(f"Memory test failed: {e}")
        return {"error": str(e)}


async def test_batch_processing():
    """Test batch processing optimization."""
    print("\n" + "="*50)
    print("Batch Processing Optimization Test")
    print("="*50)
    
    try:
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        # Test different batch sizes
        sample_text = """
        Patient: Test Patient {i}
        DOB: 01/01/198{i}
        CPT: 9921{i} - Office Visit - $15{i}.00
        ICD: M54.{i} - Back pain
        """
        
        batch_sizes = [1, 3, 5]
        batch_results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create text samples
            texts = [sample_text.format(i=i) for i in range(batch_size)]
            
            # Sequential processing
            start_time = time.time()
            sequential_results = []
            for i, text in enumerate(texts):
                result = await engine.extract_from_text(text, f"sequential_{i}")
                sequential_results.append(result)
            sequential_time = time.time() - start_time
            
            # Batch processing (if available)
            start_time = time.time()
            try:
                # Since we don't have a direct batch text method, simulate it
                batch_results_list = []
                for i, text in enumerate(texts):
                    result = await engine.extract_from_text(text, f"batch_{i}")
                    batch_results_list.append(result)
                batch_time = time.time() - start_time
            except Exception as e:
                print(f"  Batch processing not available: {e}")
                batch_time = sequential_time
                batch_results_list = sequential_results
            
            print(f"  Sequential time: {sequential_time:.2f}s")
            print(f"  Batch time: {batch_time:.2f}s")
            print(f"  Speedup: {sequential_time/batch_time:.2f}x")
            
            batch_results[batch_size] = {
                "sequential_time": sequential_time,
                "batch_time": batch_time,
                "speedup": sequential_time / batch_time,
                "success_rate": sum(1 for r in batch_results_list if r.success) / len(batch_results_list)
            }
        
        return batch_results
        
    except Exception as e:
        print(f"Batch processing test failed: {e}")
        return {"error": str(e)}


async def test_model_performance():
    """Test individual model performance characteristics."""
    print("\n" + "="*50)
    print("Individual Model Performance Test")
    print("="*50)
    
    try:
        config = ConfigManager()
        
        # Test NuExtract performance
        print("Testing NuExtract performance...")
        from src.extractors.nuextract_engine import NuExtractEngine
        nuextract = NuExtractEngine(config)
        
        nu_load_start = time.time()
        await nuextract.load_model()
        nu_load_time = time.time() - nu_load_start
        
        # Test extraction times
        test_text = """
        Patient: Alice Johnson
        DOB: 12/05/1978
        Date of Service: 11/20/2024
        CPT: 99213 - Office Visit - $150.00
        ICD: M79.3 - Panniculitis
        """
        
        nu_times = []
        for i in range(3):
            start_time = time.time()
            result = await nuextract.extract_structured_data(test_text)
            extraction_time = time.time() - start_time
            nu_times.append(extraction_time)
        
        print(f"  Load time: {nu_load_time:.2f}s")
        print(f"  Avg extraction time: {mean(nu_times):.2f}s")
        print(f"  Min extraction time: {min(nu_times):.2f}s")
        print(f"  Max extraction time: {max(nu_times):.2f}s")
        
        # Test Nanonets performance
        print("\nTesting Nanonets performance...")
        from src.processors.nanonets_ocr_engine import NanonetsOCREngine
        
        try:
            nanonets = NanonetsOCREngine(config)
            
            nano_load_start = time.time()
            await nanonets.load_models()
            nano_load_time = time.time() - nano_load_start
            
            # Create a simple test image
            from PIL import Image
            test_image = Image.new('RGB', (400, 300), color='white')
            
            nano_times = []
            for i in range(3):
                start_time = time.time()
                result = await nanonets.extract_text(test_image)
                extraction_time = time.time() - start_time
                nano_times.append(extraction_time)
            
            print(f"  Load time: {nano_load_time:.2f}s")
            print(f"  Avg extraction time: {mean(nano_times):.2f}s")
            print(f"  Min extraction time: {min(nano_times):.2f}s")
            print(f"  Max extraction time: {max(nano_times):.2f}s")
            
            return {
                "nuextract": {
                    "load_time": nu_load_time,
                    "avg_extraction": mean(nu_times),
                    "min_extraction": min(nu_times),
                    "max_extraction": max(nu_times)
                },
                "nanonets": {
                    "load_time": nano_load_time,
                    "avg_extraction": mean(nano_times),
                    "min_extraction": min(nano_times),
                    "max_extraction": max(nano_times)
                }
            }
            
        except Exception as e:
            print(f"  Nanonets test failed (model may not be available): {e}")
            return {
                "nuextract": {
                    "load_time": nu_load_time,
                    "avg_extraction": mean(nu_times),
                    "min_extraction": min(nu_times),
                    "max_extraction": max(nu_times)
                },
                "nanonets": {"error": str(e)}
            }
        
    except Exception as e:
        print(f"Model performance test failed: {e}")
        return {"error": str(e)}


def analyze_optimization_strategies():
    """Analyze the optimization strategies implemented in the system."""
    print("\n" + "="*50)
    print("Optimization Strategy Analysis")
    print("="*50)
    
    strategies = []
    
    # Check DocumentProcessor optimizations
    try:
        from src.processors.document_processor import DocumentProcessor
        config = ConfigManager()
        processor = DocumentProcessor(config)
        
        if hasattr(processor, '_optimize_for_vlm'):
            strategies.append("VLM optimization in DocumentProcessor")
        if hasattr(processor, '_calculate_optimal_batch_size'):
            strategies.append("Adaptive batch sizing")
        if hasattr(processor, '_adaptive_chunking'):
            strategies.append("Adaptive document chunking")
            
    except Exception as e:
        strategies.append(f"DocumentProcessor analysis failed: {e}")
    
    # Check OCR engine optimizations
    try:
        from src.processors.ocr_engine import UnifiedOCREngine
        config = ConfigManager()
        ocr_engine = UnifiedOCREngine(config)
        
        if hasattr(ocr_engine, '_get_optimal_device'):
            strategies.append("Optimal device selection for OCR")
        if hasattr(ocr_engine, 'load_models'):
            strategies.append("Asynchronous model loading")
            
    except Exception as e:
        strategies.append(f"OCR engine analysis failed: {e}")
    
    # Check NuExtract optimizations
    try:
        from src.extractors.nuextract_engine import NuExtractEngine
        config = ConfigManager()
        nu_engine = NuExtractEngine(config)
        
        if hasattr(nu_engine, '_create_text_image'):
            strategies.append("Text-to-image conversion for vision models")
        if hasattr(nu_engine, '_fix_json_issues'):
            strategies.append("JSON parsing error recovery")
            
    except Exception as e:
        strategies.append(f"NuExtract analysis failed: {e}")
    
    # Check multi-patient handling
    try:
        from src.extractors.multi_patient_handler import MultiPatientHandler
        config = ConfigManager()
        handler = MultiPatientHandler(config)
        
        if hasattr(handler, 'segment_text'):
            strategies.append("Multi-patient document segmentation")
        if hasattr(handler, 'process_multi_patient_document'):
            strategies.append("Parallel patient processing")
            
    except Exception as e:
        strategies.append(f"Multi-patient handler analysis failed: {e}")
    
    for strategy in strategies:
        print(f"  OK {strategy}")
    
    return strategies


async def main():
    """Run all optimization tests."""
    print("=" * 60)
    print("Medical Document Processing - Optimization Analysis")
    print("=" * 60)
    
    # System information
    sys_info = get_system_info()
    print(f"System: {sys_info['cpu_count']} CPUs, {sys_info['memory_total']}GB RAM")
    if sys_info['gpu_available']:
        print(f"GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory']}GB)")
    else:
        print("GPU: Not available")
    
    # Run tests
    results = {}
    
    print("\nRunning optimization tests...")
    
    # Memory usage test
    results['memory'] = await test_memory_usage()
    
    # Batch processing test  
    results['batch'] = await test_batch_processing()
    
    # Model performance test
    results['models'] = await test_model_performance()
    
    # Strategy analysis
    results['strategies'] = analyze_optimization_strategies()
    
    # Generate optimization report
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS REPORT")
    print("="*60)
    
    # Memory efficiency
    if 'memory' in results and 'error' not in results['memory']:
        mem_result = results['memory']
        print(f"\nMemory Efficiency:")
        print(f"  Model memory overhead: {mem_result['models_memory']:.1f} MB")
        print(f"  Model load time: {mem_result['model_load_time']:.2f}s")
        
        if 'processing_times' in mem_result:
            for size, time_val in mem_result['processing_times'].items():
                if time_val != float('inf'):
                    print(f"  {size.title()} text processing: {time_val:.2f}s")
    
    # Batch processing efficiency
    if 'batch' in results and 'error' not in results['batch']:
        print(f"\nBatch Processing:")
        for batch_size, batch_result in results['batch'].items():
            speedup = batch_result['speedup']
            success_rate = batch_result['success_rate']
            print(f"  Batch size {batch_size}: {speedup:.2f}x speedup, {success_rate:.1%} success")
    
    # Model performance
    if 'models' in results and 'error' not in results['models']:
        model_result = results['models']
        print(f"\nModel Performance:")
        if 'nuextract' in model_result:
            nu = model_result['nuextract']
            print(f"  NuExtract: {nu['load_time']:.1f}s load, {nu['avg_extraction']:.2f}s avg extraction")
        if 'nanonets' in model_result and 'error' not in model_result['nanonets']:
            nano = model_result['nanonets']
            print(f"  Nanonets: {nano['load_time']:.1f}s load, {nano['avg_extraction']:.2f}s avg extraction")
    
    # Optimization strategies
    if 'strategies' in results:
        print(f"\nImplemented Optimizations:")
        print(f"  Total strategies found: {len(results['strategies'])}")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    
    # Calculate optimization score
    score_factors = []
    
    # Memory efficiency (lower is better)
    if 'memory' in results and 'models_memory' in results['memory']:
        mem_score = max(0, 1 - (results['memory']['models_memory'] / 8000))  # Penalize > 8GB
        score_factors.append(mem_score)
    
    # Processing speed (faster is better)
    if 'models' in results and 'nuextract' in results['models']:
        speed_score = max(0, 1 - (results['models']['nuextract']['avg_extraction'] / 10))  # Penalize > 10s
        score_factors.append(speed_score)
    
    # Strategy implementation
    strategy_score = min(1.0, len(results.get('strategies', [])) / 10)  # Up to 10 strategies
    score_factors.append(strategy_score)
    
    overall_score = mean(score_factors) if score_factors else 0.5
    
    if overall_score >= 0.8:
        print("  EXCELLENT: System is highly optimized")
    elif overall_score >= 0.6:
        print("  GOOD: System is well optimized")
    elif overall_score >= 0.4:
        print("  MODERATE: Some optimizations in place")
    else:
        print("  NEEDS IMPROVEMENT: Limited optimization")
    
    print(f"  Optimization Score: {overall_score:.2f}/1.0")
    
    # Recommendations
    print(f"\nRecommendations:")
    if 'memory' in results and results['memory'].get('models_memory', 0) > 4000:
        print("  - Consider model quantization to reduce memory usage")
    if 'models' in results and results['models'].get('nuextract', {}).get('avg_extraction', 0) > 5:
        print("  - Optimize NuExtract processing for faster extraction")
    if len(results.get('strategies', [])) < 5:
        print("  - Implement additional optimization strategies")
    
    return overall_score


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(0 if exit_code > 0.5 else 1)