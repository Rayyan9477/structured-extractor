"""
Test Script for Superbill Structured Extraction

Tests the unified extraction system with real superbill documents and evaluates the results.
"""

import asyncio
import time
import json
import os
from pathlib import Path
import traceback

# Add src to Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.unified_extraction_system import UnifiedExtractionSystem


async def test_superbill_extraction():
    """Test the extraction system with superbill documents."""
    
    print("=" * 80)
    print("SUPERBILL STRUCTURED EXTRACTION TEST")
    print("=" * 80)
    
    # Initialize system
    try:
        system = UnifiedExtractionSystem()
        print("✓ System initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        traceback.print_exc()
        return
    
    # Test files
    test_files = [
        "superbills/Olivares.OV.04.10.2025-04.29.2025-done.pdf",
        "superbills/RuizPerez.Hosp.06.27.2025..3 DONE.pdf"
    ]
    
    # Output directory
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    results_summary = []
    
    for i, file_path in enumerate(test_files):
        print(f"\n[TEST {i+1}/{len(test_files)}] Processing: {file_path}")
        print("-" * 60)
        
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            continue
        
        try:
            start_time = time.time()
            
            # Extract with medical template
            output_path = output_dir / f"result_{i+1}_{Path(file_path).stem}.json"
            
            result = await system.extract_from_file(
                file_path=file_path,
                template_name="medical",
                output_format="json",
                output_path=str(output_path)
            )
            
            processing_time = time.time() - start_time
            
            # Print results summary
            print(f"✓ Processing completed in {processing_time:.2f}s")
            print(f"  OCR Confidence: {result.ocr_confidence:.3f}")
            print(f"  Extraction Confidence: {result.extraction_confidence:.3f}")
            print(f"  Overall Confidence: {result.overall_confidence:.3f}")
            print(f"  Text Length: {len(result.text)} characters")
            print(f"  Pages Processed: {result.metadata.page_count}")
            
            # Show sample of extracted structured data
            if result.structured_data and isinstance(result.structured_data, dict):
                print("\n  Extracted Structured Data Sample:")
                for key, value in list(result.structured_data.items())[:5]:
                    value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"    {key}: {value_preview}")
            
            results_summary.append({
                "file": file_path,
                "success": True,
                "processing_time": processing_time,
                "ocr_confidence": result.ocr_confidence,
                "extraction_confidence": result.extraction_confidence,
                "overall_confidence": result.overall_confidence,
                "text_length": len(result.text),
                "pages": result.metadata.page_count
            })
            
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            traceback.print_exc()
            
            results_summary.append({
                "file": file_path,
                "success": False,
                "error": str(e)
            })
    
    # Test with sample images
    print(f"\n[ADDITIONAL TESTS] Processing sample images")
    print("-" * 60)
    
    sample_images = [
        "sample_medical_document.png",
        "test_medical_document.png",
        "diagnostics/Olivares.OV.04.10.2025-04.29.2025-done_page_1_sample.png"
    ]
    
    for file_path in sample_images:
        if os.path.exists(file_path):
            print(f"\nProcessing image: {file_path}")
            try:
                start_time = time.time()
                
                output_path = output_dir / f"image_result_{Path(file_path).stem}.json"
                
                result = await system.extract_from_file(
                    file_path=file_path,
                    template_name="medical",
                    output_path=str(output_path)
                )
                
                processing_time = time.time() - start_time
                print(f"  ✓ Completed in {processing_time:.2f}s (confidence: {result.overall_confidence:.3f})")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    # Save summary
    print(f"\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = [r for r in results_summary if r.get("success", False)]
    failed_tests = [r for r in results_summary if not r.get("success", False)]
    
    print(f"Successful extractions: {len(successful_tests)}/{len(results_summary)}")
    print(f"Failed extractions: {len(failed_tests)}/{len(results_summary)}")
    
    if successful_tests:
        avg_confidence = sum(r["overall_confidence"] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average processing time: {avg_time:.2f}s")
    
    # Save detailed summary
    summary_path = output_dir / "test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": len(results_summary),
            "successful": len(successful_tests),
            "failed": len(failed_tests),
            "results": results_summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {summary_path}")
    
    return results_summary


async def test_text_extraction():
    """Test extraction from sample text."""
    print(f"\n[TEXT EXTRACTION TEST]")
    print("-" * 60)
    
    sample_medical_text = """
    SUPERBILL - MEDICAL SERVICES
    
    Patient: John Smith
    DOB: 01/15/1980
    ID: 12345
    
    Provider: Dr. Jane Doe
    NPI: 1234567890
    
    Date of Service: 07/21/2025
    
    Diagnosis:
    - Z00.00 - General adult medical examination
    - I10 - Essential hypertension
    
    Procedures:
    - 99213 - Office visit, established patient
    - 36415 - Collection of venous blood
    
    Charges:
    Office Visit: $150.00
    Lab Collection: $25.00
    Total: $175.00
    """
    
    try:
        system = UnifiedExtractionSystem()
        
        result = await system.extract_from_text(
            text=sample_medical_text,
            template_name="medical",
            output_path="test_results/sample_text_extraction.json"
        )
        
        print(f"✓ Text extraction completed")
        print(f"  Confidence: {result.extraction_confidence:.3f}")
        print(f"  Structured data fields: {len(result.structured_data) if result.structured_data else 0}")
        
        if result.structured_data:
            print("  Sample extracted data:")
            for key, value in list(result.structured_data.items())[:3]:
                print(f"    {key}: {value}")
    
    except Exception as e:
        print(f"✗ Text extraction failed: {e}")
        traceback.print_exc()


async def main():
    """Main test function."""
    print("Starting comprehensive extraction system test...")
    
    # Test with documents
    await test_superbill_extraction()
    
    # Test with sample text
    await test_text_extraction()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
