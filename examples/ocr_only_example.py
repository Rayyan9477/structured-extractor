#!/usr/bin/env python
"""
OCR-Only Example for Structured Extraction System

This example demonstrates how to use just the OCR portion of the system
with Monkey OCR and Nanonets OCR without needing a large language model.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from PIL import Image

# Add project root to path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager
from src.processors.ocr_ensemble import OCREnsembleEngine
from src.processors.document_processor import DocumentProcessor


async def extract_text_only(file_path, output_path=None):
    """
    Extract text from a document using only the OCR ensemble (Monkey OCR and Nanonets OCR).
    
    Args:
        file_path: Path to input document
        output_path: Path to save output (default: input filename with _ocr.json extension)
    """
    # Determine output path if not provided
    if not output_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_ocr.json"
    
    print(f"Extracting text from: {file_path}")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    
    try:
        # Initialize components
        config = ConfigManager()
        document_processor = DocumentProcessor(config)
        ocr_engine = OCREnsembleEngine(config)
        
        # Process document to get images
        print("Processing document...")
        images = await document_processor.process_document(file_path)
        
        if not images:
            print(f"No images extracted from {file_path}")
            return None
        
        print(f"Extracted {len(images)} page images")
        
        # Load OCR models
        await ocr_engine.load_models()
        
        # Process images with OCR ensemble
        print("Performing OCR with Monkey OCR and Nanonets OCR...")
        ocr_results = await ocr_engine.extract_text_batch(images)
        
        # Combine all page texts
        full_text = "\n\n--- PAGE BREAK ---\n\n".join([r.text for r in ocr_results])
        
        # Calculate average OCR confidence
        ocr_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
        
        # Save results
        results = {
            "text": full_text,
            "confidence": ocr_confidence,
            "page_count": len(images),
            "page_results": [
                {
                    "page": i+1,
                    "text": r.text,
                    "confidence": r.confidence,
                    "model": r.model_name
                }
                for i, r in enumerate(ocr_results)
            ]
        }
        
        # Save to file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOCR Results:")
        print(f"  Overall confidence: {ocr_confidence:.2f}")
        print(f"  Page count: {len(images)}")
        print(f"  Text length: {len(full_text)} characters")
        print(f"\nFull results saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None


async def main():
    """Main function to run the example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR-Only Example")
    parser.add_argument("--file", help="Path to file to extract")
    parser.add_argument("--output", help="Output path for results")
    
    args = parser.parse_args()
    
    if args.file:
        await extract_text_only(args.file, args.output)
    else:
        parser.print_help()
        
        # Use default sample if available
        for test_file in ["superbills/Olivares.OV.04.10.2025-04.29.2025-done.pdf", "sample_document.pdf"]:
            if Path(test_file).exists():
                print(f"\nRunning example with sample document: {test_file}")
                await extract_text_only(test_file)
                break


if __name__ == "__main__":
    asyncio.run(main()) 