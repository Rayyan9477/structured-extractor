#!/usr/bin/env python
"""
Example Unified Extraction System Usage

This example demonstrates how to use the Unified Extraction System to extract structured data
from documents using Monkey OCR, Nanonets OCR, and NuExtract-8B models.
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add project root to path if running the script directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unified_extraction_system import UnifiedExtractionSystem


async def extract_document(file_path, output_path=None, template=None):
    """
    Extract structured data from a document using the unified system with
    Monkey OCR, Nanonets OCR, and NuExtract-8B.
    
    Args:
        file_path: Path to input document
        output_path: Path to save output (default: input filename with .json extension)
        template: Name of extraction template to use
    """
    # Create extraction system
    extraction_system = UnifiedExtractionSystem()
    
    # Determine output path if not provided
    if not output_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_extracted.json"
    
    print(f"Extracting data from: {file_path}")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    print(f"Using extraction model: NumInd NuExtract 8B")
    
    try:
        # Extract data
        result = await extraction_system.extract_from_file(
            file_path,
            template_name=template,
            output_path=output_path
        )
        
        # Print results summary
        print("\nExtraction Results:")
        print(f"  Overall confidence: {result.overall_confidence:.2f}")
        print(f"  OCR confidence: {result.ocr_confidence:.2f}")
        print(f"  Extraction confidence: {result.extraction_confidence:.2f}")
        print(f"  Processing time: {result.metadata.processing_time:.2f} seconds")
        
        # Print some extracted data
        print("\nExtracted Data Preview:")
        print_data_preview(result.structured_data)
        
        print(f"\nFull results saved to: {output_path}")
        return result
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None


def print_data_preview(data, max_items=3, indent=2):
    """
    Print a preview of the structured data.
    
    Args:
        data: Structured data dictionary
        max_items: Maximum number of items to display per list
        indent: Indentation level
    """
    indent_str = " " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_data_preview(value, max_items, indent + 2)
        elif isinstance(value, list):
            print(f"{indent_str}{key}: [{len(value)} items]")
            for i, item in enumerate(value[:max_items]):
                if isinstance(item, dict):
                    print(f"{indent_str}  [{i}]:")
                    print_data_preview(item, max_items, indent + 4)
                else:
                    print(f"{indent_str}  [{i}]: {item}")
            if len(value) > max_items:
                print(f"{indent_str}  ... ({len(value) - max_items} more items)")
        else:
            print(f"{indent_str}{key}: {value}")


async def extract_medical_document(file_path, output_path=None):
    """
    Extract structured medical data from a document using the specialized medical template.
    
    Args:
        file_path: Path to input medical document
        output_path: Path to save output
    """
    # Create extraction system
    extraction_system = UnifiedExtractionSystem()
    
    # Determine output path if not provided
    if not output_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_medical_extracted.json"
    
    print(f"Extracting medical data from: {file_path}")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    print(f"Using extraction model: NumInd NuExtract 8B")
    print(f"Using template: medical")
    
    try:
        # Extract data with medical template
        result = await extraction_system.extract_from_file(
            file_path,
            template_name="medical",
            output_path=output_path
        )
        
        # Print results summary
        print("\nMedical Document Extraction Results:")
        print(f"  Overall confidence: {result.overall_confidence:.2f}")
        
        # Print patient info if available
        if "patient_info" in result.structured_data:
            patient = result.structured_data["patient_info"]
            print("\nPatient Information:")
            print(f"  Name: {patient.get('name', 'N/A')}")
            print(f"  DOB: {patient.get('dob', 'N/A')}")
            
        # Print billing codes if available
        if "billing" in result.structured_data:
            billing = result.structured_data["billing"]
            print("\nBilling Information:")
            if "cpt_codes" in billing and billing["cpt_codes"]:
                print(f"  CPT Codes: {', '.join(billing.get('cpt_codes', []))}")
            if "icd10_codes" in billing and billing["icd10_codes"]:
                print(f"  ICD-10 Codes: {', '.join(billing.get('icd10_codes', []))}")
        
        print(f"\nFull medical extraction saved to: {output_path}")
        return result
        
    except Exception as e:
        print(f"Error during medical extraction: {e}")
        return None


async def process_batch(directory, pattern="*.pdf", output_dir="output"):
    """
    Process a batch of documents in a directory.
    
    Args:
        directory: Directory containing input files
        pattern: File pattern to match
        output_dir: Directory to save outputs
    """
    import glob
    
    # Find files
    file_pattern = os.path.join(directory, pattern)
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Processing {len(files)} files...")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    print(f"Using extraction model: NumInd NuExtract 8B")
    
    # Create extraction system
    extraction_system = UnifiedExtractionSystem()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    results = []
    for file_path in files:
        try:
            # Determine output path
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.json")
            
            # Extract data
            result = await extraction_system.extract_from_file(
                file_path,
                output_path=output_path
            )
            
            results.append((file_path, result.overall_confidence))
            print(f"Processed {file_path}: confidence {result.overall_confidence:.2f}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    
    # Print summary
    print("\nBatch Processing Results:")
    for file_path, confidence in results:
        status = "✓" if confidence >= 0.7 else "!"
        print(f"{status} {os.path.basename(file_path)}: {confidence:.2f}")


async def extract_with_custom_schema(file_path, output_path=None):
    """
    Extract data with a custom schema.
    
    Args:
        file_path: Path to input file
        output_path: Path to save output
    """
    # Create extraction system
    extraction_system = UnifiedExtractionSystem()
    
    # Define custom schema
    custom_schema = {
        "document_info": {
            "title": "str",
            "date": "str",
            "type": "str"
        },
        "sections": "list[dict]",
        "keywords": "list[str]",
        "summary": "str"
    }
    
    # Determine output path
    if not output_path:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{base_name}_custom_extracted.json"
    
    print(f"Extracting with custom schema from: {file_path}")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    print(f"Using extraction model: NumInd NuExtract 8B")
    
    try:
        # Extract data with custom schema
        result = await extraction_system.extract_from_file(
            file_path,
            custom_schema=custom_schema,
            output_path=output_path
        )
        
        print(f"\nCustom extraction saved to: {output_path}")
        print_data_preview(result.structured_data)
        return result
        
    except Exception as e:
        print(f"Error during custom extraction: {e}")
        return None


async def export_to_different_formats(file_path, output_dir="output"):
    """
    Extract data and export to multiple formats.
    
    Args:
        file_path: Path to input file
        output_dir: Directory to save outputs
    """
    # Create extraction system
    extraction_system = UnifiedExtractionSystem()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process file once
    print(f"Processing file: {file_path}")
    print(f"Using OCR models: Monkey OCR, Nanonets OCR")
    print(f"Using extraction model: NumInd NuExtract 8B")
    
    try:
        # Extract data
        result = await extraction_system.extract_from_file(file_path)
        
        # Export to different formats
        formats = ["json", "csv", "xml", "txt"]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        for fmt in formats:
            output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
            await extraction_system.export_results(result, output_path, fmt)
            print(f"Exported to {fmt}: {output_path}")
        
    except Exception as e:
        print(f"Error during extraction/export: {e}")


async def main():
    """Main function to run the examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Extraction System Examples")
    parser.add_argument("--file", help="Path to file to extract")
    parser.add_argument("--medical", help="Extract medical data using medical template")
    parser.add_argument("--batch", help="Directory of files to batch process")
    parser.add_argument("--custom", help="Extract with custom schema")
    parser.add_argument("--formats", help="Export to different formats")
    parser.add_argument("--template", help="Template name to use")
    parser.add_argument("--output", help="Output path for results")
    
    args = parser.parse_args()
    
    if args.file:
        # Extract single file
        await extract_document(args.file, args.output, args.template)
    elif args.medical:
        # Extract medical document
        await extract_medical_document(args.medical, args.output)
    elif args.batch:
        # Batch process directory
        await process_batch(args.batch, output_dir=args.output or "output")
    elif args.custom:
        # Extract with custom schema
        await extract_with_custom_schema(args.custom, args.output)
    elif args.formats:
        # Export to different formats
        await export_to_different_formats(args.formats, args.output or "output")
    else:
        # No arguments, show help
        parser.print_help()
        
        # Use default sample if available
        sample_file = Path(__file__).parent.parent / "sample_document.pdf"
        if sample_file.exists():
            print("\nRunning example with sample document...")
            await extract_document(str(sample_file))


if __name__ == "__main__":
    asyncio.run(main()) 