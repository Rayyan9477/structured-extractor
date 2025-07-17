"""
Example Usage of Medical Superbill Extraction System

This script demonstrates how to use the extraction system to process medical superbills
and export the extracted data in various formats.
"""

import asyncio
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.extraction_engine import ExtractionEngine
from src.validators.data_validator import DataValidator
from src.exporters.data_exporter import DataExporter


async def basic_extraction_example():
    """Basic example of extracting data from a single file."""
    print("=== Basic Extraction Example ===")
    
    # Initialize extraction engine
    engine = ExtractionEngine()
    
    # Example file path (replace with actual file)
    file_path = "examples/sample_superbill.pdf"
    
    if not Path(file_path).exists():
        print(f"Sample file not found: {file_path}")
        print("Please add a sample superbill PDF to the examples directory")
        return
    
    try:
        # Extract data from file
        print(f"Extracting data from: {file_path}")
        results = await engine.extract_from_file(file_path)
        
        print(f"Extraction completed!")
        print(f"Total patients found: {results.total_patients}")
        print(f"Overall confidence: {results.extraction_confidence:.2f}")
        
        # Display patient information
        for i, patient in enumerate(results.patients):
            print(f"\n--- Patient {i + 1} ---")
            print(f"Name: {patient.first_name} {patient.last_name}")
            print(f"DOB: {patient.date_of_birth}")
            print(f"Patient ID: {patient.patient_id}")
            
            if patient.cpt_codes:
                print(f"CPT Codes: {[cpt.code for cpt in patient.cpt_codes]}")
            
            if patient.icd10_codes:
                print(f"ICD-10 Codes: {[icd.code for icd in patient.icd10_codes]}")
            
            if patient.financial_info:
                print(f"Total Charges: ${patient.financial_info.total_charges}")
            
            print(f"Extraction Confidence: {patient.extraction_confidence:.2f}")
            print(f"PHI Detected: {'Yes' if patient.phi_detected else 'No'}")
        
        return results
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None


async def text_extraction_example():
    """Example of extracting data from text input."""
    print("\n=== Text Extraction Example ===")
    
    # Sample superbill text
    sample_text = """
    MEDICAL SUPERBILL
    
    Patient Name: John Smith
    Date of Birth: 01/15/1980
    Patient ID: JS123456
    
    Date of Service: 03/15/2024
    Provider: Dr. Jane Johnson, MD
    NPI: 1234567890
    
    Diagnosis Codes:
    Z00.00 - General medical examination
    I10 - Essential hypertension
    
    Procedure Codes:
    99213 - Office visit, established patient
    93000 - Electrocardiogram
    
    Charges:
    Total: $350.00
    Insurance Payment: $280.00
    Patient Payment: $70.00
    """
    
    # Initialize extraction engine
    engine = ExtractionEngine()
    
    try:
        # Extract data from text
        print("Extracting data from sample text...")
        results = await engine.extract_from_text(sample_text, "sample_text")
        
        print(f"Extraction completed!")
        print(f"Total patients found: {results.total_patients}")
        
        # Display results
        for patient in results.patients:
            print(f"\nExtracted Patient Data:")
            print(f"Name: {patient.first_name} {patient.last_name}")
            print(f"DOB: {patient.date_of_birth}")
            print(f"Patient ID: {patient.patient_id}")
            
            if patient.cpt_codes:
                for cpt in patient.cpt_codes:
                    print(f"CPT: {cpt.code} - {cpt.description}")
            
            if patient.icd10_codes:
                for icd in patient.icd10_codes:
                    print(f"ICD-10: {icd.code} - {icd.description}")
        
        return results
        
    except Exception as e:
        print(f"Text extraction failed: {e}")
        return None


async def validation_example(results):
    """Example of validating extracted data."""
    if not results or not results.patients:
        print("\nNo data to validate")
        return
    
    print("\n=== Data Validation Example ===")
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate each patient
    for i, patient in enumerate(results.patients):
        print(f"\n--- Validating Patient {i + 1} ---")
        
        validation_results = validator.validate_patient_data(patient)
        
        print(f"Overall Valid: {'Yes' if validation_results['overall_valid'] else 'No'}")
        
        if validation_results['validation_errors']:
            print("Validation Errors:")
            for error in validation_results['validation_errors']:
                print(f"  - {error}")
        
        if validation_results['validation_warnings']:
            print("Validation Warnings:")
            for warning in validation_results['validation_warnings']:
                print(f"  - {warning}")
        
        # CPT code validation details
        if validation_results['cpt_validation']:
            print("CPT Code Validation:")
            for cpt_result in validation_results['cpt_validation']:
                status = "✓" if cpt_result['is_valid'] else "✗"
                print(f"  {status} {cpt_result['code']}: {cpt_result['message']}")
        
        # ICD-10 code validation details
        if validation_results['icd10_validation']:
            print("ICD-10 Code Validation:")
            for icd_result in validation_results['icd10_validation']:
                status = "✓" if icd_result['is_valid'] else "✗"
                print(f"  {status} {icd_result['code']}: {icd_result['message']}")


async def export_example(results):
    """Example of exporting data in different formats."""
    if not results or not results.patients:
        print("\nNo data to export")
        return
    
    print("\n=== Data Export Example ===")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize exporter
    exporter = DataExporter()
    
    try:
        # Export to CSV
        csv_path = output_dir / "extracted_patients.csv"
        exporter.export_patients(results.patients, str(csv_path), format_type='csv')
        print(f"✓ Exported to CSV: {csv_path}")
        
        # Export to JSON
        json_path = output_dir / "extracted_patients.json"
        exporter.export_patients(results.patients, str(json_path), format_type='json')
        print(f"✓ Exported to JSON: {json_path}")
        
        # Export extraction results to JSON
        results_path = output_dir / "extraction_results.json"
        exporter.export_extraction_results(results, str(results_path))
        print(f"✓ Exported extraction results: {results_path}")
        
        # Export to Excel (if openpyxl is available)
        try:
            excel_path = output_dir / "extracted_patients.xlsx"
            exporter.export_patients(results.patients, str(excel_path), format_type='excel')
            print(f"✓ Exported to Excel: {excel_path}")
        except ImportError:
            print("⚠ Excel export skipped (openpyxl not installed)")
        
    except Exception as e:
        print(f"Export failed: {e}")


async def batch_processing_example():
    """Example of batch processing multiple files."""
    print("\n=== Batch Processing Example ===")
    
    # Example file paths (replace with actual files)
    file_paths = [
        "examples/superbill1.pdf",
        "examples/superbill2.pdf",
        "examples/superbill3.pdf"
    ]
    
    # Filter to existing files
    existing_files = [f for f in file_paths if Path(f).exists()]
    
    if not existing_files:
        print("No sample files found for batch processing")
        print("Please add sample superbill PDFs to the examples directory")
        return
    
    # Initialize extraction engine
    engine = ExtractionEngine()
    
    try:
        print(f"Processing {len(existing_files)} files...")
        
        # Batch extract
        results_list = await engine.extract_batch(existing_files)
        
        print(f"Batch processing completed!")
        
        # Display summary
        total_patients = sum(r.total_patients for r in results_list)
        successful_extractions = sum(1 for r in results_list if r.total_patients > 0)
        
        print(f"Files processed: {len(results_list)}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Total patients found: {total_patients}")
        
        # Export batch results
        if results_list:
            exporter = DataExporter()
            output_dir = Path("output") / "batch_results"
            exporter.export_batch(results_list, str(output_dir), format_type='csv')
            print(f"✓ Batch results exported to: {output_dir}")
        
        return results_list
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return None


async def anonymization_example(results):
    """Example of anonymizing patient data."""
    if not results or not results.patients:
        print("\nNo data to anonymize")
        return
    
    print("\n=== Data Anonymization Example ===")
    
    # Initialize validator (contains anonymizer)
    validator = DataValidator()
    
    # Anonymize each patient
    anonymized_patients = []
    
    for i, patient in enumerate(results.patients):
        print(f"\n--- Anonymizing Patient {i + 1} ---")
        print(f"Original Name: {patient.first_name} {patient.last_name}")
        print(f"Original DOB: {patient.date_of_birth}")
        
        # Anonymize
        anonymized_patient = validator.anonymize_patient_data(patient)
        
        print(f"Anonymized Name: {anonymized_patient.first_name} {anonymized_patient.last_name}")
        print(f"Anonymized DOB: {anonymized_patient.date_of_birth}")
        print(f"Anonymized: {'Yes' if anonymized_patient.is_anonymized else 'No'}")
        
        anonymized_patients.append(anonymized_patient)
    
    # Export anonymized data
    try:
        exporter = DataExporter()
        output_path = Path("output") / "anonymized_patients.csv"
        exporter.export_patients(anonymized_patients, str(output_path), format_type='csv')
        print(f"\n✓ Anonymized data exported: {output_path}")
    except Exception as e:
        print(f"Failed to export anonymized data: {e}")


async def main():
    """Main function demonstrating all features."""
    print("Medical Superbill Extraction System - Example Usage")
    print("=" * 60)
    
    # Create examples directory if it doesn't exist
    Path("examples").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Basic extraction
    results = await basic_extraction_example()
    
    # Text extraction
    text_results = await text_extraction_example()
    
    # Use text results if file extraction didn't work
    if not results and text_results:
        results = text_results
    
    if results:
        # Validation
        await validation_example(results)
        
        # Export
        await export_example(results)
        
        # Anonymization
        await anonymization_example(results)
    
    # Batch processing
    await batch_processing_example()
    
    print("\n" + "=" * 60)
    print("Example usage completed!")
    print("Check the 'output' directory for exported files.")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
