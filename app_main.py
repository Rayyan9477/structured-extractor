#!/usr/bin/env python3
"""
Medical Superbill Data Extraction System

This script provides a CLI for the medical superbill extraction project.
It leverages the modularized extraction engine to process documents.
"""

import argparse
import sys
import os
import asyncio
import json
from pathlib import Path
from typing import List

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Core dependencies
try:
    from src.core.config_manager import ConfigManager
    from src.core.logger import setup_logger, get_logger
    from src.extraction_engine import ExtractionEngine
    from src.exporters.data_exporter import DataExporter
    from src.core.data_schema import ExtractionResults
    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please ensure the application is installed correctly.")
    DEPENDENCIES_OK = False

# ============================================================================
# CLI INTERFACE AND EXECUTION
# ============================================================================

def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Medical Superbill Data Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_main.py --input file.pdf           # CLI extraction
  python app_main.py --input *.pdf --format csv # Batch processing
  python app_main.py --demo                   # Run demo mode
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--input",
        nargs="+",
        help="Run in CLI mode with specified input files"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with a sample text document"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-d",
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "excel", "all"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        help="Custom configuration file path"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    return parser

async def cli_process_files(file_paths: List[str], args):
    """Process files in CLI mode."""
    config = ConfigManager(args.config)
    engine = ExtractionEngine(config)
    exporter = DataExporter(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(__name__)
    
    success_count = 0
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths, 1):
        logger.info(f"Processing file {i}/{total_files}: {file_path}")
        
        try:
            # Extract data
            results = await engine.extract_from_file(file_path)
            
            if results.patients:
                # Generate output filename
                base_name = Path(file_path).stem
                
                # Export in requested format(s)
                if args.format in ["json", "all"]:
                    json_path = output_dir / f"{base_name}_results.json"
                    exporter.export_to_json(results, str(json_path))
                
                if args.format in ["csv", "all"]:
                    csv_path = output_dir / f"{base_name}_results.csv"
                    exporter.export_to_csv(results, str(csv_path))
                
                if args.format in ["excel", "all"]:
                    excel_path = output_dir / f"{base_name}_results.xlsx"
                    exporter.export_to_excel(results, str(excel_path))
                
                success_count += 1
                
                # Print summary
                logger.info(f"  ✅ Success - Found {results.total_patients} patients")
                logger.info(f"     Confidence: {results.extraction_confidence:.1%}")
                
            else:
                logger.warning(f"  ❌ No patients found in {file_path}")
                
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path}: {e}", exc_info=True)
    
    logger.info(f"\nProcessing complete: {success_count}/{total_files} files successful")
    return success_count == total_files

async def run_demo():
    """Run demo mode with sample processing."""
    print("🏥 Medical Superbill Extractor - Demo Mode")
    print("=" * 50)
    
    # Demo text sample
    demo_text = """
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
    
    print("Processing demo text...")
    
    config = ConfigManager()
    engine = ExtractionEngine(config)
    
    try:
        results = await engine.extract_from_text(demo_text)
        
        if results.patients:
            print("✅ Demo extraction successful!")
            print(f"Patients found: {results.total_patients}")
            
            for i, patient in enumerate(results.patients):
                print(f"\n👤 Patient {i+1}:")
                patient_info = patient.model_dump()
                for key, value in patient_info.items():
                    if value:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
        
        else:
            print("❌ Demo extraction failed or no patients found.")
            if results.metadata and 'error' in results.metadata:
                print(f"Error: {results.metadata['error']}")

    except Exception as e:
        print(f"An error occurred during demo: {e}")

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main execution function for the application."""
    parser = setup_cli_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(log_level, args.log_file)
    
    if not DEPENDENCIES_OK:
        get_logger(__name__).error("Dependencies not met. Exiting.")
        sys.exit(1)
        
    if args.input:
        asyncio.run(cli_process_files(args.input, args))
    
    elif args.demo:
        asyncio.run(run_demo())
        
    else:
        print("No input files provided. Use --help for usage information.")
        parser.print_help()

if __name__ == "__main__":
    main()
