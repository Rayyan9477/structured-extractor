"""
Main Application Entry Point for Medical Superbill Data Extraction

This module provides the main application interface for processing medical superbills
and extracting structured data using advanced OCR and NLP models.
"""

import argparse
import sys
import asyncio
from pathlib import Path
from typing import List, Optional
import json

from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger, get_logger
from src.core.data_schema import ExtractionResult, SuperbillDocument


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Medical Superbill Data Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.pdf --output results.json
  python main.py *.pdf --output-dir ./results/ --format csv
  python main.py input.pdf --config custom_config.yaml --verbose
        """
    )
    
    # Input arguments
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input PDF files to process"
    )
    
    # Output arguments
    parser.add_argument(
        "--output", "-o",
        help="Output file path (for single file processing)"
    )
    
    parser.add_argument(
        "--output-dir", "-d",
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)"
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        help="Custom configuration file path"
    )
    
    parser.add_argument(
        "--models-cache-dir",
        help="Directory to cache Hugging Face models"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for processing (default: 4)"
    )
    
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=4,
        help="Maximum number of worker processes (default: 4)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage (if available)"
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only processing"
    )
    
    # PHI and security options
    parser.add_argument(
        "--anonymize-phi",
        action="store_true",
        help="Anonymize detected PHI in output"
    )
    
    parser.add_argument(
        "--detect-phi-only",
        action="store_true",
        help="Only detect PHI, don't extract other fields"
    )
    
    # Quality and validation options
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for extractions (default: 0.7)"
    )
    
    parser.add_argument(
        "--validate-codes",
        action="store_true",
        default=True,
        help="Validate CPT and ICD-10 codes (default: True)"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip code validation (faster processing)"
    )
    
    # Logging and output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    
    parser.add_argument(
        "--log-file",
        help="Log file path"
    )
    
    parser.add_argument(
        "--no-audit-log",
        action="store_true",
        help="Disable audit logging"
    )
    
    # Development and debugging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate processing results"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    return parser


async def process_single_file(
    file_path: Path,
    config: ConfigManager,
    output_path: Optional[Path] = None,
    output_format: str = "json"
) -> ExtractionResult:
    """
    Process a single PDF file.
    
    Args:
        file_path: Path to input PDF file
        config: Configuration manager
        output_path: Optional output file path
        output_format: Output format (json/csv)
        
    Returns:
        Extraction result
    """
    logger = get_logger(__name__)
    logger.info(f"Processing file: {file_path}")
    
    try:
        # Import processing modules (lazy loading)
        from src.processors.document_processor import DocumentProcessor
        from src.extractors.field_extractor import FieldExtractor
        from src.exporters.data_exporter import DataExporter
        
        # Initialize processors
        doc_processor = DocumentProcessor(config)
        field_extractor = FieldExtractor(config)
        data_exporter = DataExporter(config)
        
        # Process document
        logger.info("Converting PDF to images...")
        images = await doc_processor.process_pdf(str(file_path))
        
        logger.info("Extracting structured data...")
        extraction_result = await field_extractor.extract_from_images(images)
        
        if extraction_result.success and output_path:
            logger.info(f"Exporting results to {output_path}")
            if output_format == "json":
                await data_exporter.export_json(extraction_result.document, output_path)
            elif output_format == "csv":
                await data_exporter.export_csv(extraction_result.document, output_path)
            elif output_format == "both":
                json_path = output_path.with_suffix('.json')
                csv_path = output_path.with_suffix('.csv')
                await data_exporter.export_json(extraction_result.document, json_path)
                await data_exporter.export_csv(extraction_result.document, csv_path)
        
        logger.info("Processing completed successfully")
        return extraction_result
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return ExtractionResult(
            success=False,
            error_message=str(e)
        )


async def process_multiple_files(
    file_paths: List[Path],
    config: ConfigManager,
    output_dir: Path,
    output_format: str = "json"
) -> List[ExtractionResult]:
    """
    Process multiple PDF files.
    
    Args:
        file_paths: List of input PDF file paths
        config: Configuration manager
        output_dir: Output directory
        output_format: Output format (json/csv)
        
    Returns:
        List of extraction results
    """
    logger = get_logger(__name__)
    logger.info(f"Processing {len(file_paths)} files...")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    results = []
    for file_path in file_paths:
        output_path = output_dir / f"{file_path.stem}_extracted"
        result = await process_single_file(file_path, config, output_path, output_format)
        results.append(result)
    
    # Generate summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Processing completed: {successful}/{len(file_paths)} files successful")
    
    return results


def main():
    """Main application entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO" if args.verbose else "WARNING" if args.quiet else "INFO"
    setup_logger(
        log_level=log_level,
        log_file=args.log_file
    )
    
    logger = get_logger(__name__)
    logger.info("Medical Superbill Data Extraction System starting...")
    
    try:
        # Load configuration
        config = ConfigManager(args.config)
        
        # Update configuration with command line arguments
        if args.batch_size:
            config.update_config("performance.batch_size", args.batch_size)
        if args.max_workers:
            config.update_config("performance.max_workers", args.max_workers)
        if args.confidence_threshold:
            config.update_config("extraction.confidence_threshold", args.confidence_threshold)
        if args.anonymize_phi:
            config.update_config("security.anonymization", True)
        if args.skip_validation:
            config.update_config("validation.validate_codes", False)
        
        # Parse input files
        input_files = []
        for pattern in args.input_files:
            path = Path(pattern)
            if path.is_file():
                input_files.append(path)
            else:
                # Handle glob patterns
                input_files.extend(Path().glob(pattern))
        
        if not input_files:
            logger.error("No input files found")
            sys.exit(1)
        
        logger.info(f"Found {len(input_files)} input files")
        
        # Process files
        if len(input_files) == 1 and args.output:
            # Single file processing with specific output
            output_path = Path(args.output)
            result = asyncio.run(process_single_file(
                input_files[0], config, output_path, args.format
            ))
            
            if result.success:
                logger.info("Processing completed successfully")
                sys.exit(0)
            else:
                logger.error(f"Processing failed: {result.error_message}")
                sys.exit(1)
        else:
            # Multiple file processing
            output_dir = Path(args.output_dir)
            results = asyncio.run(process_multiple_files(
                input_files, config, output_dir, args.format
            ))
            
            # Check results
            successful = sum(1 for r in results if r.success)
            if successful == len(results):
                logger.info("All files processed successfully")
                sys.exit(0)
            else:
                logger.warning(f"{len(results) - successful} files failed processing")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
