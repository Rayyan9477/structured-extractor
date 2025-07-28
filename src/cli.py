#!/usr/bin/env python
"""
Command Line Interface for Unified Structured Extraction System

Provides a command-line interface to the structured extraction system.
"""

import argparse
import asyncio
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.unified_extraction_system import (
    UnifiedExtractionSystem, 
    extract_from_file,
    extract_from_text, 
    batch_extract
)
from src.core.config_manager import ConfigManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Structured Extraction System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # File extraction command
    file_parser = subparsers.add_parser("extract-file", help="Extract data from a file")
    file_parser.add_argument("file_path", help="Path to input file")
    file_parser.add_argument(
        "-o", "--output", 
        help="Path to save output file (default: input filename with .json extension)",
        required=False
    )
    file_parser.add_argument(
        "-t", "--template", 
        help="Name of extraction template to use",
        default="default"
    )
    file_parser.add_argument(
        "-f", "--format", 
        help="Output format (json, csv, xml, txt)",
        default="json",
        choices=["json", "csv", "xml", "txt"]
    )
    file_parser.add_argument(
        "-c", "--config", 
        help="Path to configuration file",
        required=False
    )
    
    # Text extraction command
    text_parser = subparsers.add_parser("extract-text", help="Extract data from text input")
    text_parser.add_argument(
        "text", 
        nargs="?", 
        help="Text to extract from (if not provided, reads from stdin)",
    )
    text_parser.add_argument(
        "-i", "--input-file", 
        help="Input text file (alternative to direct text input)",
        required=False
    )
    text_parser.add_argument(
        "-o", "--output", 
        help="Path to save output file (default: 'output.json')",
        default="output.json"
    )
    text_parser.add_argument(
        "-t", "--template", 
        help="Name of extraction template to use",
        default="default"
    )
    text_parser.add_argument(
        "-f", "--format", 
        help="Output format (json, csv, xml, txt)",
        default="json",
        choices=["json", "csv", "xml", "txt"]
    )
    text_parser.add_argument(
        "-c", "--config", 
        help="Path to configuration file",
        required=False
    )
    
    # Batch extraction command
    batch_parser = subparsers.add_parser("batch", help="Process multiple files")
    batch_parser.add_argument(
        "input_dir", 
        help="Directory containing input files or list of files (comma-separated)",
    )
    batch_parser.add_argument(
        "-o", "--output-dir", 
        help="Directory to save output files (default: 'output')",
        default="output"
    )
    batch_parser.add_argument(
        "-t", "--template", 
        help="Name of extraction template to use",
        default="default"
    )
    batch_parser.add_argument(
        "-f", "--format", 
        help="Output format (json, csv, xml, txt)",
        default="json",
        choices=["json", "csv", "xml", "txt"]
    )
    batch_parser.add_argument(
        "-p", "--pattern", 
        help="File pattern to match (e.g. '*.pdf', '*.png')",
        default="*.*"
    )
    batch_parser.add_argument(
        "-c", "--config", 
        help="Path to configuration file",
        required=False
    )
    
    # Configuration command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action",
        choices=["show", "create", "validate"],
        help="Configuration action to perform"
    )
    config_parser.add_argument(
        "-c", "--config", 
        help="Path to configuration file",
        required=False
    )
    
    # Version info
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version information"
    )
    
    # Debug mode
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


async def handle_extract_file(args):
    """Handle the extract-file command."""
    # Ensure file exists
    if not os.path.isfile(args.file_path):
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        return 1
    
    # Determine output path
    output_path = args.output
    if not output_path:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        output_path = f"{base_name}.{args.format}"
    
    print(f"Processing file: {args.file_path}")
    start_time = time.time()
    
    try:
        # Extract data
        result = await extract_from_file(
            args.file_path,
            template_name=args.template,
            output_path=output_path,
            config_path=args.config
        )
        
        # Show summary
        elapsed_time = time.time() - start_time
        print(f"Extraction completed in {elapsed_time:.2f} seconds")
        print(f"Overall confidence: {result.overall_confidence:.2f}")
        print(f"Output saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        return 1


async def handle_extract_text(args):
    """Handle the extract-text command."""
    # Get input text
    if args.input_file:
        if not os.path.isfile(args.input_file):
            print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
            return 1
        
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        # Read from stdin
        print("Reading from standard input (Ctrl+D to end on Unix, Ctrl+Z on Windows)...")
        text = sys.stdin.read()
    
    if not text.strip():
        print("Error: No input text provided", file=sys.stderr)
        return 1
    
    print(f"Extracting data from text ({len(text)} characters)")
    start_time = time.time()
    
    try:
        # Extract data
        result = await extract_from_text(
            text,
            template_name=args.template,
            output_path=args.output,
            config_path=args.config
        )
        
        # Show summary
        elapsed_time = time.time() - start_time
        print(f"Extraction completed in {elapsed_time:.2f} seconds")
        print(f"Confidence: {result.overall_confidence:.2f}")
        print(f"Output saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        return 1


async def handle_batch(args):
    """Handle the batch command."""
    # Determine input files
    file_paths = []
    
    if "," in args.input_dir:
        # Comma-separated list of files
        for path in args.input_dir.split(","):
            path = path.strip()
            if os.path.isfile(path):
                file_paths.append(path)
            else:
                print(f"Warning: File not found: {path}", file=sys.stderr)
    elif os.path.isdir(args.input_dir):
        # Directory of files
        import glob
        pattern = os.path.join(args.input_dir, args.pattern)
        file_paths = glob.glob(pattern)
    else:
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    
    if not file_paths:
        print(f"Error: No files found matching pattern: {args.pattern}", file=sys.stderr)
        return 1
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing {len(file_paths)} files...")
    start_time = time.time()
    
    try:
        # Process batch
        results = await batch_extract(
            file_paths,
            template_name=args.template,
            output_dir=args.output_dir,
            config_path=args.config
        )
        
        # Show summary
        elapsed_time = time.time() - start_time
        successful = len([r for r in results if r.overall_confidence > 0.5])
        print(f"Batch processing completed in {elapsed_time:.2f} seconds")
        print(f"Successfully processed {successful}/{len(file_paths)} files")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error during batch processing: {e}", file=sys.stderr)
        return 1


def handle_config(args):
    """Handle the config command."""
    config_path = args.config or "config/extraction_config.json"
    
    if args.action == "show":
        try:
            config = ConfigManager(config_path)
            config_data = config.get_all()
            
            # Print formatted JSON
            print(json.dumps(config_data, indent=2))
            return 0
            
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
    
    elif args.action == "create":
        try:
            # Create default configuration if not exists
            if os.path.exists(config_path):
                overwrite = input(f"Configuration file already exists: {config_path}. Overwrite? (y/n): ")
                if overwrite.lower() != 'y':
                    print("Aborted.")
                    return 0
            
            # Default configuration
            default_config = {
                "ocr": {
                    "ensemble": {
                        "weights": {
                            "trocr": 1.0,
                            "monkey_ocr": 1.0,
                            "nanonets_ocr": 1.0
                        },
                        "use_models": ["trocr", "monkey_ocr", "nanonets_ocr"],
                        "minimum_models": 1,
                        "method": "weighted"
                    },
                    "monkey_ocr": {
                        "api_key": "${MONKEY_OCR_API_KEY}",
                        "endpoint_url": "${MONKEY_OCR_ENDPOINT_URL:http://localhost:8000/process}"
                    },
                    "nanonets_ocr": {
                        "api_key": "${NANONETS_OCR_API_KEY}",
                        "model_id": "${NANONETS_MODEL_ID:default}"
                    }
                },
                "extraction": {
                    "confidence_threshold": 0.7,
                    "default_template": "default",
                    "confidence_weights": {
                        "ocr": 0.3,
                        "extraction": 0.7
                    },
                    "nuextract": {
                        "model_name": "numind/NuExtract-2.0-8B",
                        "max_length": 4096,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "templates": {
                            "default": {
                                "name": "default",
                                "description": "Default structured data extraction template",
                                "schema": {
                                    "title": "str",
                                    "date": "str",
                                    "content": "str",
                                    "key_points": "list[str]",
                                    "metadata": "dict"
                                },
                                "examples": []
                            }
                        }
                    }
                },
                "export": {
                    "default_format": "json",
                    "output_dir": "output"
                },
                "logging": {
                    "level": "INFO",
                    "file": "extraction.log"
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            # Write configuration file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default configuration file: {config_path}")
            return 0
            
        except Exception as e:
            print(f"Error creating configuration: {e}", file=sys.stderr)
            return 1
    
    elif args.action == "validate":
        try:
            # Validate configuration
            config = ConfigManager(config_path)
            config_data = config.get_all()
            
            # Check required sections
            required_sections = ["ocr", "extraction", "export"]
            for section in required_sections:
                if section not in config_data:
                    print(f"Error: Missing required section '{section}'", file=sys.stderr)
                    return 1
            
            print(f"Configuration file is valid: {config_path}")
            return 0
            
        except Exception as e:
            print(f"Error validating configuration: {e}", file=sys.stderr)
            return 1


def show_version():
    """Show version information."""
    version = "1.0.0"
    print(f"Unified Structured Extraction System v{version}")
    print("Using models:")
    print("  - numind nuextract 8b")
    print("  - monkey ocr")
    print("  - nanonets ocr")
    return 0


async def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Handle version flag
    if args.version:
        return show_version()
    
    # Set debug mode
    if args.debug:
        # Configure logger
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Handle commands
    if args.command == "extract-file":
        return await handle_extract_file(args)
    elif args.command == "extract-text":
        return await handle_extract_text(args)
    elif args.command == "batch":
        return await handle_batch(args)
    elif args.command == "config":
        return handle_config(args)
    else:
        # No command provided, show help
        print("Please specify a command. Use -h or --help for usage information.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 