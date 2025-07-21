#!/usr/bin/env python3
"""
Comprehensive Superbill Processing Script

This script processes the sample superbills using the downloaded models to:
1. Perform OCR using multiple models (TrOCR, MonkeyOCR, Nanonets-OCR)
2. Extract structured data using NuExtract-2.0-8B
3. Export results in JSON and CSV formats
4. Compare model accuracy and inference details
"""

import asyncio
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.extraction_engine import ExtractionEngine
from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger, get_logger
from src.core.data_schema import ExtractionResults, PatientData
from src.exporters.data_exporter import DataExporter
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_engine import OCREngine
from src.extractors.nuextract_engine import NuExtractEngine


class ModelComparison:
    """Class to track and compare model performance."""
    
    def __init__(self):
        self.results = {}
        self.inference_times = {}
        self.accuracy_metrics = {}
    
    def add_model_result(self, model_name: str, file_name: str, 
                        extraction_time: float, confidence: float,
                        patient_count: int, text_length: int):
        """Add results for a specific model and file."""
        if model_name not in self.results:
            self.results[model_name] = []
        
        self.results[model_name].append({
            'file_name': file_name,
            'extraction_time': extraction_time,
            'confidence': confidence,
            'patient_count': patient_count,
            'text_length': text_length,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': list(self.results.keys()),
            'summary': {},
            'detailed_results': self.results
        }
        
        # Calculate summary statistics for each model
        for model_name, results in self.results.items():
            if results:
                avg_time = sum(r['extraction_time'] for r in results) / len(results)
                avg_confidence = sum(r['confidence'] for r in results) / len(results)
                total_patients = sum(r['patient_count'] for r in results)
                
                report['summary'][model_name] = {
                    'files_processed': len(results),
                    'avg_extraction_time': round(avg_time, 2),
                    'avg_confidence': round(avg_confidence, 3),
                    'total_patients_extracted': total_patients,
                    'avg_patients_per_file': round(total_patients / len(results), 1)
                }
        
        return report
    
    def export_comparison(self, output_path: str):
        """Export comparison results to JSON."""
        report = self.generate_comparison_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


class SuperbillProcessor:
    """Main processor for superbill extraction and comparison."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the processor."""
        self.config = ConfigManager(config_path)
        self.logger = get_logger(__name__)
        self.comparison = ModelComparison()
        
        # Output directories
        self.output_dir = Path("output")
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.comparison_dir = self.output_dir / "comparison"
        
        # Create output directories
        for dir_path in [self.output_dir, self.json_dir, self.csv_dir, self.comparison_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extraction_engine = ExtractionEngine(self.config)
        self.data_exporter = DataExporter(self.config)
    
    async def process_single_superbill(self, file_path: Path, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single superbill with specified model configuration.
        
        Args:
            file_path: Path to the superbill PDF
            model_config: Configuration for OCR and extraction models
            
        Returns:
            Processing results with timing and accuracy data
        """
        self.logger.info(f"Processing {file_path.name} with {model_config['name']}")
        
        start_time = time.time()
        
        try:
            # Update model configuration
            self._update_model_config(model_config)
            
            # Perform extraction
            extraction_result = await self.extraction_engine.extract_from_file(str(file_path))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate metrics
            patient_count = len(extraction_result.patients) if extraction_result.patients else 0
            avg_confidence = extraction_result.extraction_confidence or 0.0
            
            # Store results
            result = {
                'model_name': model_config['name'],
                'file_name': file_path.name,
                'success': extraction_result.success,
                'processing_time': processing_time,
                'patient_count': patient_count,
                'extraction_confidence': avg_confidence,
                'extraction_result': extraction_result,
                'metadata': {
                    'ocr_model': model_config.get('ocr_model', 'unknown'),
                    'extraction_model': model_config.get('extraction_model', 'unknown'),
                    'total_pages': extraction_result.metadata.get('total_pages', 0) if extraction_result.metadata else 0
                }
            }
            
            # Add to comparison tracker
            self.comparison.add_model_result(
                model_config['name'],
                file_path.name,
                processing_time,
                avg_confidence,
                patient_count,
                extraction_result.metadata.get('total_text_length', 0) if extraction_result.metadata else 0
            )
            
            self.logger.info(f"Completed {file_path.name} with {model_config['name']} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path.name} with {model_config['name']}: {e}")
            return {
                'model_name': model_config['name'],
                'file_name': file_path.name,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _update_model_config(self, model_config: Dict[str, Any]):
        """Update configuration with specific model settings."""
        if 'ocr_model' in model_config:
            self.config.update_config("ocr.model_name", model_config['ocr_model'])
        
        if 'extraction_model' in model_config:
            self.config.update_config("extraction.nuextract.model_name", model_config['extraction_model'])
    
    async def export_results(self, results: List[Dict[str, Any]]):
        """Export all results in JSON and CSV formats."""
        self.logger.info("Exporting results...")
        
        for result in results:
            if result['success'] and 'extraction_result' in result:
                model_name = result['model_name'].replace('/', '_').replace(' ', '_')
                file_name = Path(result['file_name']).stem
                
                # Export JSON
                json_path = self.json_dir / f"{file_name}_{model_name}_results.json"
                try:
                    self.data_exporter.export_extraction_results(
                        result['extraction_result'], 
                        str(json_path)
                    )
                    self.logger.info(f"Exported JSON: {json_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export JSON for {result['file_name']}: {e}")
                
                # Export CSV
                csv_path = self.csv_dir / f"{file_name}_{model_name}_patients.csv"
                try:
                    if result['extraction_result'].patients:
                        self.data_exporter.export_patients(
                            result['extraction_result'].patients,
                            str(csv_path),
                            'csv'
                        )
                        self.logger.info(f"Exported CSV: {csv_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export CSV for {result['file_name']}: {e}")
    
    def create_summary_csv(self, results: List[Dict[str, Any]]):
        """Create a summary CSV with all extraction results."""
        self.logger.info("Creating summary CSV...")
        
        summary_data = []
        
        for result in results:
            if result['success'] and 'extraction_result' in result:
                extraction_result = result['extraction_result']
                
                if extraction_result.patients:
                    for i, patient in enumerate(extraction_result.patients):
                        row = {
                            'model_name': result['model_name'],
                            'file_name': result['file_name'],
                            'processing_time': result['processing_time'],
                            'extraction_confidence': result['extraction_confidence'],
                            'patient_index': i + 1,
                            'first_name': patient.first_name or '',
                            'last_name': patient.last_name or '',
                            'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else '',
                            'patient_id': patient.patient_id or '',
                            'cpt_codes': ', '.join([cpt.code for cpt in patient.cpt_codes]) if patient.cpt_codes else '',
                            'icd10_codes': ', '.join([icd.code for icd in patient.icd10_codes]) if patient.icd10_codes else '',
                            'total_charges': patient.financial_info.total_charges if patient.financial_info else None,
                            'service_date': patient.service_info.date_of_service.isoformat() if patient.service_info and patient.service_info.date_of_service else '',
                            'provider_name': patient.service_info.provider_name if patient.service_info else '',
                            'phi_detected': patient.phi_detected
                        }
                        summary_data.append(row)
                else:
                    # Add row even if no patients found
                    row = {
                        'model_name': result['model_name'],
                        'file_name': result['file_name'],
                        'processing_time': result['processing_time'],
                        'extraction_confidence': result.get('extraction_confidence', 0.0),
                        'patient_index': 0,
                        'first_name': '',
                        'last_name': '',
                        'date_of_birth': '',
                        'patient_id': '',
                        'cpt_codes': '',
                        'icd10_codes': '',
                        'total_charges': None,
                        'service_date': '',
                        'provider_name': '',
                        'phi_detected': False
                    }
                    summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = self.output_dir / "all_extractions_summary.csv"
            df.to_csv(summary_path, index=False)
            self.logger.info(f"Created summary CSV: {summary_path}")
        
        return summary_data
    
    async def run_comprehensive_extraction(self):
        """Run comprehensive extraction with multiple model configurations."""
        # Define model configurations to test
        model_configs = [
            {
                'name': 'TrOCR_Base_Handwritten_NuExtract',
                'ocr_model': 'microsoft/trocr-base-handwritten',
                'extraction_model': 'numind/NuExtract-2.0-8B'
            },
            {
                'name': 'TrOCR_Large_Printed_NuExtract',
                'ocr_model': 'microsoft/trocr-large-printed',
                'extraction_model': 'numind/NuExtract-2.0-8B'
            }
        ]
        
        # Check for available models in cache
        models_cache = Path("models_cache")
        
        # Check for MonkeyOCR
        monkey_paths = [
            models_cache / "echo840_MonkeyOCR",
            models_cache / "models" / "echo840_MonkeyOCR",
            models_cache / "models--echo840--MonkeyOCR"
        ]
        for path in monkey_paths:
            if path.exists():
                self.logger.info(f"Found MonkeyOCR model at: {path}")
                model_configs.append({
                    'name': 'MonkeyOCR_NuExtract',
                    'ocr_model': 'echo840/MonkeyOCR',
                    'extraction_model': 'numind/NuExtract-2.0-8B'
                })
                break
        
        # Check for Nanonets OCR
        nanonets_paths = [
            models_cache / "nanonets_Nanonets-OCR-s",
            models_cache / "models" / "nanonets_Nanonets-OCR-s",
            models_cache / "models--nanonets--Nanonets-OCR-s"
        ]
        for path in nanonets_paths:
            if path.exists():
                self.logger.info(f"Found Nanonets OCR model at: {path}")
                model_configs.append({
                    'name': 'Nanonets_OCR_NuExtract',
                    'ocr_model': 'nanonets/Nanonets-OCR-s',
                    'extraction_model': 'numind/NuExtract-2.0-8B'
                })
                break
        
        # Get superbill files
        superbill_dir = Path("superbills")
        pdf_files = list(superbill_dir.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.error("No PDF files found in superbills directory")
            return
        
        self.logger.info(f"Found {len(pdf_files)} superbill files")
        self.logger.info(f"Testing {len(model_configs)} model configurations")
        
        all_results = []
        
        # Process each file with each model configuration
        for pdf_file in pdf_files:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing: {pdf_file.name}")
            self.logger.info(f"{'='*60}")
            
            for config in model_configs:
                result = await self.process_single_superbill(pdf_file, config)
                all_results.append(result)
                
                # Brief pause between models to avoid memory issues
                await asyncio.sleep(1)
        
        # Export all results
        await self.export_results(all_results)
        
        # Create summary
        summary_data = self.create_summary_csv(all_results)
        
        # Export comparison report
        comparison_path = self.comparison_dir / "model_comparison_report.json"
        self.comparison.export_comparison(str(comparison_path))
        self.logger.info(f"Exported comparison report: {comparison_path}")
        
        # Print summary to console
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print processing summary to console."""
        print("\n" + "="*80)
        print("SUPERBILL PROCESSING SUMMARY")
        print("="*80)
        
        # Group results by model
        model_results = {}
        for result in results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Print results for each model
        for model_name, model_res in model_results.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            successful = [r for r in model_res if r['success']]
            failed = [r for r in model_res if not r['success']]
            
            print(f"Files processed: {len(model_res)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            
            if successful:
                avg_time = sum(r['processing_time'] for r in successful) / len(successful)
                avg_confidence = sum(r.get('extraction_confidence', 0) for r in successful) / len(successful)
                total_patients = sum(r.get('patient_count', 0) for r in successful)
                
                print(f"Average processing time: {avg_time:.2f} seconds")
                print(f"Average confidence: {avg_confidence:.3f}")
                print(f"Total patients extracted: {total_patients}")
                
                for result in successful:
                    print(f"  - {result['file_name']}: {result.get('patient_count', 0)} patients, "
                          f"{result['processing_time']:.2f}s, conf: {result.get('extraction_confidence', 0):.3f}")
            
            if failed:
                print("Failed files:")
                for result in failed:
                    print(f"  - {result['file_name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\nOutput files saved to: {self.output_dir}")
        print("="*80)


async def main():
    """Main function to run the superbill processing."""
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("Starting Comprehensive Superbill Processing")
    logger.info(f"Working directory: {Path.cwd()}")
    
    try:
        # Initialize processor
        processor = SuperbillProcessor()
        
        # Run comprehensive extraction
        results = await processor.run_comprehensive_extraction()
        
        logger.info("Superbill processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
