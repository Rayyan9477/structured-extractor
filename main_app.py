#!/usr/bin/env python3
"""
Unified Medical Superbill Extraction Driver Application

This is the main driver code that runs the entire project with:
- Sequential model loading (Nanonets OCR → NuExtract) to save VRAM
- GPU CUDA optimization for faster processing
- Enhanced patient differentiation and CPT/ICD-10 code marking
- Streamlined UI integration
- Comprehensive error handling and logging
"""

import asyncio
import sys
import os
import gc
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
from datetime import datetime
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger, get_logger
from src.extraction_engine import ExtractionEngine
from src.exporters.data_exporter import DataExporter


class UnifiedSuperbillProcessor:
    """Unified processor for medical superbill extraction with optimized resource management."""
    
    def __init__(self, config_path: Optional[str] = None, gpu_optimization: bool = True):
        """
        Initialize the unified processor.
        
        Args:
            config_path: Optional path to configuration file
            gpu_optimization: Enable GPU optimizations
        """
        self.config = ConfigManager(config_path)
        self.logger = get_logger(__name__)
        self.gpu_optimization = gpu_optimization
        
        # Configure processing strategy
        self._configure_processing_strategy()
        
        # Initialize components (models will be loaded sequentially on first use)
        self.extraction_engine = ExtractionEngine(self.config)
        self.data_exporter = DataExporter(self.config)
        
        # Output directories
        self.output_dir = Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Unified Medical Superbill Processor initialized")
    
    def _configure_processing_strategy(self):
        """Configure processing strategy for optimal performance."""
        # Set OCR to use Nanonets only (no TrOCR fallback)
        self.config.update_config("ocr.model_name", "nanonets/Nanonets-OCR-s")
        self.config.update_config("ocr.enable_ensemble", False)
        self.config.update_config("ocr.fallback_models", [])
        
        # Set extraction to use NuExtract
        self.config.update_config("extraction.nuextract.model_name", "numind/NuExtract-2.0-8B")
        
        # Enable sequential loading to save VRAM
        self.config.update_config("models.sequential_loading", True)
        self.config.update_config("models.unload_after_use", True)
        
        # GPU optimization settings
        if self.gpu_optimization and torch.cuda.is_available():
            self.config.update_config("processing.use_cuda", True)
            self.config.update_config("processing.mixed_precision", True)
            self.config.update_config("processing.batch_size", 1)  # Conservative for memory
            self.logger.info(f"GPU optimization enabled - CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.config.update_config("processing.use_cuda", False)
            self.logger.info("GPU optimization disabled - using CPU")
        
        # Enhanced patient processing settings
        self.config.update_config("extraction.enable_patient_differentiation", True)
        self.config.update_config("extraction.enhanced_cpt_icd_marking", True)
        self.config.update_config("extraction.multi_patient_handling", True)
    
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            # Print memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    async def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single superbill file with optimized resource management.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Processing results
        """
        self.logger.info(f"Processing file: {file_path.name}")
        start_time = datetime.now()
        
        try:
            # Optimize GPU memory before processing
            if self.gpu_optimization:
                self._optimize_gpu_memory()
            
            # Extract data using sequential model loading
            extraction_result = await self.extraction_engine.extract_from_file(str(file_path))
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'success': extraction_result.success,
                'processing_time': processing_time,
                'extraction_confidence': extraction_result.extraction_confidence,
                'total_patients': extraction_result.total_patients,
                'extraction_result': extraction_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log detailed results
            if extraction_result.success:
                self.logger.info(
                    f"✓ {file_path.name}: {extraction_result.total_patients} patients, "
                    f"{processing_time:.2f}s, confidence: {extraction_result.extraction_confidence:.3f}"
                )
                
                # Log patient details
                for i, patient in enumerate(extraction_result.patients or []):
                    cpt_count = len(patient.cpt_codes) if patient.cpt_codes else 0
                    icd_count = len(patient.icd10_codes) if patient.icd10_codes else 0
                    self.logger.info(
                        f"  Patient {i+1}: {patient.first_name or ''} {patient.last_name or ''}, "
                        f"CPT: {cpt_count}, ICD-10: {icd_count}"
                    )
            else:
                self.logger.error(f"✗ Failed to process {file_path.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat()
            }
        finally:
            # Clean up GPU memory after processing
            if self.gpu_optimization:
                self._optimize_gpu_memory()
    
    async def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple files in batch with resource optimization.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processing results
        """
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"[{i}/{len(file_paths)}] Processing: {file_path.name}")
            
            result = await self.process_single_file(file_path)
            results.append(result)
            
            # Brief pause between files to prevent memory buildup
            await asyncio.sleep(1)
        
        return results
    
    async def export_results(self, results: List[Dict[str, Any]]) -> None:
        """Export processing results in multiple formats."""
        self.logger.info("Exporting results...")
        
        successful_results = [r for r in results if r['success'] and 'extraction_result' in r]
        
        if not successful_results:
            self.logger.warning("No successful results to export")
            return
        
        # Export individual results
        for result in successful_results:
            file_stem = Path(result['file_name']).stem
            
            # Export JSON
            json_path = self.output_dir / f"{file_stem}_extraction.json"
            await self._export_json(result, json_path)
            
            # Export CSV
            csv_path = self.output_dir / f"{file_stem}_patients.csv"
            await self._export_csv(result, csv_path)
        
        # Create summary report
        await self._create_summary_report(results)
    
    async def _export_json(self, result: Dict[str, Any], output_path: Path) -> None:
        """Export result to JSON format."""
        try:
            extraction_result = result['extraction_result']
            
            json_data = {
                'metadata': {
                    'file_name': result['file_name'],
                    'processing_time': result['processing_time'],
                    'extraction_confidence': result['extraction_confidence'],
                    'total_patients': result['total_patients'],
                    'timestamp': result['timestamp']
                },
                'patients': []
            }
            
            # Convert patients to JSON format
            for patient in extraction_result.patients or []:
                patient_data = {
                    'demographics': {
                        'first_name': patient.first_name,
                        'last_name': patient.last_name,
                        'middle_name': patient.middle_name,
                        'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                        'patient_id': patient.patient_id
                    },
                    'cpt_codes': [
                        {
                            'code': cpt.code,
                            'description': cpt.description,
                            'charge': cpt.charge,
                            'confidence': cpt.confidence.overall if cpt.confidence else None
                        } for cpt in (patient.cpt_codes or [])
                    ],
                    'icd10_codes': [
                        {
                            'code': icd.code,
                            'description': icd.description,
                            'confidence': icd.confidence.overall if icd.confidence else None
                        } for icd in (patient.icd10_codes or [])
                    ],
                    'service_info': {
                        'date_of_service': patient.service_info.date_of_service.isoformat() if patient.service_info and patient.service_info.date_of_service else None,
                        'provider_name': patient.service_info.provider_name if patient.service_info else None,
                        'provider_npi': patient.service_info.provider_npi if patient.service_info else None
                    } if patient.service_info else None,
                    'financial_info': {
                        'total_charges': patient.financial_info.total_charges if patient.financial_info else None,
                        'copay': patient.financial_info.copay if patient.financial_info else None,
                        'deductible': patient.financial_info.deductible if patient.financial_info else None
                    } if patient.financial_info else None
                }
                json_data['patients'].append(patient_data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported JSON: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON to {output_path}: {e}")
    
    async def _export_csv(self, result: Dict[str, Any], output_path: Path) -> None:
        """Export result to CSV format."""
        try:
            import pandas as pd
            
            extraction_result = result['extraction_result']
            csv_data = []
            
            for i, patient in enumerate(extraction_result.patients or []):
                # Base patient information
                base_row = {
                    'file_name': result['file_name'],
                    'patient_index': i + 1,
                    'first_name': patient.first_name or '',
                    'last_name': patient.last_name or '',
                    'middle_name': patient.middle_name or '',
                    'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else '',
                    'patient_id': patient.patient_id or '',
                    'service_date': patient.service_info.date_of_service.isoformat() if patient.service_info and patient.service_info.date_of_service else '',
                    'provider_name': patient.service_info.provider_name if patient.service_info else '',
                    'total_charges': patient.financial_info.total_charges if patient.financial_info else None
                }
                
                # Handle multiple CPT and ICD codes
                cpt_codes = patient.cpt_codes or []
                icd10_codes = patient.icd10_codes or []
                max_codes = max(len(cpt_codes), len(icd10_codes), 1)
                
                for j in range(max_codes):
                    row = base_row.copy()
                    
                    # Add CPT code if available
                    if j < len(cpt_codes):
                        cpt = cpt_codes[j]
                        row.update({
                            'cpt_code': cpt.code,
                            'cpt_description': cpt.description,
                            'cpt_charge': cpt.charge,
                            'cpt_confidence': cpt.confidence.overall if cpt.confidence else None
                        })
                    
                    # Add ICD-10 code if available
                    if j < len(icd10_codes):
                        icd = icd10_codes[j]
                        row.update({
                            'icd10_code': icd.code,
                            'icd10_description': icd.description,
                            'icd10_confidence': icd.confidence.overall if icd.confidence else None
                        })
                    
                    csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(output_path, index=False)
                self.logger.info(f"Exported CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV to {output_path}: {e}")
    
    async def _create_summary_report(self, results: List[Dict[str, Any]]) -> None:
        """Create a comprehensive summary report."""
        try:
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            
            total_patients = sum(r.get('total_patients', 0) for r in successful)
            avg_processing_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
            avg_confidence = sum(r.get('extraction_confidence', 0) for r in successful) / len(successful) if successful else 0
            
            summary = {
                'processing_summary': {
                    'total_files': len(results),
                    'successful_files': len(successful),
                    'failed_files': len(failed),
                    'success_rate': len(successful) / len(results) * 100 if results else 0,
                    'total_patients_extracted': total_patients,
                    'average_processing_time': avg_processing_time,
                    'average_confidence': avg_confidence
                },
                'file_details': [
                    {
                        'file_name': r['file_name'],
                        'success': r['success'],
                        'processing_time': r['processing_time'],
                        'patients_found': r.get('total_patients', 0),
                        'confidence': r.get('extraction_confidence', 0)
                    } for r in results
                ],
                'system_info': {
                    'gpu_enabled': self.gpu_optimization and torch.cuda.is_available(),
                    'gpu_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    'processing_strategy': 'Sequential: Nanonets OCR → NuExtract',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            summary_path = self.output_dir / "processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Summary report created: {summary_path}")
            
            # Print summary to console
            print("\n" + "="*80)
            print("🏥 MEDICAL SUPERBILL PROCESSING SUMMARY")
            print("="*80)
            print(f"Files processed: {len(results)}")
            print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
            print(f"Failed: {len(failed)}")
            print(f"Total patients extracted: {total_patients}")
            print(f"Average processing time: {avg_processing_time:.2f}s")
            print(f"Average confidence: {avg_confidence:.3f}")
            print(f"GPU optimization: {'Enabled' if self.gpu_optimization and torch.cuda.is_available() else 'Disabled'}")
            print("="*80)
            
        except Exception as e:
            self.logger.error(f"Failed to create summary report: {e}")


def launch_ui():
    """Launch the Streamlit UI."""
    import subprocess
    ui_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])


async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Unified Medical Superbill Extraction System")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--input", type=str, help="Input PDF file or directory")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU optimization")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in input directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Unified Medical Superbill Extraction System")
    
    # Launch UI if requested
    if args.ui:
        logger.info("Launching Streamlit UI...")
        launch_ui()
        return
    
    # Initialize processor
    gpu_optimization = not args.no_gpu
    processor = UnifiedSuperbillProcessor(args.config, gpu_optimization)
    
    # Determine input files
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            file_paths = [input_path]
        elif input_path.is_dir():
            file_paths = list(input_path.glob("*.pdf"))
        else:
            logger.error(f"Invalid input path: {input_path}")
            return
    else:
        # Default to superbills directory
        superbills_dir = Path("superbills")
        if not superbills_dir.exists():
            logger.error("No input specified and 'superbills' directory not found")
            return
        file_paths = list(superbills_dir.glob("*.pdf"))
    
    if not file_paths:
        logger.error("No PDF files found to process")
        return
    
    logger.info(f"Found {len(file_paths)} PDF files to process")
    
    try:
        # Process files
        if len(file_paths) == 1 and not args.batch:
            results = [await processor.process_single_file(file_paths[0])]
        else:
            results = await processor.process_batch(file_paths)
        
        # Export results
        await processor.export_results(results)
        
        logger.info("✅ Processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())