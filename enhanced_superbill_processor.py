#!/usr/bin/env python3
"""
Enhanced Superbill Processing and Model Comparison Script

This script provides comprehensive superbill processing with detailed model comparison:
1. Performs OCR using multiple models (TrOCR variants, MonkeyOCR, Nanonets-OCR)
2. Extracts structured data using NuExtract-2.0-8B
3. Exports results in JSON and CSV formats with patient separation
4. Provides detailed accuracy and inference metrics comparison
5. Generates comprehensive performance reports
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
import numpy as np
from collections import defaultdict

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


class EnhancedModelComparison:
    """Enhanced class to track and compare model performance with detailed metrics."""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = defaultdict(dict)
        self.detailed_extractions = {}
    
    def add_model_result(self, model_name: str, file_name: str, 
                        extraction_time: float, confidence: float,
                        patient_count: int, text_length: int,
                        extraction_result: ExtractionResults = None):
        """Add comprehensive results for a specific model and file."""
        if model_name not in self.results:
            self.results[model_name] = []
        
        # Basic metrics
        result_data = {
            'file_name': file_name,
            'extraction_time': extraction_time,
            'confidence': confidence,
            'patient_count': patient_count,
            'text_length': text_length,
            'timestamp': datetime.now().isoformat()
        }
        
        # Detailed extraction metrics if available
        if extraction_result and extraction_result.patients:
            detailed_metrics = self._calculate_detailed_metrics(extraction_result)
            result_data.update(detailed_metrics)
        
        self.results[model_name].append(result_data)
        
        # Store detailed extraction for comparison
        if extraction_result:
            if model_name not in self.detailed_extractions:
                self.detailed_extractions[model_name] = {}
            self.detailed_extractions[model_name][file_name] = extraction_result
    
    def _calculate_detailed_metrics(self, extraction_result: ExtractionResults) -> Dict[str, Any]:
        """Calculate detailed metrics from extraction results."""
        patients = extraction_result.patients or []
        
        # Count extracted fields
        total_cpt_codes = sum(len(p.cpt_codes) for p in patients if p.cpt_codes)
        total_icd10_codes = sum(len(p.icd10_codes) for p in patients if p.icd10_codes)
        
        # Count patients with specific data
        patients_with_names = sum(1 for p in patients if p.first_name or p.last_name)
        patients_with_dob = sum(1 for p in patients if p.date_of_birth)
        patients_with_id = sum(1 for p in patients if p.patient_id)
        patients_with_service_date = sum(1 for p in patients if p.service_info and p.service_info.date_of_service)
        patients_with_financial = sum(1 for p in patients if p.financial_info and p.financial_info.total_charges)
        
        return {
            'total_cpt_codes': total_cpt_codes,
            'total_icd10_codes': total_icd10_codes,
            'patients_with_names': patients_with_names,
            'patients_with_dob': patients_with_dob,
            'patients_with_id': patients_with_id,
            'patients_with_service_date': patients_with_service_date,
            'patients_with_financial': patients_with_financial,
            'data_completeness_score': self._calculate_completeness_score(patients)
        }
    
    def _calculate_completeness_score(self, patients: List[PatientData]) -> float:
        """Calculate a data completeness score (0-100)."""
        if not patients:
            return 0.0
        
        total_score = 0
        for patient in patients:
            score = 0
            max_score = 8  # Total possible fields
            
            # Name fields (2 points)
            if patient.first_name: score += 1
            if patient.last_name: score += 1
            
            # Core fields (6 points)
            if patient.date_of_birth: score += 1
            if patient.patient_id: score += 1
            if patient.cpt_codes: score += 1
            if patient.icd10_codes: score += 1
            if patient.service_info and patient.service_info.date_of_service: score += 1
            if patient.financial_info and patient.financial_info.total_charges: score += 1
            
            total_score += (score / max_score) * 100
        
        return total_score / len(patients)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report with detailed analytics."""
        report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'models_compared': list(self.results.keys()),
                'total_files_processed': len(set(
                    result['file_name'] for model_results in self.results.values() 
                    for result in model_results
                ))
            },
            'performance_summary': {},
            'detailed_comparison': {},
            'model_rankings': {},
            'recommendations': {}
        }
        
        # Calculate performance summary for each model
        for model_name, results in self.results.items():
            if results:
                metrics = self._calculate_model_metrics(results)
                report['performance_summary'][model_name] = metrics
        
        # Generate model rankings
        report['model_rankings'] = self._generate_rankings(report['performance_summary'])
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['performance_summary'])
        
        # Detailed comparison by file
        report['detailed_comparison'] = self._generate_detailed_comparison()
        
        return report
    
    def _calculate_model_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a model."""
        successful_results = [r for r in results if r.get('patient_count', 0) > 0]
        
        if not results:
            return {'status': 'no_data'}
        
        # Basic performance metrics
        avg_time = np.mean([r['extraction_time'] for r in results])
        avg_confidence = np.mean([r['confidence'] for r in results])
        total_patients = sum(r.get('patient_count', 0) for r in results)
        success_rate = len(successful_results) / len(results) * 100
        
        metrics = {
            'files_processed': len(results),
            'success_rate': round(success_rate, 1),
            'avg_extraction_time': round(avg_time, 3),
            'min_extraction_time': round(min(r['extraction_time'] for r in results), 3),
            'max_extraction_time': round(max(r['extraction_time'] for r in results), 3),
            'avg_confidence': round(avg_confidence, 3),
            'total_patients_extracted': total_patients,
            'avg_patients_per_file': round(total_patients / len(results), 2) if results else 0
        }
        
        # Advanced metrics (if available)
        if successful_results and 'data_completeness_score' in successful_results[0]:
            advanced_metrics = {
                'avg_cpt_codes_per_file': round(np.mean([r.get('total_cpt_codes', 0) for r in successful_results]), 2),
                'avg_icd10_codes_per_file': round(np.mean([r.get('total_icd10_codes', 0) for r in successful_results]), 2),
                'avg_data_completeness': round(np.mean([r.get('data_completeness_score', 0) for r in successful_results]), 1),
                'name_extraction_rate': round(np.mean([r.get('patients_with_names', 0) for r in successful_results]) / max(np.mean([r.get('patient_count', 1) for r in successful_results]), 1) * 100, 1),
                'financial_extraction_rate': round(np.mean([r.get('patients_with_financial', 0) for r in successful_results]) / max(np.mean([r.get('patient_count', 1) for r in successful_results]), 1) * 100, 1)
            }
            metrics.update(advanced_metrics)
        
        return metrics
    
    def _generate_rankings(self, performance_summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate model rankings based on different criteria."""
        models = list(performance_summary.keys())
        
        rankings = {}
        
        # Speed ranking (faster is better)
        speed_ranking = sorted(models, key=lambda m: performance_summary[m].get('avg_extraction_time', float('inf')))
        rankings['speed'] = speed_ranking
        
        # Accuracy ranking (higher confidence is better)
        accuracy_ranking = sorted(models, key=lambda m: performance_summary[m].get('avg_confidence', 0), reverse=True)
        rankings['confidence'] = accuracy_ranking
        
        # Patient extraction ranking
        patient_ranking = sorted(models, key=lambda m: performance_summary[m].get('total_patients_extracted', 0), reverse=True)
        rankings['patient_extraction'] = patient_ranking
        
        # Data completeness ranking (if available)
        if any('avg_data_completeness' in performance_summary[m] for m in models):
            completeness_ranking = sorted(
                models, 
                key=lambda m: performance_summary[m].get('avg_data_completeness', 0), 
                reverse=True
            )
            rankings['data_completeness'] = completeness_ranking
        
        # Overall ranking (weighted composite score)
        overall_ranking = self._calculate_overall_ranking(models, performance_summary)
        rankings['overall'] = overall_ranking
        
        return rankings
    
    def _calculate_overall_ranking(self, models: List[str], performance_summary: Dict[str, Any]) -> List[str]:
        """Calculate overall ranking using weighted composite score."""
        scores = {}
        
        for model in models:
            metrics = performance_summary[model]
            
            # Normalize metrics (0-100 scale)
            speed_score = max(0, 100 - metrics.get('avg_extraction_time', 100) * 10)  # Faster = higher score
            confidence_score = metrics.get('avg_confidence', 0) * 100
            success_score = metrics.get('success_rate', 0)
            patient_score = min(100, metrics.get('total_patients_extracted', 0) * 10)
            completeness_score = metrics.get('avg_data_completeness', 0)
            
            # Weighted composite score
            composite_score = (
                speed_score * 0.2 +
                confidence_score * 0.3 +
                success_score * 0.2 +
                patient_score * 0.15 +
                completeness_score * 0.15
            )
            
            scores[model] = composite_score
        
        return sorted(models, key=lambda m: scores[m], reverse=True)
    
    def _generate_recommendations(self, performance_summary: Dict[str, Any]) -> Dict[str, str]:
        """Generate recommendations based on performance analysis."""
        recommendations = {}
        
        if not performance_summary:
            return {'general': 'No performance data available for recommendations.'}
        
        # Find best performing models
        best_speed = min(performance_summary.keys(), key=lambda m: performance_summary[m].get('avg_extraction_time', float('inf')))
        best_accuracy = max(performance_summary.keys(), key=lambda m: performance_summary[m].get('avg_confidence', 0))
        best_extraction = max(performance_summary.keys(), key=lambda m: performance_summary[m].get('total_patients_extracted', 0))
        
        recommendations['speed'] = f"For fastest processing, use {best_speed} (avg: {performance_summary[best_speed].get('avg_extraction_time', 0):.3f}s)"
        recommendations['accuracy'] = f"For highest confidence, use {best_accuracy} (confidence: {performance_summary[best_accuracy].get('avg_confidence', 0):.3f})"
        recommendations['extraction'] = f"For maximum patient extraction, use {best_extraction} ({performance_summary[best_extraction].get('total_patients_extracted', 0)} patients)"
        
        # Overall recommendation
        if 'avg_data_completeness' in performance_summary[list(performance_summary.keys())[0]]:
            best_overall = max(performance_summary.keys(), key=lambda m: performance_summary[m].get('avg_data_completeness', 0))
            recommendations['general'] = f"For best overall performance, use {best_overall} (completeness: {performance_summary[best_overall].get('avg_data_completeness', 0):.1f}%)"
        else:
            recommendations['general'] = f"For balanced performance, use {best_accuracy} based on confidence scores"
        
        return recommendations
    
    def _generate_detailed_comparison(self) -> Dict[str, Any]:
        """Generate detailed file-by-file comparison."""
        comparison = {}
        
        # Get all files processed
        all_files = set()
        for model_results in self.results.values():
            for result in model_results:
                all_files.add(result['file_name'])
        
        # Compare each file across models
        for file_name in all_files:
            file_comparison = {}
            
            for model_name, model_results in self.results.items():
                file_result = next((r for r in model_results if r['file_name'] == file_name), None)
                if file_result:
                    file_comparison[model_name] = {
                        'extraction_time': file_result['extraction_time'],
                        'confidence': file_result['confidence'],
                        'patient_count': file_result.get('patient_count', 0),
                        'completeness_score': file_result.get('data_completeness_score', 0)
                    }
            
            comparison[file_name] = file_comparison
        
        return comparison
    
    def export_comprehensive_report(self, output_path: str):
        """Export comprehensive comparison report."""
        report = self.generate_comprehensive_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def create_performance_dashboard_csv(self, output_path: str):
        """Create a CSV file suitable for performance dashboard visualization."""
        dashboard_data = []
        
        for model_name, results in self.results.items():
            for result in results:
                row = {
                    'model_name': model_name,
                    'file_name': result['file_name'],
                    'extraction_time': result['extraction_time'],
                    'confidence': result['confidence'],
                    'patient_count': result.get('patient_count', 0),
                    'text_length': result.get('text_length', 0),
                    'cpt_codes_extracted': result.get('total_cpt_codes', 0),
                    'icd10_codes_extracted': result.get('total_icd10_codes', 0),
                    'data_completeness_score': result.get('data_completeness_score', 0),
                    'timestamp': result['timestamp']
                }
                dashboard_data.append(row)
        
        if dashboard_data:
            df = pd.DataFrame(dashboard_data)
            df.to_csv(output_path, index=False)


class EnhancedSuperbillProcessor:
    """Enhanced processor for superbill extraction with detailed analytics."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced processor."""
        self.config = ConfigManager(config_path)
        self.logger = get_logger(__name__)
        self.comparison = EnhancedModelComparison()
        
        # Output directories
        self.output_dir = Path("output")
        self.json_dir = self.output_dir / "json"
        self.csv_dir = self.output_dir / "csv"
        self.comparison_dir = self.output_dir / "comparison"
        self.dashboard_dir = self.output_dir / "dashboard"
        
        # Create output directories
        for dir_path in [self.output_dir, self.json_dir, self.csv_dir, self.comparison_dir, self.dashboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extraction_engine = ExtractionEngine(self.config)
        self.data_exporter = DataExporter(self.config)
    
    async def process_single_superbill(self, file_path: Path, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single superbill with enhanced metrics collection.
        
        Args:
            file_path: Path to the superbill PDF
            model_config: Configuration for OCR and extraction models
            
        Returns:
            Enhanced processing results with detailed metrics
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
            text_length = extraction_result.metadata.get('total_text_length', 0) if extraction_result.metadata else 0
            
            # Store results with enhanced metrics
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
                    'total_pages': extraction_result.metadata.get('total_pages', 0) if extraction_result.metadata else 0,
                    'text_length': text_length
                }
            }
            
            # Add to enhanced comparison tracker
            self.comparison.add_model_result(
                model_config['name'],
                file_path.name,
                processing_time,
                avg_confidence,
                patient_count,
                text_length,
                extraction_result
            )
            
            self.logger.info(f"✓ Completed {file_path.name} with {model_config['name']} in {processing_time:.2f}s - {patient_count} patients")
            return result
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {file_path.name} with {model_config['name']}: {e}")
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
        """Export all results in multiple formats with patient separation."""
        self.logger.info("Exporting extraction results...")
        
        for result in results:
            if result['success'] and 'extraction_result' in result:
                model_name = result['model_name'].replace('/', '_').replace(' ', '_')
                file_name = Path(result['file_name']).stem
                
                # Export JSON with full structured data
                json_path = self.json_dir / f"{file_name}_{model_name}_full_results.json"
                try:
                    # Custom JSON export with enhanced metadata
                    json_data = {
                        'extraction_metadata': {
                            'model_name': result['model_name'],
                            'file_name': result['file_name'],
                            'processing_time': result['processing_time'],
                            'extraction_confidence': result['extraction_confidence'],
                            'timestamp': datetime.now().isoformat()
                        },
                        'extraction_results': self._serialize_extraction_result(result['extraction_result'])
                    }
                    
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    self.logger.info(f"📄 Exported JSON: {json_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export JSON for {result['file_name']}: {e}")
                
                # Export CSV with patient-separated rows
                csv_path = self.csv_dir / f"{file_name}_{model_name}_patients.csv"
                try:
                    if result['extraction_result'].patients:
                        self._export_patients_csv(result['extraction_result'].patients, csv_path, result)
                        self.logger.info(f"📊 Exported CSV: {csv_path}")
                except Exception as e:
                    self.logger.error(f"Failed to export CSV for {result['file_name']}: {e}")
    
    def _serialize_extraction_result(self, extraction_result: ExtractionResults) -> Dict[str, Any]:
        """Serialize extraction result to JSON-compatible format."""
        return {
            'success': extraction_result.success,
            'total_patients': len(extraction_result.patients) if extraction_result.patients else 0,
            'extraction_confidence': extraction_result.extraction_confidence,
            'patients': [self._serialize_patient(patient) for patient in (extraction_result.patients or [])],
            'metadata': extraction_result.metadata or {}
        }
    
    def _serialize_patient(self, patient: PatientData) -> Dict[str, Any]:
        """Serialize patient data to JSON-compatible format."""
        return {
            'first_name': patient.first_name,
            'last_name': patient.last_name,
            'middle_name': patient.middle_name,
            'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None,
            'patient_id': patient.patient_id,
            'contact_info': {
                'phone': patient.contact_info.phone if patient.contact_info else None,
                'email': patient.contact_info.email if patient.contact_info else None,
                'address': patient.contact_info.address if patient.contact_info else None
            } if patient.contact_info else None,
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
            } if patient.financial_info else None,
            'phi_detected': patient.phi_detected
        }
    
    def _export_patients_csv(self, patients: List[PatientData], csv_path: Path, result: Dict[str, Any]):
        """Export patients to CSV with comprehensive data."""
        csv_data = []
        
        for i, patient in enumerate(patients):
            # Create base row for patient
            base_row = {
                'model_name': result['model_name'],
                'file_name': result['file_name'],
                'processing_time': result['processing_time'],
                'extraction_confidence': result['extraction_confidence'],
                'patient_index': i + 1,
                'first_name': patient.first_name or '',
                'last_name': patient.last_name or '',
                'middle_name': patient.middle_name or '',
                'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else '',
                'patient_id': patient.patient_id or '',
                'phone': patient.contact_info.phone if patient.contact_info else '',
                'email': patient.contact_info.email if patient.contact_info else '',
                'address': patient.contact_info.address if patient.contact_info else '',
                'service_date': patient.service_info.date_of_service.isoformat() if patient.service_info and patient.service_info.date_of_service else '',
                'provider_name': patient.service_info.provider_name if patient.service_info else '',
                'provider_npi': patient.service_info.provider_npi if patient.service_info else '',
                'total_charges': patient.financial_info.total_charges if patient.financial_info else None,
                'copay': patient.financial_info.copay if patient.financial_info else None,
                'deductible': patient.financial_info.deductible if patient.financial_info else None,
                'phi_detected': patient.phi_detected
            }
            
            # Handle CPT and ICD codes - create multiple rows if needed
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
                else:
                    row.update({
                        'cpt_code': '',
                        'cpt_description': '',
                        'cpt_charge': None,
                        'cpt_confidence': None
                    })
                
                # Add ICD-10 code if available
                if j < len(icd10_codes):
                    icd = icd10_codes[j]
                    row.update({
                        'icd10_code': icd.code,
                        'icd10_description': icd.description,
                        'icd10_confidence': icd.confidence.overall if icd.confidence else None
                    })
                else:
                    row.update({
                        'icd10_code': '',
                        'icd10_description': '',
                        'icd10_confidence': None
                    })
                
                csv_data.append(row)
        
        # Create DataFrame and export
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
    
    def create_comprehensive_summary(self, results: List[Dict[str, Any]]):
        """Create comprehensive summary with enhanced analytics."""
        self.logger.info("Creating comprehensive summary...")
        
        # Create detailed summary CSV
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
                            'cpt_codes_count': len(patient.cpt_codes) if patient.cpt_codes else 0,
                            'icd10_codes_count': len(patient.icd10_codes) if patient.icd10_codes else 0,
                            'cpt_codes': ', '.join([cpt.code for cpt in patient.cpt_codes]) if patient.cpt_codes else '',
                            'icd10_codes': ', '.join([icd.code for icd in patient.icd10_codes]) if patient.icd10_codes else '',
                            'total_charges': patient.financial_info.total_charges if patient.financial_info else None,
                            'service_date': patient.service_info.date_of_service.isoformat() if patient.service_info and patient.service_info.date_of_service else '',
                            'provider_name': patient.service_info.provider_name if patient.service_info else '',
                            'phi_detected': patient.phi_detected,
                            'has_complete_name': bool(patient.first_name and patient.last_name),
                            'has_dob': bool(patient.date_of_birth),
                            'has_financial_info': bool(patient.financial_info and patient.financial_info.total_charges),
                            'has_service_date': bool(patient.service_info and patient.service_info.date_of_service)
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
                        'cpt_codes_count': 0,
                        'icd10_codes_count': 0,
                        'cpt_codes': '',
                        'icd10_codes': '',
                        'total_charges': None,
                        'service_date': '',
                        'provider_name': '',
                        'phi_detected': False,
                        'has_complete_name': False,
                        'has_dob': False,
                        'has_financial_info': False,
                        'has_service_date': False
                    }
                    summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_path = self.output_dir / "comprehensive_extractions_summary.csv"
            df.to_csv(summary_path, index=False)
            self.logger.info(f"📈 Created comprehensive summary: {summary_path}")
        
        return summary_data
    
    async def run_comprehensive_extraction(self):
        """Run comprehensive extraction with all available models."""
        # Define comprehensive model configurations
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
                self.logger.info(f"🐒 Found MonkeyOCR model at: {path}")
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
                self.logger.info(f"🔬 Found Nanonets OCR model at: {path}")
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
            self.logger.error("❌ No PDF files found in superbills directory")
            return
        
        self.logger.info(f"📁 Found {len(pdf_files)} superbill files: {[f.name for f in pdf_files]}")
        self.logger.info(f"🤖 Testing {len(model_configs)} model configurations")
        
        all_results = []
        
        # Process each file with each model configuration
        total_combinations = len(pdf_files) * len(model_configs)
        current_combination = 0
        
        for pdf_file in pdf_files:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"📄 PROCESSING FILE: {pdf_file.name}")
            self.logger.info(f"{'='*80}")
            
            for config in model_configs:
                current_combination += 1
                self.logger.info(f"[{current_combination}/{total_combinations}] Testing {config['name']}...")
                
                result = await self.process_single_superbill(pdf_file, config)
                all_results.append(result)
                
                # Brief pause between models to avoid memory issues
                await asyncio.sleep(2)
        
        # Export all results
        await self.export_results(all_results)
        
        # Create comprehensive summary
        summary_data = self.create_comprehensive_summary(all_results)
        
        # Export enhanced comparison reports
        comparison_path = self.comparison_dir / "comprehensive_model_comparison.json"
        self.comparison.export_comprehensive_report(str(comparison_path))
        self.logger.info(f"📊 Exported comprehensive comparison: {comparison_path}")
        
        # Create performance dashboard CSV
        dashboard_path = self.dashboard_dir / "model_performance_dashboard.csv"
        self.comparison.create_performance_dashboard_csv(str(dashboard_path))
        self.logger.info(f"📈 Created performance dashboard data: {dashboard_path}")
        
        # Print comprehensive summary
        self._print_comprehensive_summary(all_results)
        
        return all_results
    
    def _print_comprehensive_summary(self, results: List[Dict[str, Any]]):
        """Print comprehensive processing summary with enhanced metrics."""
        print("\n" + "="*100)
        print("🏥 COMPREHENSIVE SUPERBILL PROCESSING SUMMARY")
        print("="*100)
        
        # Group results by model
        model_results = {}
        for result in results:
            model_name = result['model_name']
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append(result)
        
        # Print overall statistics
        total_files = len(set(r['file_name'] for r in results))
        total_models = len(model_results)
        successful_extractions = len([r for r in results if r['success']])
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Files processed: {total_files}")
        print(f"   Models tested: {total_models}")
        print(f"   Total extraction attempts: {len(results)}")
        print(f"   Successful extractions: {successful_extractions}")
        print(f"   Success rate: {successful_extractions/len(results)*100:.1f}%")
        
        # Print results for each model
        for model_name, model_res in model_results.items():
            print(f"\n🤖 {model_name}:")
            print("-" * 60)
            
            successful = [r for r in model_res if r['success']]
            failed = [r for r in model_res if not r['success']]
            
            print(f"   Files processed: {len(model_res)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            
            if successful:
                avg_time = sum(r['processing_time'] for r in successful) / len(successful)
                avg_confidence = sum(r.get('extraction_confidence', 0) for r in successful) / len(successful)
                total_patients = sum(r.get('patient_count', 0) for r in successful)
                
                print(f"   Average processing time: {avg_time:.3f} seconds")
                print(f"   Average confidence: {avg_confidence:.3f}")
                print(f"   Total patients extracted: {total_patients}")
                print(f"   Average patients per file: {total_patients/len(successful):.1f}")
                
                print(f"   📋 File details:")
                for result in successful:
                    patients = result.get('patient_count', 0)
                    time_taken = result['processing_time']
                    confidence = result.get('extraction_confidence', 0)
                    print(f"      • {result['file_name']}: {patients} patients, {time_taken:.2f}s, conf: {confidence:.3f}")
            
            if failed:
                print(f"   ❌ Failed files:")
                for result in failed:
                    print(f"      • {result['file_name']}: {result.get('error', 'Unknown error')}")
        
        # Print model rankings from comparison
        try:
            report = self.comparison.generate_comprehensive_report()
            rankings = report.get('model_rankings', {})
            recommendations = report.get('recommendations', {})
            
            if rankings:
                print(f"\n🏆 MODEL RANKINGS:")
                for category, ranking in rankings.items():
                    print(f"   {category.replace('_', ' ').title()}: {' > '.join(ranking)}")
            
            if recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for category, recommendation in recommendations.items():
                    print(f"   {category.title()}: {recommendation}")
        
        except Exception as e:
            self.logger.warning(f"Could not generate rankings: {e}")
        
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"   JSON results: {self.json_dir}")
        print(f"   CSV results: {self.csv_dir}")
        print(f"   Comparison reports: {self.comparison_dir}")
        print(f"   Dashboard data: {self.dashboard_dir}")
        print(f"   Summary CSV: {self.output_dir / 'comprehensive_extractions_summary.csv'}")
        
        print("="*100)


async def main():
    """Main function to run the enhanced superbill processing."""
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Enhanced Comprehensive Superbill Processing")
    logger.info(f"📂 Working directory: {Path.cwd()}")
    
    start_time = time.time()
    
    try:
        # Initialize enhanced processor
        processor = EnhancedSuperbillProcessor()
        
        # Run comprehensive extraction with all models
        results = await processor.run_comprehensive_extraction()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"✅ Enhanced superbill processing completed successfully in {total_time:.2f} seconds!")
        logger.info(f"📊 Processed {len(results)} extraction attempts")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Processing interrupted by user")
    except Exception as e:
        logger.error(f"💥 Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
