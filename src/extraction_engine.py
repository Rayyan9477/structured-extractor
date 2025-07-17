"""
Main Data Extraction Engine

Integrates OCR, field detection, NuExtract processing, and multi-patient handling
to provide a complete extraction pipeline for medical superbills.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import PatientData, SuperbillDocument, ExtractionResults
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_engine import OCREngine
from src.extractors.field_detector import FieldDetector, DetectionResult
from src.extractors.nuextract_engine import NuExtractEngine
from src.extractors.multi_patient_handler import MultiPatientHandler
from src.validators.date_validator import DateValidator
from src.validators.data_validator import DataValidator


class ExtractionPipeline:
    """Main extraction pipeline orchestrating all components."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize extraction pipeline.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.document_processor = DocumentProcessor(config)
        self.ocr_engine = OCREngine(config)
        self.field_detector = FieldDetector(config)
        self.nuextract_engine = NuExtractEngine(config)
        self.multi_patient_handler = MultiPatientHandler(config)
        self.date_validator = DateValidator(config)
        self.data_validator = DataValidator(config)
        
        # Configuration
        self.extraction_config = config.get("extraction", {})
        self.max_retries = self.extraction_config.get("max_retries", 3)
        self.confidence_threshold = self.extraction_config.get("confidence_threshold", 0.7)
        
        # Track model loading status
        self._models_loaded = False
        
    async def extract_from_file(self, file_path: str) -> ExtractionResults:
        """
        Extract structured data from a superbill file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Extraction results with patient data
        """
        self.logger.info(f"Starting extraction from file: {file_path}")
        
        try:
            # Ensure models are loaded
            await self._ensure_models_loaded()
            
            # Step 1: Process document
            images = await self.document_processor.process_document(file_path)
            
            if not images:
                raise ValueError(f"No images extracted from {file_path}")
            
            # Step 2: OCR processing
            ocr_results = []
            for i, image in enumerate(images):
                self.logger.debug(f"Processing page {i + 1}/{len(images)}")
                page_text = await self.ocr_engine.extract_text(image)
                ocr_results.append(page_text)
            
            # Combine all page texts
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(ocr_results)
            
            # Step 3: Extract structured data
            patients = await self._extract_patient_data(full_text)
            
            # Step 4: Create results
            results = ExtractionResults(
                file_path=file_path,
                extraction_timestamp=datetime.now(),
                total_patients=len(patients),
                patients=patients,
                extraction_confidence=self._calculate_overall_confidence(patients),
                metadata={
                    'total_pages': len(images),
                    'total_text_length': len(full_text),
                    'processing_method': 'multi_patient_pipeline'
                }
            )
            
            self.logger.info(f"Extraction completed. Found {len(patients)} patients")
            return results
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {file_path}: {str(e)}")
            raise
    
    async def extract_from_text(self, text: str, source_name: str = "text_input") -> ExtractionResults:
        """
        Extract structured data from text.
        
        Args:
            text: Input text
            source_name: Name for the source
            
        Returns:
            Extraction results with patient data
        """
        self.logger.info(f"Starting extraction from text: {source_name}")
        
        try:
            # Ensure models are loaded
            await self._ensure_models_loaded()
            
            # Extract structured data
            patients = await self._extract_patient_data(text)
            
            # Create results
            results = ExtractionResults(
                file_path=source_name,
                extraction_timestamp=datetime.now(),
                total_patients=len(patients),
                patients=patients,
                extraction_confidence=self._calculate_overall_confidence(patients),
                metadata={
                    'total_text_length': len(text),
                    'processing_method': 'text_only_pipeline'
                }
            )
            
            self.logger.info(f"Extraction completed. Found {len(patients)} patients")
            return results
            
        except Exception as e:
            self.logger.error(f"Text extraction failed for {source_name}: {str(e)}")
            raise
    
    async def _extract_patient_data(self, text: str) -> List[PatientData]:
        """
        Extract patient data from text using multi-patient handling.
        
        Args:
            text: Input text
            
        Returns:
            List of patient data
        """
        async def extract_single_patient(segment_text: str, patient_index: int) -> Optional[PatientData]:
            """Extract data for a single patient segment."""
            try:
                # Method 1: NuExtract structured extraction
                nuextract_data = await self.nuextract_engine.extract_structured_data(segment_text)
                
                if nuextract_data:
                    # Enhance with field detection
                    field_results = self.field_detector.detect_all_fields(segment_text)
                    enhanced_data = self._enhance_with_field_detection(nuextract_data, field_results)
                    
                    # Validate and standardize dates
                    enhanced_data = self._validate_patient_dates(enhanced_data)
                    
                    # Set patient index
                    enhanced_data.patient_index = patient_index
                    
                    return enhanced_data
                
                # Method 2: Fallback to field detection only
                self.logger.warning(f"NuExtract failed for patient {patient_index}, using field detection")
                patient_data = self._create_patient_from_fields(segment_text, patient_index)
                
                # Validate and standardize dates if patient data was created
                if patient_data:
                    patient_data = self._validate_patient_dates(patient_data)
                
                return patient_data
                
            except Exception as e:
                self.logger.error(f"Failed to extract patient {patient_index}: {e}")
                return None
        
        # Use multi-patient handler
        return await self.multi_patient_handler.process_multi_patient_document(
            text, extract_single_patient
        )
    
    def _enhance_with_field_detection(
        self, 
        patient_data: PatientData, 
        field_results: Dict[str, Any]
    ) -> PatientData:
        """
        Enhance NuExtract results with field detection.
        
        Args:
            patient_data: Patient data from NuExtract
            field_results: Results from field detection
            
        Returns:
            Enhanced patient data
        """
        # Enhance CPT codes if missing or incomplete
        if not patient_data.cpt_codes and 'cpt_codes' in field_results:
            from src.core.data_schema import CPTCode
            patient_data.cpt_codes = [
                CPTCode(code=code, description="", confidence=0.8)
                for code in field_results['cpt_codes']
            ]
        
        # Enhance ICD-10 codes
        if not patient_data.icd10_codes and 'icd10_codes' in field_results:
            from src.core.data_schema import ICD10Code
            patient_data.icd10_codes = [
                ICD10Code(code=code, description="", confidence=0.8)
                for code in field_results['icd10_codes']
            ]
        
        # Enhance dates
        if not patient_data.date_of_birth and 'dates' in field_results:
            dates = field_results['dates']
            if dates:
                # Process all detected dates
                standardized_dates = []
                date_types = {}
                
                # Standardize all dates found
                for date_obj in dates:
                    if isinstance(date_obj, dict) and 'value' in date_obj:
                        date_str = date_obj['value']
                        std_date = self.date_validator.standardize_date(date_str)
                        if std_date:
                            standardized_dates.append(std_date)
                            
                            # Check context to determine date type
                            if 'context' in date_obj:
                                context = date_obj['context'].lower()
                                
                                # Identify date type based on context
                                if any(term in context for term in ["birth", "dob", "born"]):
                                    date_types[std_date] = "date_of_birth"
                                elif any(term in context for term in ["service", "visit", "appointment", "exam"]):
                                    date_types[std_date] = "date_of_service"
                                elif any(term in context for term in ["claim", "submission", "filed"]):
                                    date_types[std_date] = "claim_date"
                    elif isinstance(date_obj, str):
                        std_date = self.date_validator.standardize_date(date_obj)
                        if std_date:
                            standardized_dates.append(std_date)
                
                # Assign dates based on identified types
                for std_date, date_type in date_types.items():
                    if date_type == "date_of_birth" and not patient_data.date_of_birth:
                        patient_data.date_of_birth = std_date
                    elif date_type == "date_of_service" and not patient_data.date_of_service:
                        patient_data.date_of_service = std_date
                    elif date_type == "claim_date" and hasattr(patient_data, 'claim_info') and patient_data.claim_info:
                        if not patient_data.claim_info.claim_date:
                            patient_data.claim_info.claim_date = std_date
                
                # If we couldn't determine specific date types but have dates, make best guess
                if not patient_data.date_of_birth and standardized_dates:
                    # Assume first date might be DOB (this is heuristic)
                    patient_data.date_of_birth = standardized_dates[0]
        
        # Enhance financial data
        if 'amounts' in field_results and field_results['amounts']:
            if not patient_data.financial_info:
                from src.core.data_schema import FinancialInfo
                patient_data.financial_info = FinancialInfo()
            
            amounts = field_results['amounts']
            if not patient_data.financial_info.total_charges and amounts:
                patient_data.financial_info.total_charges = amounts[0]
        
        # Enhance PHI detection confidence
        if 'phi_detected' in field_results:
            patient_data.phi_detected = field_results['phi_detected']
        
        return patient_data
    
    def _validate_patient_dates(self, patient_data: PatientData) -> PatientData:
        """
        Validate and standardize dates within patient data.
        
        Args:
            patient_data: Patient data to validate
            
        Returns:
            Patient data with validated and standardized dates
        """
        # Collect all dates from patient data
        dates = {}
        
        # Extract dates from patient data fields
        if patient_data.date_of_birth:
            dates["date_of_birth"] = patient_data.date_of_birth
            
        if patient_data.date_of_service:
            dates["date_of_service"] = patient_data.date_of_service
            
        if hasattr(patient_data, 'service_info') and patient_data.service_info and hasattr(patient_data.service_info, 'service_date') and patient_data.service_info.service_date:
            dates["service_date"] = patient_data.service_info.service_date
            
        if hasattr(patient_data, 'claim_info') and patient_data.claim_info:
            if hasattr(patient_data.claim_info, 'claim_date') and patient_data.claim_info.claim_date:
                dates["claim_date"] = patient_data.claim_info.claim_date
            if hasattr(patient_data.claim_info, 'submission_date') and patient_data.claim_info.submission_date:
                dates["submission_date"] = patient_data.claim_info.submission_date
                
        # Skip validation if no dates are present
        if not dates:
            return patient_data
            
        # Standardize all dates
        standardized_dates = {}
        for field, date_str in dates.items():
            std_date = self.date_validator.standardize_date(date_str)
            if std_date:
                standardized_dates[field] = std_date
                
                # Update the patient data with standardized dates
                if field == "date_of_birth":
                    patient_data.date_of_birth = std_date
                elif field == "date_of_service":
                    patient_data.date_of_service = std_date
                elif field == "service_date" and hasattr(patient_data, 'service_info') and patient_data.service_info:
                    patient_data.service_info.service_date = std_date
                elif field == "claim_date" and hasattr(patient_data, 'claim_info') and patient_data.claim_info:
                    patient_data.claim_info.claim_date = std_date
                elif field == "submission_date" and hasattr(patient_data, 'claim_info') and patient_data.claim_info:
                    patient_data.claim_info.submission_date = std_date
        
        # Validate date consistency
        if len(standardized_dates) > 1:
            validity = self.date_validator.validate_date_consistency(standardized_dates)
            
            # Log any inconsistencies
            for field, is_valid in validity.items():
                if not is_valid:
                    self.logger.warning(
                        f"Date inconsistency detected in {field}: {dates.get(field)} "
                        f"(standardized: {standardized_dates.get(field)})"
                    )
        
        return patient_data
    
    def _create_patient_from_fields(self, text: str, patient_index: int) -> Optional[PatientData]:
        """
        Create patient data using only field detection as fallback.
        
        Args:
            text: Input text
            patient_index: Patient index
            
        Returns:
            Patient data or None if insufficient data
        """
        field_results = self.field_detector.detect_all_fields(text)
        
        # Check if we have enough data to create a patient record
        has_cpt = 'cpt_codes' in field_results and field_results['cpt_codes']
        has_name = 'names' in field_results and field_results['names']
        has_dates = 'dates' in field_results and field_results['dates']
        
        if not (has_cpt or has_name or has_dates):
            self.logger.warning(f"Insufficient data for patient {patient_index}")
            return None
        
        try:
            from src.core.data_schema import CPTCode, ICD10Code, ServiceInfo, FinancialInfo
            
            # Create basic patient data
            patient_data = PatientData(patient_index=patient_index)
            
            # Set name if available
            if has_name:
                names = field_results['names']
                if names:
                    name_parts = names[0].split()
                    if len(name_parts) >= 2:
                        patient_data.first_name = name_parts[0]
                        patient_data.last_name = " ".join(name_parts[1:])
                    else:
                        patient_data.first_name = names[0]
            
            # Set CPT codes
            if has_cpt:
                patient_data.cpt_codes = [
                    CPTCode(code=code, description="", confidence=0.7)
                    for code in field_results['cpt_codes']
                ]
            
            # Set ICD-10 codes
            if 'icd10_codes' in field_results:
                patient_data.icd10_codes = [
                    ICD10Code(code=code, description="", confidence=0.7)
                    for code in field_results['icd10_codes']
                ]
            
            # Set dates
            if has_dates:
                dates = field_results['dates']
                if dates:
                    # Process detected dates and contexts
                    date_values = []
                    date_types = {}
                    
                    for date_obj in dates:
                        # Extract date value and context for analysis
                        if isinstance(date_obj, DetectionResult):
                            date_str = date_obj.value
                            context = date_obj.context.lower() if date_obj.context else ""
                            
                            # Standardize the date
                            std_date = self.date_validator.standardize_date(date_str)
                            if std_date:
                                date_values.append(std_date)
                                
                                # Identify date type based on context
                                if any(term in context for term in ["birth", "dob", "born"]):
                                    date_types[std_date] = "date_of_birth"
                                elif any(term in context for term in ["service", "visit", "appointment", "exam"]):
                                    date_types[std_date] = "date_of_service"
                        elif isinstance(date_obj, str):
                            # Handle string dates for backward compatibility
                            std_date = self.date_validator.standardize_date(date_obj)
                            if std_date:
                                date_values.append(std_date)
                    
                    # Assign dates based on identified types
                    for std_date, date_type in date_types.items():
                        if date_type == "date_of_birth":
                            patient_data.date_of_birth = std_date
                        elif date_type == "date_of_service":
                            if not patient_data.service_info:
                                patient_data.service_info = ServiceInfo()
                            patient_data.service_info.date_of_service = std_date
                    
                    # If we couldn't determine specific date types but have dates, make best guess
                    if date_values and not patient_data.date_of_birth:
                        patient_data.date_of_birth = date_values[0]
                        
                        if len(date_values) > 1 and not (patient_data.service_info and patient_data.service_info.date_of_service):
                            patient_data.service_info = ServiceInfo(
                                date_of_service=date_values[1]
                            )
            
            # Set financial information
            if 'amounts' in field_results and field_results['amounts']:
                patient_data.financial_info = FinancialInfo(
                    total_charges=field_results['amounts'][0]
                )
            
            # Set PHI detection
            patient_data.phi_detected = field_results.get('phi_detected', False)
            
            # Set extraction confidence
            patient_data.extraction_confidence = 0.6  # Lower confidence for field-only extraction
            
            return patient_data
            
        except Exception as e:
            self.logger.error(f"Failed to create patient from fields: {e}")
            return None
    
    def _calculate_overall_confidence(self, patients: List[PatientData]) -> float:
        """Calculate overall extraction confidence."""
        if not patients:
            return 0.0
        
        confidences = [p.extraction_confidence for p in patients if p.extraction_confidence is not None]
        
        if not confidences:
            return 0.5  # Default confidence
        
        return sum(confidences) / len(confidences)
    
    async def _ensure_models_loaded(self) -> None:
        """Ensure all required models are loaded before extraction."""
        if self._models_loaded:
            return
        
        try:
            self.logger.info("Loading required models...")
            
            # Load OCR model
            await self.ocr_engine.load_models()
            
            # Load NuExtract model
            await self.nuextract_engine.load_model()
            
            self._models_loaded = True
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Failed to load required models: {e}")
    
    async def batch_extract(self, file_paths: List[str]) -> List[ExtractionResults]:
        """
        Extract data from multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction for {len(file_paths)} files")
        
        # Ensure models are loaded before batch processing
        await self._ensure_models_loaded()
        
        results = []
        
        for file_path in file_paths:
            try:
                result = await self.extract_from_file(file_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch extraction failed for {file_path}: {e}")
                # Create error result
                error_result = ExtractionResults(
                    file_path=file_path,
                    extraction_timestamp=datetime.now(),
                    total_patients=0,
                    patients=[],
                    extraction_confidence=0.0,
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        self.logger.info(f"Batch extraction completed. Processed {len(results)} files")
        return results


class ExtractionEngine:
    """
    High-level extraction engine interface.
    
    This is the main class that users will interact with for extracting
    structured data from medical superbills.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize extraction engine.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.logger = get_logger(__name__)
        self.pipeline = ExtractionPipeline(self.config)
        
        self.logger.info("Medical Superbill Extraction Engine initialized")
    
    async def extract_from_file(self, file_path: str) -> ExtractionResults:
        """
        Extract structured data from a superbill file.
        
        Args:
            file_path: Path to the input file (PDF, image, etc.)
            
        Returns:
            Extraction results containing patient data
        """
        return await self.pipeline.extract_from_file(file_path)
    
    async def extract_from_text(self, text: str, source_name: str = "text_input") -> ExtractionResults:
        """
        Extract structured data from text.
        
        Args:
            text: Input text containing superbill data
            source_name: Optional name for the source
            
        Returns:
            Extraction results containing patient data
        """
        return await self.pipeline.extract_from_text(text, source_name)
    
    async def extract_batch(self, file_paths: List[str]) -> List[ExtractionResults]:
        """
        Extract data from multiple files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of extraction results
        """
        return await self.pipeline.batch_extract(file_paths)
    
    def export_to_json(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export extraction results to JSON.
        
        Args:
            results: Extraction results
            output_path: Output file path
        """
        try:
            # Convert to JSON-serializable format
            json_data = {
                'file_path': results.file_path,
                'extraction_timestamp': results.extraction_timestamp.isoformat(),
                'total_patients': results.total_patients,
                'extraction_confidence': results.extraction_confidence,
                'metadata': results.metadata,
                'patients': [patient.model_dump() for patient in results.patients]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Results exported to JSON: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            raise
    
    def export_to_csv(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export extraction results to CSV.
        
        Args:
            results: Extraction results
            output_path: Output file path
        """
        try:
            import pandas as pd
            
            # Flatten patient data for CSV export
            rows = []
            
            for patient in results.patients:
                base_row = {
                    'patient_index': patient.patient_index,
                    'first_name': patient.first_name,
                    'last_name': patient.last_name,
                    'date_of_birth': patient.date_of_birth.isoformat() if patient.date_of_birth else None,
                    'patient_id': patient.patient_id,
                    'extraction_confidence': patient.extraction_confidence,
                    'phi_detected': patient.phi_detected
                }
                
                # Add service information
                if patient.service_info:
                    base_row.update({
                        'service_date': patient.service_info.date_of_service.isoformat() if patient.service_info.date_of_service else None,
                        'provider_name': patient.service_info.provider_name,
                        'facility_name': patient.service_info.facility_name
                    })
                
                # Add financial information
                if patient.financial_info:
                    base_row.update({
                        'total_charges': patient.financial_info.total_charges,
                        'insurance_payment': patient.financial_info.insurance_payment,
                        'patient_payment': patient.financial_info.patient_payment
                    })
                
                # Add CPT codes (create separate rows for each CPT)
                if patient.cpt_codes:
                    for cpt in patient.cpt_codes:
                        row = base_row.copy()
                        row.update({
                            'cpt_code': cpt.code,
                            'cpt_description': cpt.description,
                            'cpt_confidence': cpt.confidence
                        })
                        rows.append(row)
                else:
                    rows.append(base_row)
            
            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Results exported to CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
            raise


# Convenience functions for easy usage
async def extract_from_file(file_path: str, config_path: Optional[str] = None) -> ExtractionResults:
    """
    Convenience function to extract data from a file.
    
    Args:
        file_path: Path to the input file
        config_path: Optional configuration file path
        
    Returns:
        Extraction results
    """
    engine = ExtractionEngine(config_path)
    return await engine.extract_from_file(file_path)


async def extract_from_text(text: str, config_path: Optional[str] = None) -> ExtractionResults:
    """
    Convenience function to extract data from text.
    
    Args:
        text: Input text
        config_path: Optional configuration file path
        
    Returns:
        Extraction results
    """
    engine = ExtractionEngine(config_path)
    return await engine.extract_from_text(text)


async def extract_batch(file_paths: List[str], config_path: Optional[str] = None) -> List[ExtractionResults]:
    """
    Convenience function to extract data from multiple files.
    
    Args:
        file_paths: List of file paths
        config_path: Optional configuration file path
        
    Returns:
        List of extraction results
    """
    engine = ExtractionEngine(config_path)
    return await engine.extract_batch(file_paths)
