"""
Main Data Extraction Engine

Integrates OCR, field detection, NuExtract processing, and multi-patient handling
to provide a complete extraction pipeline for medical superbills.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import PatientData, ExtractionResults
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_engine import UnifiedOCREngine
from src.extractors.field_detector import FieldDetectionEngine, DetectionResult
from src.extractors.nuextract_engine import NuExtractEngine
from src.extractors.multi_patient_handler import MultiPatientHandler
from src.extractors.mixed_patient_validator import MixedPatientValidator
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
        self.ocr_engine = UnifiedOCREngine(config)  # Use new unified engine
        self.field_detector = FieldDetectionEngine(config)
        self.nuextract_engine = NuExtractEngine(config)
        self.multi_patient_handler = MultiPatientHandler(config)
        self.mixed_patient_validator = MixedPatientValidator(config)
        self.date_validator = DateValidator(config)
        self.data_validator = DataValidator(config)
        
        # Configuration
        self.extraction_config = config.get("extraction", {})
        self.max_retries = self.extraction_config.get("max_retries", 3)
        self.confidence_threshold = self.extraction_config.get("confidence_threshold", 0.7)
        
        # Track model loading status
        self._models_loaded = False
        
    async def _initialize_models(self):
        """
        Initialize all required models sequentially to optimize VRAM usage.
        
        Sequential Loading Strategy:
        1. Load Nanonets OCR model first (optimized for GPU)
        2. Clear GPU cache and optimize memory
        3. Load NuExtract model second (ensuring optimal memory usage)
        4. Monitor GPU memory throughout the process
        """
        if self._models_loaded:
            return
        
        import torch
        import gc
        
        self.logger.info("🚀 Initializing models sequentially for optimal VRAM usage...")
        start_time = time.time()
        
        # Initial GPU memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"Initial GPU memory: {initial_memory:.2f}GB")
        
        try:
            # Phase 1: Load OCR model (Nanonets)
            self.logger.info("📄 Phase 1: Loading Nanonets OCR model...")
            await self.ocr_engine.load_models()
            
            if torch.cuda.is_available():
                ocr_memory = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"✓ OCR model loaded - GPU memory: {ocr_memory:.2f}GB")
                
            # Memory optimization after OCR loading
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                self.logger.warning(f"Memory optimization warning: {e}")
            
            self.logger.info("✅ OCR model (Nanonets) loaded successfully")
            
            # Brief pause to stabilize memory and ensure CUDA operations are complete
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
                await asyncio.sleep(2)
            except Exception as e:
                self.logger.warning(f"CUDA synchronization warning: {e}")
                await asyncio.sleep(2)  # Still wait even if synchronization fails            # Phase 2: Load extraction model (NuExtract)
            self.logger.info("🧠 Phase 2: Loading NuExtract model...")
            await self.nuextract_engine.load_model()
            
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                total_used = final_memory - initial_memory
                self.logger.info(f"✓ NuExtract model loaded - GPU memory: {final_memory:.2f}GB (total used: {total_used:.2f}GB)")
                
                # Final memory optimization
                torch.cuda.empty_cache()
                gc.collect()
            
            self.logger.info("✅ NuExtract model loaded successfully")
            
            # Mark as loaded
            self._models_loaded = True
            
            # Final statistics
            total_time = time.time() - start_time
            self.logger.info(f"🎉 All models initialized sequentially in {total_time:.2f}s")
            
            if torch.cuda.is_available():
                final_optimized_memory = torch.cuda.memory_allocated() / 1024**3
                self.logger.info(f"Final optimized GPU memory: {final_optimized_memory:.2f}GB")
        
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize models sequentially: {e}")
            # Clean up on failure with proper error handling
            await self._safe_cleanup_resources()
            # Clean up any partially loaded models
            if hasattr(self, 'ocr_engine') and self.ocr_engine:
                try:
                    await self.ocr_engine.cleanup()
                except Exception as cleanup_err:
                    self.logger.warning(f"OCR engine cleanup failed: {cleanup_err}")
            if hasattr(self, 'nuextract_engine') and self.nuextract_engine:
                try:
                    await self.nuextract_engine.cleanup()
                except Exception as cleanup_err:
                    self.logger.warning(f"NuExtract engine cleanup failed: {cleanup_err}")
            raise

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
            await self._initialize_models()
            
            # Step 1: Process document with chunking
            processed_pages = await self.document_processor.process_pdf(file_path)
            
            if not processed_pages:
                raise ValueError("No pages could be processed from the document")
            
            # Step 2: Extract text using chunk-based OCR for better accuracy
            all_patients = []
            total_chunks = sum(page['chunk_count'] for page in processed_pages)
            self.logger.info(f"Processing {len(processed_pages)} pages with {total_chunks} total chunks")
            
            # Process each page and its chunks
            for page_idx, page_data in enumerate(processed_pages):
                page_num = page_data['page_number']
                chunks = page_data['chunks']
                strategy = page_data['processing_strategy']
                
                self.logger.info(f"Processing page {page_num} using {strategy} strategy with {len(chunks)} chunks")
                
                # Extract text from each chunk for better OCR accuracy
                chunk_texts = []
                for chunk_idx, chunk in enumerate(chunks):
                    self.logger.debug(f"Processing chunk {chunk_idx+1}/{len(chunks)} on page {page_num}")
                    
                    # Extract text from this chunk
                    chunk_ocr_result = await self.ocr_engine.extract_text(chunk['image'])
                    
                    if chunk_ocr_result.text.strip():
                        chunk_texts.append({
                            'text': chunk_ocr_result.text,
                            'bbox': chunk['bbox'],
                            'confidence': chunk_ocr_result.confidence,
                            'chunk_index': chunk_idx,
                            'estimated_tokens': chunk['estimated_tokens']
                        })
                        self.logger.debug(f"  Chunk {chunk_idx+1}: extracted {len(chunk_ocr_result.text)} chars, confidence: {chunk_ocr_result.confidence:.2f}")
                    else:
                        self.logger.warning(f"  Chunk {chunk_idx+1}: no text extracted")
                
                # Combine chunk texts intelligently
                if chunk_texts:
                    # Sort chunks by position (top-to-bottom, left-to-right)
                    chunk_texts.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))  # Sort by Y then X
                    
                    # Combine texts with spatial awareness
                    combined_text = self._combine_chunk_texts(chunk_texts)
                    
                    self.logger.info(f"Page {page_num}: combined {len(chunk_texts)} chunks into {len(combined_text)} characters")
                    
                    # Extract patients from the combined page text
                    page_patients = await self._extract_patient_data(
                        combined_text, 
                        page_number=page_num,
                        total_pages=len(processed_pages)
                    )
                    
                    # Add enhanced metadata to each patient
                    for patient in page_patients:
                        patient.page_number = page_num
                        patient.source_page_text = combined_text[:500]  # Store truncated source text
                        patient.chunk_count = len(chunks)
                        patient.processing_strategy = strategy
                    
                    all_patients.extend(page_patients)
                    self.logger.info(f"Page {page_num}: extracted {len(page_patients)} patients")
                else:
                    self.logger.warning(f"Page {page_num}: no text extracted from any chunks")
            
            # Validate that we got some text
            if not all_patients and total_chunks > 0:
                self.logger.error("No patients extracted despite having processed chunks")
                # This could indicate an issue with the extraction logic
            
            # Deduplicate patients across pages
            patients = await self._deduplicate_patients_across_pages(all_patients)
            
            # Step 4: Create results
            results = ExtractionResults(
                success=True,
                file_path=file_path,
                extraction_timestamp=datetime.now(),
                total_patients=len(patients),
                patients=patients,
                extraction_confidence=self._calculate_overall_confidence(patients),
                metadata={
                    'total_pages': len(images),
                    'total_text_length': sum(len(r.text) for r in ocr_results),
                    'processing_method': 'multi_patient_page_based_pipeline'
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
            await self._initialize_models()
            
            # Extract structured data
            patients = await self._extract_patient_data(text)
            
            # Create results
            results = ExtractionResults(
                success=True,
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
    
    async def _extract_patient_data(
        self, 
        text: str, 
        page_number: int = 1,
        total_pages: int = 1
    ) -> List[PatientData]:
        """
        Extract patient data from text using multi-patient handling.
        
        Args:
            text: Input text
            page_number: Current page number (for multi-page documents)
            total_pages: Total number of pages
            
        Returns:
            List of patient data
        """
        async def extract_single_patient(segment_text: str, patient_index: int) -> Optional[PatientData]:
            """Extract data for a single patient segment."""
            try:
                # Method 1: NuExtract structured extraction (text-only mode)
                # Use NuExtract for structured data extraction from text (not vision processing)
                structured_data = await self.nuextract_engine.extract_structured_data(segment_text, "medical_superbill")
                nuextract_patient = self._convert_structured_to_patient_data(structured_data, patient_index) if structured_data else None
                
                if nuextract_patient:
                    # Enhance with field detection
                    field_results = self.field_detector.detect_all_fields(segment_text)
                    enhanced_data = self._enhance_with_field_detection(nuextract_patient, field_results)
                    
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
            text, extract_single_patient, page_number, total_pages
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
            if not patient_data.financial_info.total_charge and amounts:
                patient_data.financial_info.total_charge = amounts[0]
        
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
            from src.core.data_schema import CPTCode, ICD10Code, ServiceInfo, FinancialInfo, FieldConfidence
            
            # Extract name first since it's required
            first_name = "Unknown"
            last_name = "Patient"
            
            if has_name:
                names = field_results['names']
                if names:
                    name_parts = names[0].split()
                    if len(name_parts) >= 2:
                        first_name = name_parts[0]
                        last_name = " ".join(name_parts[1:])
                    else:
                        first_name = names[0]
                        last_name = ""
            
            # Create basic patient data with required fields
            patient_data = PatientData(
                first_name=first_name,
                last_name=last_name,
                patient_index=patient_index
            )
            
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
                            # Set both the direct field and service_info field
                            patient_data.date_of_service = std_date
                            if not patient_data.service_info:
                                patient_data.service_info = ServiceInfo()
                            patient_data.service_info.date_of_service = std_date
                    
                    # If we couldn't determine specific date types but have dates, make best guess
                    if date_values and not patient_data.date_of_birth:
                        patient_data.date_of_birth = date_values[0]
                        
                        if len(date_values) > 1 and not patient_data.date_of_service:
                            patient_data.date_of_service = date_values[1]
                            if not patient_data.service_info:
                                patient_data.service_info = ServiceInfo()
                            patient_data.service_info.date_of_service = date_values[1]
            
            # Set financial information
            if 'amounts' in field_results and field_results['amounts']:
                patient_data.financial_info = FinancialInfo(
                    total_charge=field_results['amounts'][0]
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
    
    def _convert_structured_to_patient_data(self, structured_data: Dict[str, Any], patient_index: int) -> Optional[PatientData]:
        """Convert NuExtract structured data to PatientData format."""
        try:
            if not structured_data or "patients" not in structured_data:
                return None
            
            patients = structured_data["patients"]
            if not patients or len(patients) <= patient_index:
                return None
            
            patient_data = patients[patient_index]
            patient_info = patient_data.get("patient_info", {})
            
            # Create PatientData using the conversion method from NuExtract engine
            return self.nuextract_engine._convert_to_patient_data(patient_data, structured_data.get("provider_data"))
            
        except Exception as e:
            self.logger.error(f"Failed to convert structured data to PatientData: {e}")
            return None
    
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
        await self._initialize_models()
        
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
        
    async def _deduplicate_patients_across_pages(self, patients: List[PatientData]) -> List[PatientData]:
        """
        Deduplicate patients found across multiple pages.
        
        Args:
            patients: List of all patients found across all pages
            
        Returns:
            Deduplicated list of patients
        """
        if not patients or len(patients) <= 1:
            return patients
            
        self.logger.info(f"Deduplicating {len(patients)} patients found across pages")
        
        # Group patients by similarity
        unique_patients = []
        duplicate_indices = set()
        
        for i, patient1 in enumerate(patients):
            if i in duplicate_indices:
                continue
                
            # Find potential duplicates
            duplicates = []
            
            for j, patient2 in enumerate(patients):
                if i == j or j in duplicate_indices:
                    continue
                    
                # Calculate similarity based on key attributes
                similarity = self._calculate_patient_similarity(patient1, patient2)
                
                # If similarity is high, consider it a duplicate
                if similarity > 0.7:  # Threshold can be adjusted
                    duplicates.append((j, similarity, patient2))
                    duplicate_indices.add(j)
            
            # Merge duplicates (prioritize data from pages with more complete info)
            if duplicates:
                merged_patient = self._merge_duplicate_patients(patient1, [d[2] for d in duplicates])
                unique_patients.append(merged_patient)
            else:
                unique_patients.append(patient1)
        
        self.logger.info(f"After deduplication: {len(unique_patients)} unique patients")
        return unique_patients
    
    def _calculate_patient_similarity(self, patient1: PatientData, patient2: PatientData) -> float:
        """
        Calculate similarity between two patients.
        
        Args:
            patient1: First patient
            patient2: Second patient
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not patient1 or not patient2:
            return 0.0
            
        similarities = []
        
        # Name similarity
        if hasattr(patient1, 'first_name') and hasattr(patient2, 'first_name') and patient1.first_name and patient2.first_name:
            name_sim = self._calculate_string_similarity(patient1.first_name, patient2.first_name)
            similarities.append(name_sim * 0.25)  # 25% weight
        
        if patient1.last_name and patient2.last_name:
            name_sim = self._calculate_string_similarity(patient1.last_name, patient2.last_name)
            similarities.append(name_sim * 0.25)  # 25% weight
        
        # DOB similarity
        if patient1.date_of_birth and patient2.date_of_birth:
            dob_sim = 1.0 if patient1.date_of_birth == patient2.date_of_birth else 0.0
            similarities.append(dob_sim * 0.3)  # 30% weight
        
        # Patient ID similarity
        if patient1.patient_id and patient2.patient_id:
            id_sim = self._calculate_string_similarity(patient1.patient_id, patient2.patient_id)
            similarities.append(id_sim * 0.2)  # 20% weight
        
        return sum(similarities) / max(1, len(similarities))
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using difflib."""
        import difflib
        return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _merge_duplicate_patients(self, primary_patient: PatientData, duplicate_patients: List[PatientData]) -> PatientData:
        """
        Merge duplicate patients, prioritizing more complete information.
        
        Args:
            primary_patient: Primary patient record
            duplicate_patients: List of duplicate patient records
            
        Returns:
            Merged patient record
        """
        # Start with the primary patient
        merged = primary_patient
        
        # Track the most complete page
        page_completeness = self._calculate_patient_completeness(primary_patient)
        
        # Initialize source_pages attribute if it doesn't exist
        if not hasattr(merged, 'source_pages'):
            merged.source_pages = []
            
        if hasattr(primary_patient, 'page_number'):
            merged.source_pages.append(primary_patient.page_number)
        
        # Merge data from duplicates
        for duplicate in duplicate_patients:
            duplicate_completeness = self._calculate_patient_completeness(duplicate)
            
            # Add this page to the source pages
            if hasattr(duplicate, 'page_number'):
                merged.source_pages.append(duplicate.page_number)
            
            # Check for signs of mixed patient data before merging
            mixed_patient_issues = self.mixed_patient_validator.validate_for_mixed_patient_data(duplicate)
            if mixed_patient_issues:
                self.logger.warning(f"Possible mixed patient data detected when merging patients: {mixed_patient_issues}")
                # Flag the record as potentially containing mixed data
                if not hasattr(merged, 'validation_warnings'):
                    merged.validation_warnings = []
                merged.validation_warnings.extend(mixed_patient_issues)
                # Continue with merging but with caution
            
            # If the duplicate has more complete data, prefer it
            if duplicate_completeness > page_completeness:
                # Keep only non-empty fields from the primary record
                self._merge_non_empty_fields(merged, duplicate)
                page_completeness = duplicate_completeness
            else:
                # Only copy fields that are empty in the primary record
                self._copy_missing_fields(merged, duplicate)
                
            # Always merge lists (like CPT codes, ICD codes, etc.)
            self._merge_list_fields(merged, duplicate)
        
        return merged
    
    def _calculate_patient_completeness(self, patient: PatientData) -> float:
        """Calculate completeness score for a patient record."""
        total_fields = 0
        filled_fields = 0
        
        # Count basic fields
        for field in ['first_name', 'last_name', 'date_of_birth', 'gender', 
                      'address', 'phone', 'email', 'patient_id']:
            total_fields += 1
            if hasattr(patient, field) and getattr(patient, field, None):
                filled_fields += 1
        
        # Count code lists
        for list_field in ['cpt_codes', 'icd10_codes']:
            total_fields += 1
            if hasattr(patient, list_field) and getattr(patient, list_field, None):
                filled_fields += 1
        
        # Count nested objects
        for nested_field in ['service_info', 'financial_info', 'provider_info']:
            if hasattr(patient, nested_field) and getattr(patient, nested_field, None):
                nested_obj = getattr(patient, nested_field)
                for attr in dir(nested_obj):
                    if not attr.startswith('_') and not callable(getattr(nested_obj, attr)):
                        total_fields += 1
                        if getattr(nested_obj, attr, None):
                            filled_fields += 1
        
        return filled_fields / max(1, total_fields)
    
    def _merge_non_empty_fields(self, target: PatientData, source: PatientData) -> None:
        """Copy all non-empty fields from source to target."""
        # Skip special fields and lists
        skip_fields = {'page_number', 'source_pages', 'source_page_text', 
                      'cpt_codes', 'icd10_codes', 'validation_errors'}
        
        # Create a copy of attributes to iterate over to avoid dictionary mutation issues
        for attr in list(dir(source)):
            if attr.startswith('_') or callable(getattr(source, attr)) or attr in skip_fields:
                continue
                
            source_value = getattr(source, attr, None)
            if source_value:
                setattr(target, attr, source_value)
    
    def _copy_missing_fields(self, target: PatientData, source: PatientData) -> None:
        """Copy fields from source to target only if target's field is empty."""
        # Skip special fields and lists
        skip_fields = ['page_number', 'source_pages', 'source_page_text', 
                       'cpt_codes', 'icd10_codes', 'validation_errors']
        
        for attr in dir(source):
            if attr.startswith('_') or callable(getattr(source, attr)) or attr in skip_fields:
                continue
                
            target_value = getattr(target, attr, None)
            source_value = getattr(source, attr, None)
            
            if not target_value and source_value:
                setattr(target, attr, source_value)
    
    def _combine_chunk_texts(self, chunk_texts: List[Dict[str, Any]]) -> str:
        """
        Intelligently combine text from multiple chunks with spatial awareness.
        
        Args:
            chunk_texts: List of chunk text data with bbox information
            
        Returns:
            Combined text string
        """
        if not chunk_texts:
            return ""
        
        if len(chunk_texts) == 1:
            return chunk_texts[0]['text']
        
        # Sort chunks by position (already done before calling this method)
        combined_lines = []
        
        for i, chunk in enumerate(chunk_texts):
            text = chunk['text'].strip()
            if not text:
                continue
                
            # Add chunk boundary markers for debugging (optional)
            if self.logger.level <= 10:  # DEBUG level
                combined_lines.append(f"[CHUNK {chunk['chunk_index']+1}]")
            
            # Split text into lines and add them
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    combined_lines.append(line)
            
            # Add spacing between chunks if they're not adjacent
            if i < len(chunk_texts) - 1:
                next_chunk = chunk_texts[i + 1]
                current_bbox = chunk['bbox']
                next_bbox = next_chunk['bbox']
                
                # Check if chunks are vertically separated (different rows)
                vertical_gap = next_bbox[1] - (current_bbox[1] + current_bbox[3])
                if vertical_gap > 20:  # Significant vertical gap
                    combined_lines.append("")  # Add blank line
        
        return '\n'.join(combined_lines)

    def _merge_list_fields(self, target: PatientData, source: PatientData) -> None:
        """Merge list fields like CPT codes and ICD codes."""
        # Handle CPT codes
        if hasattr(source, 'cpt_codes') and source.cpt_codes:
            if not hasattr(target, 'cpt_codes') or not target.cpt_codes:
                target.cpt_codes = []
                
            # Add non-duplicate codes
            existing_codes = {code.code for code in target.cpt_codes}
            for code in source.cpt_codes:
                if code.code not in existing_codes:
                    target.cpt_codes.append(code)
                    existing_codes.add(code.code)
        
        # Handle ICD10 codes
        if hasattr(source, 'icd10_codes') and source.icd10_codes:
            if not hasattr(target, 'icd10_codes') or not target.icd10_codes:
                target.icd10_codes = []
                
            # Add non-duplicate codes
            existing_codes = {code.code for code in target.icd10_codes}
            for code in source.icd10_codes:
                if code.code not in existing_codes:
                    target.icd10_codes.append(code)
                    existing_codes.add(code.code)


class ExtractionEngine:
    """
    High-level extraction engine interface.
    
    This is the main class that users will interact with for extracting
    structured data from medical superbills.
    """
    
    async def _safe_cleanup_resources(self):
        """Safely clean up GPU and system resources with comprehensive error handling."""
        cleanup_tasks = []
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                self.logger.debug("GPU cache cleared")
            except Exception as e:
                self.logger.warning(f"GPU cache cleanup failed: {e}")
        
        # System memory cleanup
        try:
            import gc
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")
        except Exception as e:
            self.logger.warning(f"Garbage collection failed: {e}")
        
        # Additional CUDA cleanup if available
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                # Force memory defragmentation
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"CUDA synchronization failed: {e}")
    
    def __init__(self, config: Optional[ConfigManager] = None, config_path: Optional[str] = None):
        """
        Initialize extraction engine.
        
        Args:
            config: Optional ConfigManager instance.
            config_path: Optional path to configuration file.
        """
        self.config = config or ConfigManager(config_path)
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
                'patients': [patient.to_dict() for patient in results.patients]
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
