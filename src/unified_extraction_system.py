"""
Unified Structured Extraction System

Main entry point for the extraction system that integrates multiple OCR and extraction models
to provide accurate structured data extraction from documents.
"""

import asyncio
import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import (
    OCRResult, StructuredData, ExtractionResults,
    FieldConfidence, DocumentMetadata
)
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_ensemble import OCREnsembleEngine
from src.extractors.nuextract_structured_extractor import NuExtractStructuredExtractor
from src.exporters.export_manager import ExportManager


class UnifiedExtractionSystem:
    """
    Unified system for extracting structured data from documents using multiple models.
    Integrates OCR ensemble and NuExtract for optimal extraction results.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the unified extraction system.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize configuration
        self.config = ConfigManager(config_path)
        self.logger = get_logger(__name__)
        
        # Load components
        self.document_processor = DocumentProcessor(self.config)
        self.ocr_engine = OCREnsembleEngine(self.config)
        self.structured_extractor = NuExtractStructuredExtractor(self.config)
        self.export_manager = ExportManager(self.config)
        
        # Configure extraction options
        self.extraction_config = self.config.get("extraction", {})
        self.default_template = self.extraction_config.get("default_template", "default")
        self.confidence_threshold = self.extraction_config.get("confidence_threshold", 0.7)
        
        # Track model loading status
        self._models_loaded = False
        
        self.logger.info("Unified Extraction System initialized")
        
    async def _initialize_models(self):
        """Load all required models."""
        if self._models_loaded:
            return
            
        self.logger.info("Loading all models...")
        
        # Load models in parallel
        await asyncio.gather(
            self.ocr_engine.load_models(),
            self.structured_extractor.load_model()
        )
        
        self._models_loaded = True
        self.logger.info("All models loaded successfully")
    
    async def extract_from_file(
        self,
        file_path: str,
        template_name: Optional[str] = None,
        output_format: str = "json",
        output_path: Optional[str] = None
    ) -> ExtractionResults:
        """
        Extract structured data from a document file.
        
        Args:
            file_path: Path to the input document file
            template_name: Name of extraction template to use
            output_format: Format for output (json, csv, etc.)
            output_path: Path to save output (if None, will not save)
            
        Returns:
            Extraction results
        """
        start_time = time.time()
        self.logger.info(f"Starting extraction from file: {file_path}")
        
        try:
            # Ensure models are loaded
            await self._initialize_models()
            
            # Process document to get image(s)
            images = await self.document_processor.process_document(file_path)
            
            if not images:
                raise ValueError(f"No images extracted from {file_path}")
            
            # Extract text using OCR ensemble for each page
            ocr_results = await self.ocr_engine.extract_text_batch(images)
            
            # Store page texts separately for multi-page processing
            page_texts = [r.text for r in ocr_results]
            
            # Process as multi-page document if more than one page
            if len(page_texts) > 1:
                self.logger.info(f"Processing as multi-page document with {len(page_texts)} pages")
                
                # Import needed components
                from src.extractors.multi_patient_handler import MultiPatientHandler
                from src.extractors.nuextract_engine import NuExtractEngine
                
                # Initialize handlers
                multi_patient_handler = MultiPatientHandler(self.config)
                structured_extractor_engine = NuExtractEngine(self.config)
                
                # Ensure extractor is loaded
                if not hasattr(structured_extractor_engine, 'model') or structured_extractor_engine.model is None:
                    await structured_extractor_engine.load_model()
                
                # Create extraction function
                async def extract_patient_data(text, patient_index):
                    return await structured_extractor_engine.extract_patient_data(text, patient_index)
                
                # Process multi-page document with patient detection across pages
                patient_data_list = await multi_patient_handler.process_multi_page_document(
                    page_texts, 
                    extract_patient_data
                )
                
                # Convert to structured data format
                structured_data = {
                    "patients": [patient.to_dict() for patient in patient_data_list],
                    "meta": {
                        "page_count": len(page_texts),
                        "patient_count": len(patient_data_list)
                    }
                }
                
                # Calculate extraction confidence from patient data
                extraction_confidence = sum(p.extraction_confidence for p in patient_data_list) / len(patient_data_list) if patient_data_list else 0.5
                
                # Create structured data result
                structured_data_result = StructuredData(
                    data=structured_data,
                    confidence=extraction_confidence,
                    model_name=structured_extractor_engine.model_name
                )
                
            else:
                # For single page documents, use the standard extraction
                # Combine all page texts (though there's just one)
                full_text = "\n\n".join(page_texts)
                
                # Extract structured data using template
                template = template_name or self.default_template
                structured_data_result = await self.structured_extractor.extract_structured_data(
                    full_text, 
                    template_name=template
                )
            
            # Combine all page texts for output
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
            
            # Calculate average OCR confidence
            ocr_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results)
            
            # Create document metadata
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            processing_time = time.time() - start_time
            
            metadata = DocumentMetadata(
                file_name=file_name,
                file_path=file_path,
                file_size=file_size,
                page_count=len(images),
                processing_time=processing_time,
                extraction_date=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Create extraction results
            results = ExtractionResults(
                metadata=metadata,
                text=full_text,
                structured_data=structured_data_result.data,
                ocr_confidence=ocr_confidence,
                extraction_confidence=structured_data_result.confidence,
                overall_confidence=self._calculate_overall_confidence(ocr_confidence, structured_data_result.confidence)
            )
            
            # Export results if output path is provided
            if output_path:
                await self.export_results(results, output_path, output_format)
            
            self.logger.info(f"Extraction completed in {processing_time:.2f}s with confidence {results.overall_confidence:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}", exc_info=True)
            raise
    
    async def extract_from_text(
        self,
        text: str,
        template_name: Optional[str] = None,
        output_format: str = "json",
        output_path: Optional[str] = None,
        source_name: str = "text_input"
    ) -> ExtractionResults:
        """
        Extract structured data from text.
        
        Args:
            text: Text to extract from
            template_name: Name of extraction template to use
            output_format: Format for output (json, csv, etc.)
            output_path: Path to save output (if None, will not save)
            source_name: Name to use for the source in metadata
            
        Returns:
            Extraction results
        """
        start_time = time.time()
        self.logger.info(f"Starting extraction from text input")
        
        try:
            # Ensure models are loaded
            await self._initialize_models()
            
            # Extract structured data using template
            template = template_name or self.default_template
            structured_data = await self.structured_extractor.extract_structured_data(
                text, 
                template_name=template
            )
            
            # Create document metadata
            processing_time = time.time() - start_time
            
            metadata = DocumentMetadata(
                file_name=source_name,
                file_path=source_name,
                file_size=len(text),
                page_count=1,
                processing_time=processing_time,
                extraction_date=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Create extraction results
            results = ExtractionResults(
                metadata=metadata,
                text=text,
                structured_data=structured_data.data,
                ocr_confidence=1.0,  # No OCR performed on direct text
                extraction_confidence=structured_data.confidence,
                overall_confidence=structured_data.confidence
            )
            
            # Export results if output path is provided
            if output_path:
                await self.export_results(results, output_path, output_format)
            
            self.logger.info(f"Text extraction completed in {processing_time:.2f}s with confidence {results.overall_confidence:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}", exc_info=True)
            raise
    
    async def batch_extract(
        self,
        file_paths: List[str],
        template_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_format: str = "json"
    ) -> List[ExtractionResults]:
        """
        Process a batch of document files.
        
        Args:
            file_paths: List of file paths to process
            template_name: Name of extraction template to use
            output_dir: Directory to save outputs (if None, will not save)
            output_format: Format for outputs
            
        Returns:
            List of extraction results
        """
        self.logger.info(f"Starting batch extraction for {len(file_paths)} files")
        
        results = []
        for file_path in file_paths:
            try:
                output_path = None
                if output_dir:
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]
                    output_path = os.path.join(output_dir, f"{base_name}.{output_format}")
                
                result = await self.extract_from_file(
                    file_path,
                    template_name=template_name,
                    output_format=output_format,
                    output_path=output_path
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                # Continue with next file
        
        self.logger.info(f"Batch extraction completed for {len(results)}/{len(file_paths)} files")
        return results
    
    async def export_results(
        self,
        results: ExtractionResults,
        output_path: str,
        output_format: str = "json"
    ) -> None:
        """
        Export extraction results to file.
        
        Args:
            results: Extraction results to export
            output_path: Path to save output file
            output_format: Format for output
        """
        try:
            await self.export_manager.export(results, output_path, output_format)
            self.logger.info(f"Results exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}", exc_info=True)
    
    def _calculate_overall_confidence(self, ocr_confidence: float, extraction_confidence: float) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            ocr_confidence: Confidence of OCR results
            extraction_confidence: Confidence of structured extraction
            
        Returns:
            Overall confidence score
        """
        # Weight extraction confidence more heavily
        weights = self.extraction_config.get("confidence_weights", {
            "ocr": 0.3,
            "extraction": 0.7
        })
        
        overall = (
            ocr_confidence * weights["ocr"] +
            extraction_confidence * weights["extraction"]
        )
        
        return overall


async def extract_from_file(
    file_path: str,
    template_name: Optional[str] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> ExtractionResults:
    """
    Convenience function to extract from a file.
    
    Args:
        file_path: Path to input file
        template_name: Name of extraction template
        output_path: Path to save output
        config_path: Path to configuration
        
    Returns:
        Extraction results
    """
    system = UnifiedExtractionSystem(config_path)
    return await system.extract_from_file(
        file_path,
        template_name=template_name,
        output_path=output_path
    )


async def extract_from_text(
    text: str,
    template_name: Optional[str] = None,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> ExtractionResults:
    """
    Convenience function to extract from text.
    
    Args:
        text: Input text
        template_name: Name of extraction template
        output_path: Path to save output
        config_path: Path to configuration
        
    Returns:
        Extraction results
    """
    system = UnifiedExtractionSystem(config_path)
    return await system.extract_from_text(
        text,
        template_name=template_name,
        output_path=output_path
    )


async def batch_extract(
    file_paths: List[str],
    template_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None
) -> List[ExtractionResults]:
    """
    Convenience function to batch extract from files.
    
    Args:
        file_paths: List of input file paths
        template_name: Name of extraction template
        output_dir: Directory to save outputs
        config_path: Path to configuration
        
    Returns:
        List of extraction results
    """
    system = UnifiedExtractionSystem(config_path)
    return await system.batch_extract(
        file_paths,
        template_name=template_name,
        output_dir=output_dir
    ) 