"""
OCR Integration Module for Medical Superbill Extraction

Integrates multiple OCR models and provides a unified interface for text extraction.
This engine can handle both printed and handwritten text by using appropriate models.
"""

import asyncio
import torch
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import re
from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence
from src.models.model_downloader import ModelLoader, ModelPerformanceMetrics


class OCREngine:
    """
    Unified OCR engine that dynamically loads and uses the best model 
    for a given task based on configuration.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the OCR engine.
        
        Args:
            config: Configuration manager.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        self.ocr_config = self.config.get("ocr", {})
        self.model_name = self.ocr_config.get("model_name", "microsoft/trocr-base-handwritten")
        self.batch_size = self.ocr_config.get("batch_size", 4)
        
        self.device = self._get_device()
        self.logger.info(f"OCR engine initialized with device: {self.device}")
        
        self.processor = None
        self.model = None
        self.models_loaded = False
        
    def _get_device(self) -> str:
        """Get the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def load_models(self) -> None:
        """Load the configured OCR model and processor."""
        if self.models_loaded:
            return
            
        self.logger.info(f"Loading OCR model: {self.model_name}")
        
        try:
            # Use AutoProcessor for TrOCR models
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Use AutoModelForCTC for TrOCR models
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            self.models_loaded = True
            
            self.logger.info(f"OCR model '{self.model_name}' loaded successfully.")
            
        except Exception as e:
            self.logger.error(f"Failed to load OCR model '{self.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Could not load OCR model: {e}")

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using the configured OCR model.
        
        Args:
            image: A PIL Image to process.
            
        Returns:
            An OCRResult object with the extracted text and metadata.
        """
        if not self.models_loaded:
            await self.load_models()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        start_time = time.time()
        
        try:
            # Prepare image for the model
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate token IDs
            generated_ids = self.model.generate(pixel_values)
            
            # Decode token IDs to text
            extracted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            cleaned_text = self._clean_text(extracted_text)
            
            # Confidence is not directly available from TrOCR, so we use a default
            confidence = 0.9  # High confidence for successful extraction
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Extracted {len(cleaned_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Text extraction failed for model {self.model_name}: {e}", exc_info=True)
            return OCRResult(text="", confidence=0.0, model_name=self.model_name, processing_time=0.0)

    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process.
            
        Returns:
            A list of OCRResult objects.
        """
        if not self.models_loaded:
            await self.load_models()
            
        self.logger.info(f"Starting batch OCR for {len(images)} images.")
        
        results = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            start_time = time.time()
            
            try:
                # Pre-process batch
                pixel_values = self.processor(images=batch_images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate text for the batch
                generated_ids = self.model.generate(pixel_values)
                
                # Decode batch results
                batch_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                processing_time = time.time() - start_time
                
                for text in batch_texts:
                    cleaned_text = self._clean_text(text)
                    results.append(OCRResult(
                        text=cleaned_text,
                        confidence=0.9,
                        model_name=self.model_name,
                        processing_time=processing_time / len(batch_images)
                    ))
                    
            except Exception as e:
                self.logger.error(f"Batch OCR failed: {e}", exc_info=True)
                # Add empty results for the failed batch
                for _ in batch_images:
                    results.append(OCRResult(text="", confidence=0.0, model_name=self.model_name))

        self.logger.info(f"Batch OCR completed for {len(images)} images.")
        return results

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: The raw extracted text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Correct common OCR errors if needed
        # (e.g., text = text.replace("I", "1"))
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
