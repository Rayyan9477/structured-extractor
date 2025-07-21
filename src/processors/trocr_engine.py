"""
TrOCR Engine for Medical Document Processing

Integrates Microsoft's TrOCR model for text recognition from document images.
Designed to work with the UnifiedOCREngine system.
"""

import asyncio
import torch
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class TrOCREngine:
    """
    TrOCR engine for handwritten and printed text recognition.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the TrOCR engine.
        
        Args:
            config: Configuration manager.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get TrOCR configuration
        self.ocr_config = self.config.get("ocr.trocr", {})
        self.model_name = self.ocr_config.get("model_name", "microsoft/trocr-base-handwritten")
        
        self.device = self._get_device()
        self.logger.info(f"TrOCR engine initialized with device: {self.device}")
        
        # Model components
        self.model = None
        self.processor = None
        self.models_loaded = False
        
    def _get_device(self) -> str:
        """Get the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def load_models(self) -> None:
        """Load the TrOCR model and processor."""
        if self.models_loaded:
            return
            
        self.logger.info(f"Loading TrOCR model: {self.model_name}")
        
        try:
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()
            
            self.models_loaded = True
            self.logger.info("TrOCR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load TrOCR model: {e}")

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using TrOCR.
        
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
            # Process image with TrOCR
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate predictions
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode the predicted token ids to text
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"TrOCR extracted text in {processing_time:.2f}s")
            
            # Calculate confidence (TrOCR doesn't provide confidence scores,
            # so we use a fixed value based on model performance)
            confidence = 0.85
            
            return OCRResult(
                text=predicted_text,
                confidence=confidence,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"TrOCR text extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name=self.model_name, 
                processing_time=time.time() - start_time
            )

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
            
        self.logger.info(f"Starting TrOCR batch processing for {len(images)} images")
        
        results = []
        for i, image in enumerate(images):
            self.logger.debug(f"Processing image {i+1}/{len(images)} with TrOCR")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"TrOCR batch processing completed for {len(images)} images")
        return results
