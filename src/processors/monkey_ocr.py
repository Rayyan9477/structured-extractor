"""
Monkey OCR Integration Module

Integrates Monkey OCR for handling complex document layouts and mixed text types.
"""

import asyncio
import time
import json
import requests
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import io
import base64

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class MonkeyOCRProcessor:
    """
    Integration with Monkey OCR for document text extraction.
    Monkey OCR is particularly effective for complex document layouts.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Monkey OCR processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.ocr_config = self.config.get("ocr", {}).get("monkey_ocr", {})
        self.api_key = self.ocr_config.get("api_key", None)
        self.endpoint_url = self.ocr_config.get("endpoint_url", "http://localhost:8000/process")
        self.timeout = self.ocr_config.get("timeout", 30)
        self.models_loaded = True  # Monkey OCR is API-based, so no loading needed
        
        self.logger.info("Monkey OCR processor initialized")
    
    async def load_models(self) -> None:
        """
        Verify API connectivity.
        No actual model loading is needed since Monkey OCR is API-based.
        """
        # Just a placeholder for API verification if needed
        self.models_loaded = True
        self.logger.info("Monkey OCR ready for processing")
        
    def _encode_image(self, image: Image.Image) -> str:
        """
        Encode a PIL Image to base64 string.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Base64 encoded string representation of the image
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using Monkey OCR.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            OCR result containing extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Convert image to base64
            img_base64 = self._encode_image(image)
            
            # Prepare request payload
            payload = {
                "image": img_base64,
                "extract_tables": True,
                "extract_text_blocks": True,
                "detect_orientation": True
            }
            
            # Add API key if provided
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            # Make API request
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse response
            result = response.json()
            
            # Extract text from response
            if "full_text" in result:
                extracted_text = result["full_text"]
            else:
                # Combine text blocks if full_text not provided
                text_blocks = result.get("text_blocks", [])
                extracted_text = "\n".join([block.get("text", "") for block in text_blocks])
            
            # Extract confidence
            confidence = result.get("confidence", 0.8)
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Monkey OCR extracted {len(extracted_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                model_name="monkey_ocr",
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Monkey OCR text extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="monkey_ocr",
                processing_time=time.time() - start_time
            )
    
    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process
            
        Returns:
            A list of OCRResult objects
        """
        self.logger.info(f"Starting Monkey OCR batch processing for {len(images)} images")
        
        # Process images in parallel
        tasks = [self.extract_text(image) for image in images]
        results = await asyncio.gather(*tasks)
        
        self.logger.info(f"Completed Monkey OCR batch processing for {len(images)} images")
        return results 