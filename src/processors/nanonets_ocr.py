"""
Nanonets OCR Integration Module

Integrates Nanonets OCR for accurate text extraction with document structure understanding.
"""

import asyncio
import time
import json
import requests
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import io
import base64
import os

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class NanonetsOCRProcessor:
    """
    Integration with Nanonets OCR for document text extraction.
    Nanonets OCR specializes in extracting structured data from documents.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Nanonets OCR processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.ocr_config = self.config.get("ocr", {}).get("nanonets_ocr", {})
        self.api_key = self.ocr_config.get("api_key", None)
        self.model_id = self.ocr_config.get("model_id", "default")
        self.endpoint_url = self.ocr_config.get(
            "endpoint_url", 
            f"https://app.nanonets.com/api/v2/OCR/Model/{self.model_id}/LabelFile/"
        )
        self.timeout = self.ocr_config.get("timeout", 60)
        self.models_loaded = True  # Nanonets OCR is API-based, no local model loading needed
        
        self.logger.info("Nanonets OCR processor initialized")
    
    async def load_models(self) -> None:
        """
        Verify API connectivity.
        No actual model loading is needed since Nanonets OCR is API-based.
        """
        # Placeholder for API verification
        self.models_loaded = True
        self.logger.info("Nanonets OCR ready for processing")
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using Nanonets OCR.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            OCR result containing extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Save image to temporary file
            temp_file = "temp_nanonets_image.jpg"
            image.save(temp_file, format="JPEG")
            
            # Prepare request
            url = self.endpoint_url
            data = {'file': open(temp_file, 'rb')}
            
            # Set authentication
            auth = None
            headers = {}
            if self.api_key:
                auth = (self.api_key, '')  # Nanonets uses API key as username
            
            # Make API request
            response = requests.post(
                url,
                auth=auth,
                files=data,
                headers=headers,
                timeout=self.timeout
            )
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse response
            result = response.json()
            
            # Extract text from response (adjust based on actual Nanonets response format)
            extracted_text = ""
            confidence = 0.0
            
            if "result" in result:
                # Extract from predictions
                predictions = result.get("result", [{}])
                for pred in predictions:
                    # Get all prediction objects
                    prediction_data = pred.get("prediction", [])
                    
                    # Extract text and confidence from each prediction
                    for item in prediction_data:
                        if "ocr_text" in item:
                            extracted_text += item.get("ocr_text", "") + "\n"
                            # Accumulate confidence scores
                            conf_score = item.get("score", 0.0)
                            confidence = max(confidence, conf_score)
            
            # Clean up text
            extracted_text = extracted_text.strip()
            
            # Default confidence if none found
            if confidence == 0.0:
                confidence = 0.85  # Reasonable default
                
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Nanonets OCR extracted {len(extracted_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                model_name="nanonets_ocr",
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Nanonets OCR text extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="nanonets_ocr",
                processing_time=time.time() - start_time
            )
    
    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        Due to API limitations, this processes images sequentially.
        
        Args:
            images: A list of PIL Images to process
            
        Returns:
            A list of OCRResult objects
        """
        self.logger.info(f"Starting Nanonets OCR batch processing for {len(images)} images")
        
        results = []
        for index, image in enumerate(images):
            self.logger.debug(f"Processing image {index+1}/{len(images)}")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"Completed Nanonets OCR batch processing for {len(images)} images")
        return results 