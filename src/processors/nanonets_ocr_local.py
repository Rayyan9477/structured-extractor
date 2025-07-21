"""
Local Nanonets OCR Integration Module

Integrates local Nanonets OCR model for accurate text extraction with document structure understanding.
Uses the locally cached nanonets/Nanonets-OCR-s model for offline processing.
"""

import asyncio
import time
import json
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
import io
import base64
from pathlib import Path

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForVision2Seq,
    AutoProcessor,
    GenerationConfig,
    pipeline
)

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class NanonetsOCRLocalProcessor:
    """
    Local integration with Nanonets OCR for document text extraction.
    Uses the cached nanonets/Nanonets-OCR-s model for structured data extraction.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the local Nanonets OCR processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.ocr_config = self.config.get("ocr", {}).get("nanonets_ocr", {})
        self.model_name = "nanonets/Nanonets-OCR-s"
        self.local_model_path = self.config.get_model_path(self.model_name)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.models_loaded = False
        
        # Generation parameters
        self.max_length = self.ocr_config.get("max_length", 4096)
        self.temperature = self.ocr_config.get("temperature", 0.1)
        self.top_p = self.ocr_config.get("top_p", 0.9)
        
        self.logger.info(f"Nanonets OCR Local processor initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Get optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_models(self) -> None:
        """
        Load the local Nanonets OCR models.
        """
        if self.models_loaded:
            return
            
        try:
            self.logger.info(f"Loading local Nanonets OCR model from: {self.local_model_path}")
            
            # Check if local model exists
            if not self.local_model_path.exists():
                self.logger.error(f"Local model not found at {self.local_model_path}")
                raise FileNotFoundError(f"Nanonets OCR model not found at {self.local_model_path}")
            
            # Load tokenizer from local cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model from local cache
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.local_model_path),
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception as e:
                self.logger.warning(f"Failed to load with AutoModelForCausalLM: {e}")
                # Try with processor approach
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        str(self.local_model_path),
                        trust_remote_code=True,
                        local_files_only=True
                    )
                except Exception as e2:
                    self.logger.error(f"Failed to load processor: {e2}")
                    raise
            
            # Move model to device if needed
            if self.model and self.device != "cuda":  # device_map="auto" handles CUDA
                self.model = self.model.to(self.device)
            
            self.models_loaded = True
            self.logger.info(f"Nanonets OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Nanonets OCR model: {e}", exc_info=True)
            # Fallback to API-based processing if available
            self.models_loaded = False
            raise RuntimeError(f"Could not load local Nanonets OCR model: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal OCR results.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (Nanonets works well with high resolution)
        max_size = 1600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _create_ocr_prompt(self, image_context: str = "") -> str:
        """
        Create OCR prompt for the Nanonets model.
        
        Args:
            image_context: Optional context about the image
            
        Returns:
            Formatted OCR prompt
        """
        base_prompt = """<|im_start|>system
You are an expert OCR assistant. Extract all text content from the provided image with high accuracy. 
Maintain the original structure and formatting where possible.
Focus on:
1. All visible text including headers, body text, tables, and captions
2. Proper spacing and line breaks
3. Numerical data and special characters
4. Any structured data or forms

Return only the extracted text without any additional commentary.<|im_end|>
<|im_start|>user
Please extract all text from this image:
"""
        return base_prompt
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using local Nanonets OCR.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            OCR result containing extracted text and metadata
        """
        if not self.models_loaded:
            await self.load_models()
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text using the model
            if self.processor:
                # Use processor-based approach (vision model)
                extracted_text, confidence = await self._extract_with_processor(processed_image)
            elif self.model and self.tokenizer:
                # Use model + tokenizer approach (text model with image encoding)
                extracted_text, confidence = await self._extract_with_model(processed_image)
            else:
                raise RuntimeError("No valid model or processor loaded")
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Nanonets OCR extracted {len(extracted_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                model_name="nanonets_ocr_local",
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Nanonets OCR text extraction failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="nanonets_ocr_local",
                processing_time=processing_time
            )
    
    async def _extract_with_processor(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text using processor-based approach (for vision models).
        
        Args:
            image: Preprocessed PIL Image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Check if model and processor are available
            if self.model is None:
                self.logger.warning("Model is None, cannot use processor-based extraction")
                return "", 0.0
            
            if self.processor is None:
                self.logger.warning("Processor is None, cannot use processor-based extraction")
                return "", 0.0
                
            # Create OCR prompt
            prompt = self._create_ocr_prompt()
            
            # Process image and text with processor
            inputs = self.processor(
                text=prompt,
                images=image, 
                return_tensors="pt"
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            tokenizer_for_decode = self.tokenizer if self.tokenizer else self.processor.tokenizer
            decoded = tokenizer_for_decode.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if "<|im_start|>assistant" in decoded:
                parts = decoded.split("<|im_start|>assistant")
                if len(parts) > 1:
                    extracted_text = parts[-1].replace("<|im_end|>", "").strip()
                else:
                    extracted_text = decoded.strip()
            else:
                # Fallback: extract everything after the prompt
                extracted_text = decoded[len(self._create_ocr_prompt()):].strip()
            
            # Calculate confidence based on output quality
            confidence = self._calculate_confidence(extracted_text)
            
            return extracted_text, confidence
            
        except Exception as e:
            self.logger.error(f"Processor-based extraction failed: {e}")
            return "", 0.0
    
    async def _extract_with_model(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text using model + tokenizer approach.
        
        Args:
            image: Preprocessed PIL Image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Convert image to base64 for text-based models
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # Create OCR prompt with image data
            prompt = f"""{self._create_ocr_prompt()}
<image_data>{img_str}</image_data>
<|im_end|>
<|im_start|>assistant
"""
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length // 2  # Leave room for generation
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and extract generated part
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if "<|im_start|>assistant" in full_text:
                parts = full_text.split("<|im_start|>assistant")
                extracted_text = parts[-1].replace("<|im_end|>", "").strip()
            else:
                # Fallback
                extracted_text = full_text[len(prompt):].strip()
            
            # Calculate confidence
            confidence = self._calculate_confidence(extracted_text)
            
            return extracted_text, confidence
            
        except Exception as e:
            self.logger.error(f"Model-based extraction failed: {e}")
            return "", 0.0
    
    def _calculate_confidence(self, extracted_text: str) -> float:
        """
        Calculate confidence score based on extracted text quality.
        
        Args:
            extracted_text: The extracted text
            
        Returns:
            Confidence score between 0 and 1
        """
        if not extracted_text:
            return 0.0
        
        # Base confidence
        confidence = 0.6
        
        # Boost confidence based on text length (more text usually means better extraction)
        length_bonus = min(0.3, len(extracted_text) / 1000)
        confidence += length_bonus
        
        # Boost confidence if text contains structured elements
        structured_indicators = ['\n', '\t', ':', '-', '•', '1.', '2.', 'TABLE', 'FORM']
        structure_count = sum(1 for indicator in structured_indicators if indicator in extracted_text)
        structure_bonus = min(0.1, structure_count * 0.02)
        confidence += structure_bonus
        
        return min(0.95, confidence)
    
    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process
            
        Returns:
            A list of OCRResult objects
        """
        if not self.models_loaded:
            await self.load_models()
        
        self.logger.info(f"Starting Nanonets OCR local batch processing for {len(images)} images")
        
        # Process images sequentially to avoid memory issues with large models
        results = []
        for idx, image in enumerate(images):
            self.logger.debug(f"Processing image {idx+1}/{len(images)}")
            result = await self.extract_text(image)
            results.append(result)
            
            # Clear CUDA cache after each image to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info(f"Completed Nanonets OCR local batch processing for {len(images)} images")
        return results
