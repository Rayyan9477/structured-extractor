"""
Local Monkey OCR Integration Module

Integrates local Monkey OCR model for handling complex document layouts and mixed text types.
Uses the locally cached echo840/MonkeyOCR model for offline processing.
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


class MonkeyOCRLocalProcessor:
    """
    Local integration with Monkey OCR for document text extraction.
    Uses the cached echo840/MonkeyOCR model for complex document layouts.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the local Monkey OCR processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.ocr_config = self.config.get("ocr", {}).get("monkey_ocr", {})
        self.model_name = "echo840/MonkeyOCR"
        self.local_model_path = self.config.get_model_path(self.model_name)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.models_loaded = False
        
        # Generation parameters
        self.max_length = self.ocr_config.get("max_length", 2048)
        self.temperature = self.ocr_config.get("temperature", 0.1)
        self.top_p = self.ocr_config.get("top_p", 0.9)
        
        self.logger.info(f"Monkey OCR Local processor initialized with device: {self.device}")
    
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
        Load the local Monkey OCR models.
        
        MonkeyOCR uses a Structure-Recognition-Relation (SRR) paradigm with separate models.
        """
        if self.models_loaded:
            return
            
        try:
            self.logger.info(f"Loading local Monkey OCR model from: {self.local_model_path}")
            
            # Check if local model exists
            if not self.local_model_path.exists():
                self.logger.error(f"Local model not found at {self.local_model_path}")
                raise FileNotFoundError(f"Monkey OCR model not found at {self.local_model_path}")
            
            # Check for Recognition subfolder
            recognition_path = self.local_model_path / "Recognition"
            if not recognition_path.exists():
                self.logger.error(f"Recognition folder not found at {recognition_path}")
                raise FileNotFoundError(f"MonkeyOCR Recognition model not found")
            
            # Load tokenizer and processor from Recognition folder
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(recognition_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load processor with error handling
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(recognition_path),
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception as proc_error:
                self.logger.warning(f"Failed to load processor: {proc_error}, continuing without processor")
                self.processor = None
            
            # Load the recognition model
            try:
                # Try loading as Vision2Seq model first (MonkeyOCR is vision-language model)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    str(recognition_path),
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except Exception as e:
                self.logger.warning(f"Failed to load as Vision2Seq model: {e}")
                # Fallback to CausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(recognition_path),
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                )
            
            # Move model to device if needed
            if self.model and self.device != "cuda":  # device_map="auto" handles CUDA
                self.model = self.model.to(self.device)
            
            self.models_loaded = True
            self.logger.info(f"Monkey OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Monkey OCR model: {e}", exc_info=True)
            # Set models_loaded to False for fallback
            self.models_loaded = False
            raise RuntimeError(f"Could not load local Monkey OCR model: {e}")
        
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
        
        # Resize if too large (Monkey OCR works better with reasonable sizes)
        max_size = 1280
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using local Monkey OCR.
        
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
                # Use processor-based approach
                extracted_text, confidence = await self._extract_with_processor(processed_image)
            elif self.model and self.tokenizer:
                # Use model + tokenizer approach
                extracted_text, confidence = await self._extract_with_model(processed_image)
            else:
                raise RuntimeError("No valid model or processor loaded")
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"Monkey OCR extracted {len(extracted_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=extracted_text,
                confidence=confidence,
                model_name="monkey_ocr_local",
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Monkey OCR text extraction failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="monkey_ocr_local",
                processing_time=processing_time
            )
    
    async def _extract_with_processor(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text using processor-based approach.
        
        Args:
            image: Preprocessed PIL Image
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        try:
            # Check if processor is available
            if self.processor is None:
                self.logger.warning("Processor is None, cannot use processor-based extraction")
                return "", 0.0
            
            # Process image with processor
            inputs = self.processor(image, return_tensors="pt")
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
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0
                )
            
            # Decode output
            if self.tokenizer:
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                decoded = str(outputs[0])
            
            # Calculate confidence (simple heuristic based on output length)
            confidence = min(0.95, max(0.5, len(decoded) / 1000))
            
            return decoded.strip(), confidence
            
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
            
            # Create OCR prompt
            prompt = f"Extract all text from this image: <image>{img_str}</image>"
            
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
            generated_text = full_text[len(prompt):].strip()
            
            # Calculate confidence
            confidence = min(0.90, max(0.4, len(generated_text) / 500))
            
            return generated_text, confidence
            
        except Exception as e:
            self.logger.error(f"Model-based extraction failed: {e}")
            return "", 0.0
    
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
        
        self.logger.info(f"Starting Monkey OCR local batch processing for {len(images)} images")
        
        # Process images sequentially to avoid memory issues
        results = []
        for idx, image in enumerate(images):
            self.logger.debug(f"Processing image {idx+1}/{len(images)}")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"Completed Monkey OCR local batch processing for {len(images)} images")
        return results
