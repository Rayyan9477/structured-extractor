"""
OCR Integration Module for Medical Superbill Extraction

Integrates multiple OCR models including MonkeyOCR for handwriting recognition
and Nanonets-OCR for printed text, with confidence-based result merging.
"""

import asyncio
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import re

from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor,
    pipeline, Pipeline
)
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import FieldConfidence


class OCRResult:
    """Represents OCR extraction result from a single model."""
    
    def __init__(
        self,
        text: str,
        confidence: float,
        model_name: str,
        bounding_boxes: Optional[List[Dict]] = None,
        processing_time: Optional[float] = None
    ):
        self.text = text
        self.confidence = confidence
        self.model_name = model_name
        self.bounding_boxes = bounding_boxes or []
        self.processing_time = processing_time


class MonkeyOCREngine:
    """OCR engine using MonkeyOCR for handwriting recognition."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize MonkeyOCR engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get model configuration
        self.model_config = config.get_model_config("ocr", "monkey_ocr")
        self.model_name = self.model_config.get("model_name", "echo840/MonkeyOCR")
        self.max_length = self.model_config.get("max_length", 512)
        self.confidence_threshold = self.model_config.get("confidence_threshold", 0.7)
        
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        self.logger.info(f"Initializing MonkeyOCR with model: {self.model_name}")
    
    def _get_device(self) -> str:
        """Get optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_model(self) -> None:
        """Load the MonkeyOCR model and tokenizer."""
        try:
            self.logger.info("Loading MonkeyOCR model...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model.eval()
            self.logger.info("MonkeyOCR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load MonkeyOCR model: {e}")
            raise
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from image using MonkeyOCR.
        
        Args:
            image: PIL Image to process
            
        Returns:
            OCR result with extracted text and confidence
        """
        if self.model is None:
            await self.load_model()
        
        try:
            import time
            start_time = time.time()
            
            # Prepare image for model
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert image to tensor
            image_array = np.array(image)
            
            with torch.no_grad():
                # Process image through model
                # Note: This is a placeholder implementation
                # The actual MonkeyOCR API may differ
                try:
                    # Attempt to use the model's built-in OCR function
                    if hasattr(self.model, 'generate_text'):
                        result = self.model.generate_text(image)
                        text = result.get('text', '')
                        confidence = result.get('confidence', 0.8)
                    else:
                        # Fallback implementation
                        text = self._extract_with_transformers(image)
                        confidence = 0.8  # Default confidence
                        
                except Exception as model_error:
                    self.logger.warning(f"MonkeyOCR model error: {model_error}")
                    # Use a simpler OCR approach as fallback
                    text = self._fallback_ocr(image)
                    confidence = 0.6
            
            processing_time = time.time() - start_time
            
            # Clean extracted text
            cleaned_text = self._clean_text(text)
            
            result = OCRResult(
                text=cleaned_text,
                confidence=confidence,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
            self.logger.debug(f"MonkeyOCR extracted {len(cleaned_text)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"MonkeyOCR extraction failed: {e}")
            # Return empty result with low confidence
            return OCRResult(
                text="",
                confidence=0.0,
                model_name=self.model_name,
                processing_time=0.0
            )
    
    def _extract_with_transformers(self, image: Image.Image) -> str:
        """Extract text using transformers pipeline."""
        try:
            # Create OCR pipeline
            ocr_pipeline = pipeline(
                "image-to-text",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            result = ocr_pipeline(image)
            return result[0]['generated_text'] if result else ""
            
        except Exception as e:
            self.logger.warning(f"Transformers pipeline failed: {e}")
            return ""
    
    def _fallback_ocr(self, image: Image.Image) -> str:
        """Fallback OCR using pytesseract."""
        try:
            import pytesseract
            return pytesseract.image_to_string(image)
        except ImportError:
            self.logger.warning("pytesseract not available for fallback")
            return ""
        except Exception as e:
            self.logger.warning(f"Fallback OCR failed: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class NanonetsOCREngine:
    """OCR engine using Nanonets-OCR for printed text recognition."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Nanonets OCR engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get model configuration
        self.model_config = config.get_model_config("ocr", "nanonets_ocr")
        self.model_name = self.model_config.get("model_name", "nanonets/Nanonets-OCR-s")
        self.max_length = self.model_config.get("max_length", 1024)
        self.confidence_threshold = self.model_config.get("confidence_threshold", 0.8)
        
        self.processor = None
        self.model = None
        self.device = self._get_device()
        
        self.logger.info(f"Initializing Nanonets OCR with model: {self.model_name}")
    
    def _get_device(self) -> str:
        """Get optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def load_model(self) -> None:
        """Load the Nanonets OCR model and processor."""
        try:
            self.logger.info("Loading Nanonets OCR model...")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model.eval()
            self.logger.info("Nanonets OCR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Nanonets OCR model: {e}")
            raise
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from image using Nanonets OCR.
        
        Args:
            image: PIL Image to process
            
        Returns:
            OCR result with extracted text and confidence
        """
        if self.model is None:
            await self.load_model()
        
        try:
            import time
            start_time = time.time()
            
            # Prepare image for model
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            with torch.no_grad():
                # Process image
                try:
                    # Use processor to prepare inputs
                    inputs = self.processor(image, return_tensors="pt").to(self.device)
                    
                    # Generate text
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decode output
                    text = self.processor.decode(outputs[0], skip_special_tokens=True)
                    confidence = 0.85  # Default high confidence for printed text
                    
                except Exception as model_error:
                    self.logger.warning(f"Nanonets OCR model error: {model_error}")
                    # Use fallback OCR
                    text = self._fallback_ocr(image)
                    confidence = 0.7
            
            processing_time = time.time() - start_time
            
            # Clean extracted text
            cleaned_text = self._clean_text(text)
            
            result = OCRResult(
                text=cleaned_text,
                confidence=confidence,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
            self.logger.debug(f"Nanonets OCR extracted {len(cleaned_text)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"Nanonets OCR extraction failed: {e}")
            # Return empty result with low confidence
            return OCRResult(
                text="",
                confidence=0.0,
                model_name=self.model_name,
                processing_time=0.0
            )
    
    def _fallback_ocr(self, image: Image.Image) -> str:
        """Fallback OCR using pytesseract."""
        try:
            import pytesseract
            return pytesseract.image_to_string(image, config='--psm 6')
        except ImportError:
            self.logger.warning("pytesseract not available for fallback")
            return ""
        except Exception as e:
            self.logger.warning(f"Fallback OCR failed: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class OCRResultMerger:
    """Merges results from multiple OCR engines with confidence-based scoring."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize OCR result merger.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
    
    def merge_results(self, results: List[OCRResult]) -> OCRResult:
        """
        Merge multiple OCR results into a single best result.
        
        Args:
            results: List of OCR results from different engines
            
        Returns:
            Merged OCR result
        """
        if not results:
            return OCRResult("", 0.0, "merged")
        
        if len(results) == 1:
            return results[0]
        
        self.logger.debug(f"Merging {len(results)} OCR results")
        
        # Filter results by confidence threshold
        valid_results = [r for r in results if r.confidence > 0.5]
        
        if not valid_results:
            # Return the best of the invalid results
            best_result = max(results, key=lambda r: r.confidence)
            return best_result
        
        # Strategy 1: Use highest confidence result if significantly better
        best_result = max(valid_results, key=lambda r: r.confidence)
        second_best = max([r for r in valid_results if r != best_result], 
                         key=lambda r: r.confidence, default=None)
        
        if second_best is None or best_result.confidence - second_best.confidence > 0.2:
            return best_result
        
        # Strategy 2: Merge similar results
        merged_text = self._merge_text_content(valid_results)
        merged_confidence = self._calculate_merged_confidence(valid_results)
        
        return OCRResult(
            text=merged_text,
            confidence=merged_confidence,
            model_name="merged",
            processing_time=sum(r.processing_time or 0 for r in valid_results)
        )
    
    def _merge_text_content(self, results: List[OCRResult]) -> str:
        """Merge text content from multiple results."""
        if not results:
            return ""
        
        # Get all text results
        texts = [r.text for r in results if r.text.strip()]
        
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Find the longest common subsequence approach
        # For simplicity, use the result with highest confidence and longest text
        best_text = max(texts, key=lambda t: (
            next(r.confidence for r in results if r.text == t),
            len(t)
        ))
        
        return best_text
    
    def _calculate_merged_confidence(self, results: List[OCRResult]) -> float:
        """Calculate merged confidence score."""
        if not results:
            return 0.0
        
        # Weighted average based on text length and original confidence
        total_weight = 0
        weighted_confidence = 0
        
        for result in results:
            weight = len(result.text) * result.confidence
            total_weight += weight
            weighted_confidence += weight * result.confidence
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0


class OCREngine:
    """Main OCR engine that coordinates multiple OCR models."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize OCR engines
        self.monkey_ocr = MonkeyOCREngine(config)
        self.nanonets_ocr = NanonetsOCREngine(config)
        self.merger = OCRResultMerger(config)
        
        self.models_loaded = False
    
    async def load_models(self) -> None:
        """Load all OCR models."""
        if self.models_loaded:
            return
        
        self.logger.info("Loading OCR models...")
        
        try:
            # Load models in parallel
            await asyncio.gather(
                self.monkey_ocr.load_model(),
                self.nanonets_ocr.load_model()
            )
            
            self.models_loaded = True
            self.logger.info("All OCR models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load OCR models: {e}")
            raise
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from image using all available OCR engines.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Merged OCR result
        """
        if not self.models_loaded:
            await self.load_models()
        
        self.logger.debug("Starting OCR extraction")
        
        # Run OCR engines in parallel
        tasks = [
            self.monkey_ocr.extract_text(image),
            self.nanonets_ocr.extract_text(image)
        ]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"OCR engine {i} failed: {result}")
                else:
                    valid_results.append(result)
            
            if not valid_results:
                self.logger.error("All OCR engines failed")
                return OCRResult("", 0.0, "failed")
            
            # Merge results
            merged_result = self.merger.merge_results(valid_results)
            
            self.logger.debug(f"OCR extraction completed, confidence: {merged_result.confidence:.2f}")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return OCRResult("", 0.0, "error")
    
    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from multiple images in batch.
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of OCR results
        """
        if not self.models_loaded:
            await self.load_models()
        
        self.logger.info(f"Starting batch OCR extraction for {len(images)} images")
        
        # Process images in parallel with concurrency limit
        semaphore = asyncio.Semaphore(4)  # Limit concurrent OCR operations
        
        async def process_single_image(image):
            async with semaphore:
                return await self.extract_text(image)
        
        results = await asyncio.gather(
            *[process_single_image(img) for img in images],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to process image {i}: {result}")
                processed_results.append(OCRResult("", 0.0, "failed"))
            else:
                processed_results.append(result)
        
        self.logger.info(f"Batch OCR extraction completed: {len(processed_results)} results")
        return processed_results
