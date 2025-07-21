"""
Unified OCR Engine for Medical Superbill Extraction

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
from pathlib import Path

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence
from src.processors.monkey_ocr_engine import MonkeyOCREngine
from src.processors.nanonets_ocr_engine import NanonetsOCREngine
from src.processors.trocr_engine import TrOCREngine


class UnifiedOCREngine:
    """
    Unified OCR engine that can use multiple models independently 
    and provide the best results for medical document processing.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the unified OCR engine.
        
        Args:
            config: Configuration manager.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get OCR configuration
        self.ocr_config = self.config.get("ocr", {})
        self.use_models = self.ocr_config.get("ensemble.use_models", ["nanonets_ocr", "monkey_ocr"])
        self.weights = self.ocr_config.get("ensemble.weights", {})
        self.method = self.ocr_config.get("ensemble.method", "best_confidence")
        
        self.device = self._get_device()
        self.logger.info(f"Unified OCR engine initialized with device: {self.device}")
        
        # Initialize available engines
        self.engines = {}
        self._initialize_engines()
        
    def _get_device(self) -> str:
        """Get the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize_engines(self) -> None:
        """Initialize available OCR engines."""
        if "monkey_ocr" in self.use_models:
            try:
                self.engines["monkey_ocr"] = MonkeyOCREngine(self.config)
                self.logger.info("MonkeyOCR engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MonkeyOCR: {e}")
        
        if "nanonets_ocr" in self.use_models:
            try:
                self.engines["nanonets_ocr"] = NanonetsOCREngine(self.config)
                self.logger.info("Nanonets OCR engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Nanonets OCR: {e}")
        
        if "trocr" in self.use_models:
            try:
                self.engines["trocr"] = TrOCREngine(self.config)
                self.logger.info("TrOCR engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TrOCR: {e}")
        
        if not self.engines:
            raise RuntimeError("No OCR engines could be initialized")

    async def load_models(self) -> None:
        """Load all configured OCR models."""
        self.logger.info("Loading OCR models...")
        
        load_tasks = []
        for engine_name, engine in self.engines.items():
            load_tasks.append(self._load_engine_safely(engine_name, engine))
        
        await asyncio.gather(*load_tasks)
        
        loaded_engines = [name for name, engine in self.engines.items() if hasattr(engine, 'models_loaded') and engine.models_loaded]
        self.logger.info(f"Loaded OCR engines: {loaded_engines}")

    async def _load_engine_safely(self, engine_name: str, engine) -> None:
        """Load an engine safely with error handling."""
        try:
            await engine.load_models()
            self.logger.info(f"{engine_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load {engine_name}: {e}")

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using the best available model.
        
        Args:
            image: A PIL Image to process.
            
        Returns:
            An OCRResult object with the extracted text and metadata.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        start_time = time.time()
        
        # Run all available engines
        results = []
        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'models_loaded') and engine.models_loaded:
                    result = await engine.extract_text(image)
                    result.model_name = f"{result.model_name}_{engine_name}"
                    results.append(result)
                    self.logger.debug(f"{engine_name} completed extraction")
            except Exception as e:
                self.logger.warning(f"{engine_name} extraction failed: {e}")
        
        if not results:
            self.logger.error("All OCR engines failed")
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="unified_ocr", 
                processing_time=time.time() - start_time
            )
        
        # Select best result based on method
        best_result = self._select_best_result(results)
        best_result.processing_time = time.time() - start_time
        
        self.logger.debug(f"Best result from {best_result.model_name} with confidence {best_result.confidence:.2f}")
        return best_result

    def _select_best_result(self, results: List[OCRResult]) -> OCRResult:
        """Select the best result from multiple OCR engines."""
        if not results:
            return OCRResult(text="", confidence=0.0, model_name="none")
        
        if len(results) == 1:
            return results[0]
        
        if self.method == "best_confidence":
            # Return result with highest confidence
            return max(results, key=lambda r: r.confidence)
        
        elif self.method == "longest_text":
            # Return result with most extracted text
            return max(results, key=lambda r: len(r.text))
        
        elif self.method == "weighted_average":
            # Combine results using weights
            return self._weighted_combination(results)
        
        else:
            # Default: best confidence
            return max(results, key=lambda r: r.confidence)

    def _weighted_combination(self, results: List[OCRResult]) -> OCRResult:
        """Combine results using weighted average."""
        if not results:
            return OCRResult(text="", confidence=0.0, model_name="weighted")
        
        # For simplicity, just return the highest confidence result
        # In a more sophisticated implementation, you could combine texts
        best_result = max(results, key=lambda r: r.confidence)
        best_result.model_name = "weighted_combination"
        
        return best_result

    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process.
            
        Returns:
            A list of OCRResult objects.
        """
        self.logger.info(f"Starting unified OCR batch processing for {len(images)} images")
        
        results = []
        for i, image in enumerate(images):
            self.logger.debug(f"Processing image {i+1}/{len(images)}")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"Unified OCR batch processing completed for {len(images)} images")
        return results

    def get_available_engines(self) -> List[str]:
        """Get list of available and loaded engines."""
        return [
            name for name, engine in self.engines.items() 
            if hasattr(engine, 'models_loaded') and engine.models_loaded
        ]

    def get_engine_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of each engine."""
        capabilities = {}
        
        for name, engine in self.engines.items():
            if hasattr(engine, 'get_feature_capabilities'):
                capabilities[name] = engine.get_feature_capabilities()
            else:
                capabilities[name] = {"basic_ocr": True}
        
        return capabilities


# Maintain backwards compatibility
class OCREngine(UnifiedOCREngine):
    """Backwards compatible OCR engine class."""
    pass
