"""
OCR Ensemble Engine

Combines multiple OCR engines to provide better accuracy through model ensembling.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from difflib import SequenceMatcher

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult
from src.processors.ocr_engine import OCREngine
from src.processors.monkey_ocr import MonkeyOCRProcessor
from src.processors.nanonets_ocr import NanonetsOCRProcessor
from src.processors.monkey_ocr_local import MonkeyOCRLocalProcessor
from src.processors.nanonets_ocr_local import NanonetsOCRLocalProcessor
from src.processors.mock_ocr import MockOCRProcessor


class OCREnsembleEngine:
    """
    Ensemble engine that combines results from multiple OCR models.
    Uses weighted voting to combine results based on model confidence.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the OCR ensemble engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.ensemble_config = self.config.get("ocr", {}).get("ensemble", {})
        self.weights = self.ensemble_config.get("weights", {
            "trocr": 1.0,
            "monkey_ocr": 1.0,
            "nanonets_ocr": 1.0,
            "mock_ocr": 1.0
        })
        self.use_models = self.ensemble_config.get("use_models", ["mock_ocr"])  # Default to mock OCR only
        self.minimum_models = self.ensemble_config.get("minimum_models", 1)
        
        # Initialize OCR engines (prefer local implementations)
        self.trocr_engine = OCREngine(config)
        
        # Use unified engines for better performance and offline capability
        try:
            # Try to use the fixed unified engines first
            from src.processors.monkey_ocr_engine import MonkeyOCREngine
            from src.processors.nanonets_ocr_engine import NanonetsOCREngine
            
            self.monkey_ocr_engine = MonkeyOCREngine(config)
            self.nanonets_ocr_engine = NanonetsOCREngine(config)
            self.use_unified_engines = True
            self.use_local = False  # Mark as not using local processors
            self.logger.info("Using unified OCR engines")
            
        except Exception as unified_error:
            self.logger.warning(f"Unified engines not available: {unified_error}, trying local processors")
            try:
                self.monkey_ocr_local = MonkeyOCRLocalProcessor(config)
                self.nanonets_ocr_local = NanonetsOCRLocalProcessor(config)
                self.use_local = True
                self.use_unified_engines = False
            except Exception as e:
                self.logger.warning(f"Local processors not available: {e}, falling back to API")
                self.monkey_ocr = MonkeyOCRProcessor(config)
                self.nanonets_ocr = NanonetsOCRProcessor(config)
                self.use_local = False
                self.use_unified_engines = False
        
        self.mock_ocr = MockOCRProcessor(config)
        
        # Track if models have been loaded
        self.models_loaded = False
        
        self.logger.info(f"OCR Ensemble initialized with models: {self.use_models}")
        
    async def load_models(self) -> None:
        """Load all OCR models."""
        if self.models_loaded:
            return
        
        self.logger.info("Loading all OCR models...")
        
        load_tasks = []
        
        if "trocr" in self.use_models:
            load_tasks.append(self.trocr_engine.load_models())
            
        if "monkey_ocr" in self.use_models:
            if hasattr(self, 'monkey_ocr_engine') and self.use_unified_engines:
                load_tasks.append(self.monkey_ocr_engine.load_models())
            elif self.use_local and hasattr(self, 'monkey_ocr_local'):
                load_tasks.append(self.monkey_ocr_local.load_models())
            elif hasattr(self, 'monkey_ocr'):
                load_tasks.append(self.monkey_ocr.load_models())
            else:
                self.logger.warning("Monkey OCR requested but not available")
            
        if "nanonets_ocr" in self.use_models:
            if hasattr(self, 'nanonets_ocr_engine') and self.use_unified_engines:
                load_tasks.append(self.nanonets_ocr_engine.load_models())
            elif self.use_local and hasattr(self, 'nanonets_ocr_local'):
                load_tasks.append(self.nanonets_ocr_local.load_models())
            elif hasattr(self, 'nanonets_ocr'):
                load_tasks.append(self.nanonets_ocr.load_models())
            else:
                self.logger.warning("Nanonets OCR requested but not available")
            
        if "mock_ocr" in self.use_models:
            load_tasks.append(self.mock_ocr.load_models())
        
        if load_tasks:
            await asyncio.gather(*load_tasks)
            
        self.models_loaded = True
        self.logger.info("All OCR models loaded successfully")

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using all available OCR models and ensemble the results.
        
        Args:
            image: A PIL Image to process
            
        Returns:
            An OCRResult object with the ensembled extraction results
        """
        if not self.models_loaded:
            await self.load_models()
        
        start_time = time.time()
        
        # Collect results from each model
        results = []
        
        if "trocr" in self.use_models:
            trocr_result = await self.trocr_engine.extract_text(image)
            if trocr_result.text:
                results.append(trocr_result)
        
        if "monkey_ocr" in self.use_models:
            if hasattr(self, 'monkey_ocr_engine') and self.use_unified_engines:
                monkey_result = await self.monkey_ocr_engine.extract_text(image)
            elif self.use_local and hasattr(self, 'monkey_ocr_local'):
                monkey_result = await self.monkey_ocr_local.extract_text(image)
            elif hasattr(self, 'monkey_ocr'):
                monkey_result = await self.monkey_ocr.extract_text(image)
            else:
                monkey_result = None
            
            if monkey_result and monkey_result.text:
                results.append(monkey_result)
        
        if "nanonets_ocr" in self.use_models:
            if hasattr(self, 'nanonets_ocr_engine') and self.use_unified_engines:
                nanonets_result = await self.nanonets_ocr_engine.extract_text(image)
            elif self.use_local and hasattr(self, 'nanonets_ocr_local'):
                nanonets_result = await self.nanonets_ocr_local.extract_text(image)
            elif hasattr(self, 'nanonets_ocr'):
                nanonets_result = await self.nanonets_ocr.extract_text(image)
            else:
                nanonets_result = None
                
            if nanonets_result and nanonets_result.text:
                results.append(nanonets_result)
                
        if "mock_ocr" in self.use_models:
            mock_result = await self.mock_ocr.extract_text(image)
            if mock_result.text:
                results.append(mock_result)
        
        # Ensure we have enough valid results
        if len(results) < self.minimum_models:
            self.logger.warning(f"Not enough valid OCR results: got {len(results)}, need {self.minimum_models}")
            # Return the result with highest confidence if we have at least one
            if results:
                best_result = max(results, key=lambda x: x.confidence)
                return best_result
            else:
                return OCRResult(text="", confidence=0.0, model_name="ensemble_failed")
        
        # Perform ensemble voting
        ensembled_result = self._ensemble_results(results)
        processing_time = time.time() - start_time
        
        self.logger.info(f"OCR ensemble completed in {processing_time:.2f}s with confidence {ensembled_result.confidence:.2f}")
        
        return ensembled_result

    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images using model ensembling.
        
        Args:
            images: A list of PIL Images to process
            
        Returns:
            A list of OCRResult objects
        """
        if not self.models_loaded:
            await self.load_models()
            
        self.logger.info(f"Starting OCR ensemble batch processing for {len(images)} images")
        
        # Process images in parallel
        tasks = [self.extract_text(image) for image in images]
        results = await asyncio.gather(*tasks)
        
        self.logger.info(f"Completed OCR ensemble batch processing for {len(images)} images")
        return results

    def _ensemble_results(self, results: List[OCRResult]) -> OCRResult:
        """
        Ensemble multiple OCR results to produce the best combined result.
        
        Args:
            results: List of OCR results from different models
            
        Returns:
            A single OCRResult with the ensembled text
        """
        # If only one result, return it directly
        if len(results) == 1:
            return results[0]
        
        # Calculate normalized weights for each result
        weights = []
        model_names = []
        for result in results:
            model_weight = self.weights.get(result.model_name.split("/")[-1].lower(), 1.0)
            # Use both model weight and result confidence
            weight = model_weight * result.confidence
            weights.append(weight)
            model_names.append(result.model_name)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Equal weights if all weights are zero
            weights = [1.0 / len(weights)] * len(weights)
        
        # Strategy 1: Find the result with the best overall weighted score
        if self.ensemble_config.get("method", "weighted") == "best":
            weighted_scores = [results[i].confidence * weights[i] for i in range(len(results))]
            best_idx = weighted_scores.index(max(weighted_scores))
            return results[best_idx]
        
        # Strategy 2: Use weighted voting between models
        # Compare results pairwise to determine similarity and find the most agreed-upon text
        similarity_matrix = [[0 for _ in range(len(results))] for _ in range(len(results))]
        
        for i in range(len(results)):
            for j in range(len(results)):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate text similarity
                    matcher = SequenceMatcher(None, results[i].text, results[j].text)
                    similarity_matrix[i][j] = matcher.ratio()
        
        # Calculate consensus score for each result
        consensus_scores = []
        for i in range(len(results)):
            score = sum(similarity_matrix[i][j] * weights[j] for j in range(len(results)))
            consensus_scores.append(score)
        
        # Choose the result with highest consensus
        best_idx = consensus_scores.index(max(consensus_scores))
        best_result = results[best_idx]
        
        # Create a new result with the best text and combined metadata
        avg_confidence = sum(r.confidence * weights[i] for i, r in enumerate(results))
        avg_time = sum(r.processing_time for r in results) / len(results)
        
        return OCRResult(
            text=best_result.text,
            confidence=avg_confidence,
            model_name=f"ensemble({','.join(model_names)})",
            processing_time=avg_time
        ) 