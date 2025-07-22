"""
Unified OCR Engine for Medical Superbill Extraction

Integrates multiple OCR models and provides a unified interface for text extraction.
This engine can handle both printed and handwritten text by using appropriate models.
"""

import asyncio
import torch
import time
import gc
from typing import List, Dict, Any, Optional, Tuple
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
from src.processors.ocr_error_handling import OCRErrorHandler, OCRErrorType
from src.processors.ocr_ensemble_manager import OCREnsembleManager
from src.processors.resource_manager import (
    ResourceManager, ModelType, ModelPriority, ModelResource
)


class UnifiedOCREngine(OCRErrorHandler):
    """
    Unified OCR engine that can use multiple models independently 
    and provide the best results for medical document processing.
    
    Features:
    1. Intelligent model management and resource allocation
    2. Enhanced result verification and validation
    3. Robust error handling and recovery
    4. Memory-efficient model loading/unloading
    5. Optimized batch processing
    """
    
    async def cleanup(self):
        """
        Clean up resources and unload models.
        Should be called when the engine is no longer needed.
        """
        self.logger.info("Starting OCR engine cleanup...")
        
        # Unload all models through resource manager
        for engine_name in list(self.engines.keys()):
            try:
                await self.resource_manager.unload_model(f"ocr_{engine_name}")
                self.logger.debug(f"Unloaded {engine_name}")
            except Exception as e:
                self.logger.warning(f"Error unloading {engine_name}: {e}")
            
            # Clear engine reference
            self.engines.pop(engine_name, None)
        
        # Force memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("OCR engine cleanup completed")
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the unified OCR engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        super().__init__(self.logger)
        
        # Initialize managers
        self.resource_manager = ResourceManager(config)
        self.ensemble_manager = OCREnsembleManager(config)
        
        # Get OCR configuration
        self.ocr_config = self.config.get("ocr", {})
        self.use_models = self.ocr_config.get("ensemble.use_models", ["nanonets_ocr", "monkey_ocr"])
        self.weights = self.ocr_config.get("ensemble.weights", {})
        self.method = self.ocr_config.get("ensemble.method", "best_confidence")
        
        # Device selection moved to resource manager
        self.default_device = self._get_device()  # Fallback device
        self.logger.info(f"Unified OCR engine initialized with default device: {self.default_device}")
        
        # Engine configuration
        self.engine_configs = {
            "monkey_ocr": {
                "type": ModelType.OCR,
                "priority": ModelPriority.HIGH,
                "class": MonkeyOCREngine,
                "sub_models": ["structure", "recognition", "relation"]
            },
            "nanonets_ocr": {
                "type": ModelType.OCR,
                "priority": ModelPriority.HIGH,
                "class": NanonetsOCREngine,
                "sub_models": []
            },
            "trocr": {
                "type": ModelType.OCR,
                "priority": ModelPriority.MEDIUM,
                "class": TrOCREngine,
                "sub_models": []
            }
        }
        
        # Track initialized engines
        self.engines = {}
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
        """Initialize available OCR engines with proper error handling and fallbacks."""
        initialized = []
        failed = []
        
        for engine_name in self.use_models:
            config = self.engine_configs.get(engine_name)
            if not config:
                self.logger.warning(f"Unknown engine type: {engine_name}")
                failed.append(engine_name)
                continue
            
            try:
                # Create engine instance for initialization only
                # Actual loading happens in load_models()
                engine = config["class"](self.config)
                self.engines[engine_name] = engine
                initialized.append(engine_name)
                self.logger.debug(f"{engine_name} prepared for initialization")
                
            except Exception as e:
                error = self.handle_error(
                    e,
                    context=f"Preparing {engine_name}",
                    error_type=OCRErrorType.INITIALIZATION
                )
                failed.append(engine_name)
        
        # Log initialization results
        if not initialized:
            error = "No OCR engines could be initialized"
            self.logger.error(error)
            raise RuntimeError(error)
        
        total = len(self.use_models)
        self.logger.info(
            f"Prepared {len(initialized)}/{total} requested OCR engines. "
            f"Failed: {', '.join(failed) if failed else 'None'}"
        )
        
        if failed:
            self.logger.warning(
                "Some engines failed to initialize. "
                "System will continue with reduced capabilities."
            )

    async def load_models(self) -> None:
        """
        Load all configured OCR models with comprehensive error handling
        and resource management.
        
        Features:
        1. Parallel loading with controlled concurrency
        2. Resource-aware loading order
        3. Smart fallback mechanisms
        4. Progress tracking and reporting
        """
        self.logger.info("Loading OCR models...")
        start_time = time.time()
        
        # Track loading progress
        total_count = len(self.engines)
        success_count = 0
        failed_engines = []
        
        # Determine loading order based on priority
        loading_order = sorted(
            self.engines.items(),
            key=lambda x: self.engine_configs[x[0]]["priority"].value,
            reverse=True
        )
        
        # Load models with controlled concurrency
        sem = asyncio.Semaphore(2)  # Load up to 2 models at once
        
        async def load_engine(engine_name: str, engine_class) -> bool:
            async with sem:
                try:
                    return await self._load_engine_safely(engine_name, engine_class)
                except Exception as e:
                    error = self.handle_error(
                        e,
                        context=f"Loading {engine_name}",
                        error_type=OCRErrorType.MODEL_LOADING
                    )
                    return False
        
        # Create loading tasks
        load_tasks = [
            load_engine(name, engine)
            for name, engine in loading_order
        ]
        
        # Process loading results
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        for (engine_name, _), success in zip(loading_order, results):
            if isinstance(success, bool) and success:
                success_count += 1
                self.logger.info(f"{engine_name} loaded successfully")
            else:
                failed_engines.append(engine_name)
                self.logger.error(f"{engine_name} failed to load")
        
        # Log final status
        processing_time = time.time() - start_time
        self.logger.info(
            f"Model loading completed in {processing_time:.1f}s. "
            f"Loaded: {success_count}/{total_count}"
        )
        
        if failed_engines:
            self.logger.warning(f"Failed engines: {', '.join(failed_engines)}")
            if success_count == 0:
                raise RuntimeError("All OCR engines failed to load")
            else:
                self.logger.warning("Continuing with reduced model availability")
                
        # Initialize ensemble manager with loaded engines
        await self.ensemble_manager.initialize(
            self.engines,
            {name: self.weights.get(name, 1.0) for name in self.engines}
        )

    async def _load_engine_safely(self, engine_name: str, engine_class) -> bool:
        """
        Load an engine safely using resource management and error handling.
        
        Args:
            engine_name: Name of the engine to load
            engine_class: Class of the engine to instantiate
            
        Returns:
            bool: True if engine loaded successfully, False otherwise
        """
        config = self.engine_configs.get(engine_name)
        if not config:
            self.logger.error(f"No configuration found for engine {engine_name}")
            return False
        
        try:
            # Create engine instance
            engine = engine_class(self.config)
            
            # Define model loader function
            async def model_loader(device: str):
                engine.device = device
                await engine.load_models()
                return engine
            
            # Load using resource manager
            loaded_engine, device = await self.resource_manager.load_model(
                model_id=f"ocr_{engine_name}",
                model_type=config["type"],
                priority=config["priority"],
                loader_func=model_loader,
                preferred_device=self.default_device,
                metadata={
                    "engine_name": engine_name,
                    "sub_models": config["sub_models"],
                }
            )
            
            # Store in engines dictionary
            self.engines[engine_name] = loaded_engine
            self.logger.info(f"{engine_name} loaded successfully on {device}")
            
            return True
            
        except Exception as e:
            error = self.handle_error(
                e,
                context=f"Loading {engine_name}",
                error_type=OCRErrorType.MODEL_LOADING
            )
            
            return False

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from an image using available OCR engines with comprehensive
        error handling, result verification, and intelligent result selection.
        
        Features:
        1. Multi-engine parallel processing
        2. Result verification and validation
        3. Dynamic fallback mechanisms
        4. Intelligent result selection
        5. Resource-aware processing
        
        Args:
            image: A PIL Image to process.
            
        Returns:
            An OCRResult object with the extracted text and metadata.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        self.logger.debug("Starting text extraction...")
        start_time = time.time()
        
        # Track extraction state
        results = []
        errors = []
        degraded_mode = False
        
        # Check available engines
        available_engines = [
            (name, engine) for name, engine in self.engines.items()
            if self.resource_manager.get_model_info(f"ocr_{name}")
        ]
        
        if not available_engines:
            error = "No OCR engines currently available"
            self.logger.error(error)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="unified_ocr_not_ready", 
                processing_time=time.time() - start_time
            )
        
        # Process with each engine concurrently
        sem = asyncio.Semaphore(2)  # Limit concurrent processing
        
        async def process_engine(engine_name: str, engine) -> Tuple[str, OCRResult]:
            async with sem:
                try:
                    result = await engine.extract_text(image)
                    # Update last used timestamp
                    self.resource_manager.get_model_info(f"ocr_{engine_name}").last_used = time.time()
                    return engine_name, result
                except Exception as e:
                    error = self.handle_error(
                        e,
                        context=f"Extracting with {engine_name}",
                        error_type=OCRErrorType.INFERENCE
                    )
                    raise
        
        # Create processing tasks
        tasks = [
            process_engine(name, engine)
            for name, engine in available_engines
        ]
        
        # Run extractions
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for task_result in completed_tasks:
            if isinstance(task_result, tuple):
                # Successful extraction
                engine_name, result = task_result
                if result and result.text:
                    result.model_name = f"{result.model_name}_{engine_name}"
                    results.append(result)
            elif isinstance(task_result, Exception):
                # Failed extraction
                errors.append(task_result)
                degraded_mode = True
        
        # Verify results
        verified_results, needs_retry = await self.ensemble_manager.verify_results(results)
        
        # Handle verification failures
        if needs_retry and not degraded_mode:
            self.logger.warning("Initial results need improvement, attempting retry")
            # Attempt retry with adjusted parameters
            retry_results = []
            for engine_name, engine in available_engines:
                if engine_name not in [r.model_name.split("_")[-1] for r in verified_results]:
                    try:
                        result = await engine.extract_text(image)
                        if result and result.text:
                            result.model_name = f"{result.model_name}_{engine_name}"
                            retry_results.append(result)
                    except Exception as e:
                        self.handle_error(
                            e,
                            context=f"Retry extraction with {engine_name}",
                            error_type=OCRErrorType.INFERENCE
                        )
            verified_results.extend(retry_results)
        
        # Handle case where all engines failed
        if not verified_results:
            self.logger.error("All OCR engines failed")
            self.logger.error("No valid results after verification")
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name="unified_ocr_failed", 
                processing_time=time.time() - start_time
            )
        
        # Update consistency scores
        final_results = self.ensemble_manager.update_consistency_scores(verified_results)
        
        # Select best result based on method and mode
        if len(final_results) == 1:
            best_result = final_results[0]
        else:
            if degraded_mode:
                # In degraded mode, prefer higher confidence results
                best_result = max(final_results, key=lambda x: x.confidence)
            else:
                # Normal mode - use configured selection method
                if self.method == "weighted":
                    # Use engine weights from config
                    weighted_results = []
                    for result in final_results:
                        engine_name = result.model_name.split("_")[-1]
                        weight = self.weights.get(engine_name, 1.0)
                        weighted_results.append(
                            (result, result.confidence * weight)
                        )
                    best_result = max(weighted_results, key=lambda x: x[1])[0]
                    
                elif self.method == "ensemble":
                    # Combine results using ensemble manager
                    ensemble_result = await self.ensemble_manager.combine_results(final_results)
                    best_result = ensemble_result
                    
                else:  # "best_confidence"
                    best_result = max(final_results, key=lambda x: x.confidence)
        
        # Record processing time
        best_result.processing_time = time.time() - start_time
        
        # Log results
        self.logger.debug(
            f"Best result from {best_result.model_name} "
            f"with confidence {best_result.confidence:.2f} "
            f"in {best_result.processing_time:.2f}s"
        )
        
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
        Extract text from a batch of images using all available models with robust error handling.
        
        Features:
        1. Parallel processing with controlled concurrency
        2. Per-image error handling and retries
        3. Batch-level progress tracking
        4. Graceful degradation if some images fail
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of OCRResult objects, one per input image
        """
        if not images:
            return []
            
        self.logger.info(f"Starting batch extraction for {len(images)} images")
        start_time = time.time()
        
        # Process in parallel with controlled concurrency
        max_concurrent = min(len(images), 4)  # Limit concurrent processing
        sem = asyncio.Semaphore(max_concurrent)
        
        async def process_image(image: Image.Image, index: int) -> Tuple[int, OCRResult]:
            async with sem:
                try:
                    result = await self.extract_text(image)
                    self.logger.debug(f"Completed image {index + 1}/{len(images)}")
                    return index, result
                except Exception as e:
                    error = self.handle_error(
                        e,
                        context=f"Batch processing image {index + 1}",
                        error_type=OCRErrorType.INFERENCE
                    )
                    # Return empty result on failure
                    return index, OCRResult(
                        text="",
                        confidence=0.0,
                        model_name="batch_failed",
                        processing_time=0.0
                    )
        
        # Create tasks for all images
        tasks = [
            process_image(image, i) 
            for i, image in enumerate(images)
        ]
        
        # Process all images
        results_with_index = await asyncio.gather(*tasks)
        
        # Sort results back into original order
        sorted_results = [
            r[1] for r in sorted(results_with_index, key=lambda x: x[0])
        ]
        
        # Log batch statistics
        failed = len([r for r in sorted_results if r.confidence == 0.0])
        processing_time = time.time() - start_time
        
        self.logger.info(
            f"Batch processing completed in {processing_time:.2f}s. "
            f"Processed: {len(images)}, Failed: {failed}"
        )
        
        return sorted_results

    def get_available_engines(self) -> List[str]:
        """Get list of available and loaded engines."""
        return [
            name for name, engine in self.engines.items() 
            if hasattr(engine, 'models_loaded') and engine.models_loaded
        ]

    async def _extract_with_engine(
        self, 
        engine_name: str, 
        engine: Any, 
        image: Image.Image
    ) -> Tuple[str, OCRResult]:
        """
        Extract text from an image using a specific engine with error handling.
        
        Args:
            engine_name: Name of the engine to use
            engine: Engine instance
            image: Image to process
            
        Returns:
            Tuple of (engine_name, OCRResult)
            
        Raises:
            Exception if extraction fails after retries
        """
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                result = await engine.extract_text(image)
                
                # Validate result
                if not result or not result.text:
                    raise ValueError(f"Empty result from {engine_name}")
                
                return engine_name, result
                
            except Exception as e:
                error = self.handle_error(
                    e,
                    context=f"Extracting with {engine_name} (attempt {retry_count + 1}/{max_retries + 1})",
                    error_type=OCRErrorType.INFERENCE
                )
                
                if retry_count == max_retries:
                    self.logger.error(f"Failed to extract with {engine_name} after {max_retries + 1} attempts")
                    raise
                
                retry_count += 1
                delay = retry_count  # Linear backoff
                self.logger.warning(f"Retrying {engine_name} extraction after {delay}s delay")
                await asyncio.sleep(delay)
    
    def _select_best_result(self, results: List[OCRResult], degraded_mode: bool = False) -> OCRResult:
        """
        Select the best result from multiple OCR outputs.
        
        Args:
            results: List of OCR results to choose from
            degraded_mode: Whether we're operating in degraded mode
            
        Returns:
            The best OCR result based on configured selection method
        """
        if not results:
            return OCRResult(text="", confidence=0.0, model_name="no_results")
        
        if len(results) == 1:
            return results[0]
        
        if degraded_mode:
            # In degraded mode, prefer higher confidence results more strongly
            self.logger.debug("Using degraded mode result selection")
            return max(results, key=lambda x: x.confidence)
        
        # Apply configured selection method
        if self.method == "weighted":
            # Use engine weights from config
            weighted_results = []
            for result in results:
                engine_name = result.model_name.split("_")[-1]
                weight = self.weights.get(engine_name, 1.0)
                weighted_results.append(
                    (result, result.confidence * weight)
                )
            return max(weighted_results, key=lambda x: x[1])[0]
            
        elif self.method == "ensemble":
            # Simple averaging of results
            text_votes = {}
            for result in results:
                text = result.text
                if text not in text_votes:
                    text_votes[text] = []
                text_votes[text].append(result.confidence)
            
            # Select text with highest average confidence
            best_text = max(
                text_votes.items(),
                key=lambda x: sum(x[1]) / len(x[1])
            )[0]
            
            # Find result with this text and highest confidence
            matching_results = [r for r in results if r.text == best_text]
            return max(matching_results, key=lambda x: x.confidence)
            
        else:  # "best_confidence"
            return max(results, key=lambda x: x.confidence)

    def get_engine_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of each engine with detailed status."""
        capabilities = {}
        
        for name, engine in self.engines.items():
            engine_info = {
                "ready": hasattr(engine, 'models_loaded') and engine.models_loaded,
                "error_count": len([e for e in self.get_error_history() 
                                if e.details.get('engine') == name]),
                "supports_retry": hasattr(engine, 'should_retry'),
            }
            
            # Get engine-specific capabilities
            if hasattr(engine, 'get_feature_capabilities'):
                engine_info.update(engine.get_feature_capabilities())
            else:
                engine_info["basic_ocr"] = True
                
            capabilities[name] = engine_info
        
        return capabilities


# Maintain backwards compatibility
class OCREngine(UnifiedOCREngine):
    """Backwards compatible OCR engine class."""
    pass
