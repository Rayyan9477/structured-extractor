"""
OCR Ensemble Engine with Enhanced Error Handling

This module provides an enhanced OCR ensemble that combines multiple OCR engines
with intelligent fallback mechanisms and result verification.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
from difflib import SequenceMatcher
import re
from dataclasses import dataclass

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence
from src.processors.ocr_error_handling import OCRErrorHandler, OCRErrorType


@dataclass
class ResultConfidence:
    """Detailed confidence metrics for OCR results."""
    text_quality: float = 0.0  # Measure of text quality (completeness, formatting)
    structural_quality: float = 0.0  # Measure of structural elements detected
    consistency: float = 0.0  # Agreement with other results
    length_score: float = 0.0  # Score based on expected text length
    model_confidence: float = 0.0  # Original model confidence
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total confidence score."""
        weights = {
            'text_quality': 0.3,
            'structural_quality': 0.2,
            'consistency': 0.25,
            'length_score': 0.1,
            'model_confidence': 0.15
        }
        return sum(
            getattr(self, field) * weight 
            for field, weight in weights.items()
        )


class OCREnsembleManager(OCRErrorHandler):
    """
    Enhanced OCR ensemble manager with robust error handling and result verification.
    
    Features:
    1. Intelligent engine selection
    2. Result verification and validation
    3. Dynamic fallback mechanisms
    4. Enhanced confidence scoring
    5. Structured error handling
    """
    
    def __init__(self, config: ConfigManager):
        """Initialize the OCR ensemble manager."""
        self.config = config
        self.logger = get_logger(__name__)
        super().__init__(self.logger)
        
        # Load configuration
        self.ensemble_config = config.get("ocr.ensemble", {})
        
        # Configurable parameters
        self.text_length_threshold = self.ensemble_config.get("text_length_threshold", 100)
        self.confidence_threshold = self.ensemble_config.get("confidence_threshold", 0.6)
        self.similarity_threshold = self.ensemble_config.get("similarity_threshold", 0.7)
        
        # Initialize quality checkers
        self._init_quality_checks()
    
    def _init_quality_checks(self):
        """Initialize text quality validation patterns."""
        self.quality_patterns = {
            'medical_terms': re.compile(
                r'\b(patient|diagnosis|treatment|procedure|prescription|'
                r'medication|dosage|icd|cpt|npi)\b', 
                re.IGNORECASE
            ),
            'structured_elements': re.compile(
                r'(<table>|</table>|<page|☐|☑|<img>|<watermark>)'
            ),
            'noise_patterns': re.compile(
                r'([^\x20-\x7E\n]|[\x00-\x1F\x7F-\xFF])+|'
                r'(.)\1{3,}'  # Repeated chars
            )
        }
    
    async def verify_results(
        self, 
        results: List[OCRResult],
        min_results: int = 2
    ) -> Tuple[List[OCRResult], bool]:
        """
        Verify OCR results and filter out low-quality ones.
        
        Args:
            results: List of OCR results to verify
            min_results: Minimum number of results needed
            
        Returns:
            Tuple of (filtered_results, needs_retry)
        """
        if not results:
            return [], True
            
        verified_results = []
        
        for result in results:
            confidence_metrics = self._calculate_detailed_confidence(result)
            
            # Update result confidence with new metrics
            result.confidence = confidence_metrics.total_score
            
            if confidence_metrics.total_score >= self.confidence_threshold:
                verified_results.append(result)
        
        needs_retry = len(verified_results) < min_results
        return verified_results, needs_retry
    
    def _calculate_detailed_confidence(self, result: OCRResult) -> ResultConfidence:
        """
        Calculate detailed confidence metrics for a result.
        
        Args:
            result: The OCR result to analyze
            
        Returns:
            Detailed confidence metrics
        """
        confidence = ResultConfidence()
        
        # Store original model confidence
        confidence.model_confidence = result.confidence
        
        # Text quality checks
        text = result.text.strip()
        total_chars = len(text)
        
        if not text:
            return confidence
            
        # Check for medical terms and structural elements
        medical_term_count = len(self.quality_patterns['medical_terms'].findall(text))
        structural_elements = len(self.quality_patterns['structured_elements'].findall(text))
        
        # Calculate noise ratio
        noise_matches = self.quality_patterns['noise_patterns'].findall(text)
        noise_chars = sum(len(m[0]) for m in noise_matches)
        noise_ratio = noise_chars / total_chars if total_chars > 0 else 1.0
        
        # Text quality score
        confidence.text_quality = min(1.0, max(0.0,
            0.5 +  # Base score
            min(0.3, medical_term_count * 0.05) +  # Medical terms bonus
            (0.2 if total_chars > self.text_length_threshold else 0.0) -  # Length bonus
            (noise_ratio * 0.5)  # Noise penalty
        ))
        
        # Structural quality score
        confidence.structural_quality = min(1.0, structural_elements * 0.1)
        
        # Length score
        expected_length = self.text_length_threshold
        length_ratio = total_chars / expected_length
        confidence.length_score = min(1.0, max(0.0,
            0.5 + (0.5 * min(1.0, length_ratio))
        ))
        
        return confidence
    
    def _calculate_result_similarity(self, result1: OCRResult, result2: OCRResult) -> float:
        """Calculate similarity between two OCR results."""
        matcher = SequenceMatcher(None, result1.text, result2.text)
        return matcher.ratio()
    
    def update_consistency_scores(self, results: List[OCRResult]) -> List[OCRResult]:
        """
        Update consistency scores based on agreement between results.
        
        Args:
            results: List of results to analyze
            
        Returns:
            Results with updated confidence scores
        """
        if len(results) < 2:
            return results
            
        # Calculate pairwise similarities
        for i, result1 in enumerate(results):
            similarities = []
            for j, result2 in enumerate(results):
                if i != j:
                    similarity = self._calculate_result_similarity(result1, result2)
                    similarities.append(similarity)
            
            # Update result's consistency score
            avg_similarity = sum(similarities) / len(similarities)
            confidence_metrics = self._calculate_detailed_confidence(result1)
            confidence_metrics.consistency = avg_similarity
            result1.confidence = confidence_metrics.total_score
        
        return results
