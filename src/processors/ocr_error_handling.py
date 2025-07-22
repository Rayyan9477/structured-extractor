"""
OCR Error Handling Classes

Defines standardized error handling and fallback mechanisms for OCR engines.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import traceback


class OCRErrorType(Enum):
    """Types of errors that can occur during OCR processing."""
    MODEL_LOADING = "model_loading"
    INITIALIZATION = "initialization"
    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


@dataclass
class OCRError:
    """Structured information about an OCR error."""
    error_type: OCRErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None
    
    @classmethod
    def from_exception(cls, e: Exception, error_type: OCRErrorType = OCRErrorType.UNKNOWN) -> 'OCRError':
        """Create an OCRError from an exception."""
        return cls(
            error_type=error_type,
            message=str(e),
            details=getattr(e, '__dict__', {}),
            traceback=traceback.format_exc()
        )


class OCREngineError(Exception):
    """Base exception class for OCR engine errors."""
    def __init__(self, error: OCRError):
        self.ocr_error = error
        super().__init__(str(error.message))


class ModelLoadError(OCREngineError):
    """Error raised when model loading fails."""
    pass


class PreprocessingError(OCREngineError):
    """Error raised during image preprocessing."""
    pass


class InferenceError(OCREngineError):
    """Error raised during model inference."""
    pass


class OCRErrorHandler:
    """
    Base class for OCR error handling functionality.
    Provides standardized error handling and recovery mechanisms.
    """
    
    def __init__(self, logger):
        """Initialize error handler with logger."""
        self.logger = logger
        self._errors = []
        self._recovery_attempts = {}
    
    def handle_error(self, error: Exception, context: str = "", error_type: OCRErrorType = OCRErrorType.UNKNOWN) -> OCRError:
        """
        Handle an OCR-related error and return structured error info.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            error_type: The type of error that occurred
            
        Returns:
            Structured error information
        """
        ocr_error = OCRError.from_exception(error, error_type)
        self._errors.append(ocr_error)
        
        # Log error with context
        self.logger.error(
            f"OCR Error ({error_type.value}) in {context}: {str(error)}",
            exc_info=True
        )
        
        return ocr_error
    
    def get_error_history(self) -> List[OCRError]:
        """Get list of errors that have occurred."""
        return self._errors.copy()
    
    def clear_error_history(self):
        """Clear the error history."""
        self._errors = []
    
    def should_attempt_recovery(self, error_type: OCRErrorType) -> bool:
        """
        Determine if recovery should be attempted for an error type.
        
        Args:
            error_type: The type of error to check
            
        Returns:
            True if recovery should be attempted, False otherwise
        """
        # Get current attempt count
        attempts = self._recovery_attempts.get(error_type, 0)
        
        # Allow up to 3 attempts for most errors
        max_attempts = {
            OCRErrorType.MODEL_LOADING: 2,
            OCRErrorType.INITIALIZATION: 2,
            OCRErrorType.PREPROCESSING: 3,
            OCRErrorType.INFERENCE: 3,
            OCRErrorType.POSTPROCESSING: 3,
            OCRErrorType.RESOURCE: 1,
            OCRErrorType.UNKNOWN: 1
        }.get(error_type, 1)
        
        # Increment attempt counter
        self._recovery_attempts[error_type] = attempts + 1
        
        return attempts < max_attempts
    
    def reset_recovery_attempts(self):
        """Reset all recovery attempt counters."""
        self._recovery_attempts = {}
