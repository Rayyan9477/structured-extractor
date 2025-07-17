"""
Logging Configuration for Medical Superbill Extraction System

Provides centralized logging configuration with HIPAA-compliant settings.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Setup comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to console
        log_format: Custom log format string
        rotation: Log rotation setting
        retention: Log retention period
    """
    # Remove default loguru handler
    loguru_logger.remove()
    
    # Default format with timestamp and location info
    if log_format is None:
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
    
    # Console handler
    loguru_logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        filter=_hipaa_filter
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        loguru_logger.add(
            str(log_path),
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            filter=_hipaa_filter
        )
    
    # Configure standard library logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Replace standard library root logger
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    loguru_logger.info(f"Logging initialized - Level: {log_level}")


def _hipaa_filter(record) -> bool:
    """
    Filter log records to remove potential PHI information.
    
    Args:
        record: Log record to filter
        
    Returns:
        True if record should be logged, False otherwise
    """
    import re
    
    message = record["message"]
    
    # PHI patterns to redact
    phi_patterns = [
        r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{16}\b',  # Credit card numbers
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # Dates that might be DOB
    ]
    
    # Redact PHI patterns
    for pattern in phi_patterns:
        message = re.sub(pattern, '[REDACTED]', message)
    
    # Update the message
    record["message"] = message
    
    return True


def get_logger(name: str) -> loguru_logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return loguru_logger.bind(name=name)


class AuditLogger:
    """Special logger for audit trails and compliance logging."""
    
    def __init__(self, audit_file: str = "logs/audit.log"):
        """
        Initialize audit logger.
        
        Args:
            audit_file: Path to audit log file
        """
        self.audit_file = Path(audit_file)
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dedicated audit logger
        self.logger = loguru_logger.bind(audit=True)
        self.logger.add(
            str(self.audit_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AUDIT | {message}",
            level="INFO",
            rotation="100 MB",
            retention="7 years",  # Long retention for compliance
            compression="zip"
        )
    
    def log_document_processed(
        self,
        document_path: str,
        user_id: str,
        extraction_fields: list,
        success: bool
    ) -> None:
        """Log document processing event."""
        self.logger.info(
            f"DOCUMENT_PROCESSED | "
            f"user={user_id} | "
            f"document={document_path} | "
            f"fields={','.join(extraction_fields)} | "
            f"success={success}"
        )
    
    def log_phi_access(
        self,
        user_id: str,
        phi_type: str,
        action: str,
        success: bool
    ) -> None:
        """Log PHI access event."""
        self.logger.info(
            f"PHI_ACCESS | "
            f"user={user_id} | "
            f"phi_type={phi_type} | "
            f"action={action} | "
            f"success={success}"
        )
    
    def log_export(
        self,
        user_id: str,
        export_format: str,
        record_count: int,
        destination: str
    ) -> None:
        """Log data export event."""
        self.logger.info(
            f"DATA_EXPORT | "
            f"user={user_id} | "
            f"format={export_format} | "
            f"records={record_count} | "
            f"destination={destination}"
        )
