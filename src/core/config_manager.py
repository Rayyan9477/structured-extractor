"""
Configuration Manager for Medical Superbill Extraction System

Handles loading and managing configuration settings from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and access for the extraction system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        current_dir = Path(__file__).parent.parent.parent
        return str(current_dir / "config" / "config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def get_cache_dir(self) -> str:
        """Get model cache directory."""
        # Check if models_cache exists, otherwise use models directory
        cache_dir = self.get("global.cache_dir", "models")
        project_root = Path(__file__).parent.parent.parent
        
        # Check both possible model directories
        models_cache_path = project_root / "models_cache"
        models_path = project_root / "models"
        
        if models_cache_path.exists():
            return str(models_cache_path)
        elif models_path.exists():
            return str(models_path)
        else:
            # Default to models directory
            return str(models_path)

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {"level": "INFO", "log_file": None})

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'models.ocr.monkey_ocr')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"Configuration key '{key}' not found, using default: {default}")
            return default
    
    def get_model_config(self, model_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Args:
            model_type: Type of model ('ocr' or 'extraction')
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return self.get(f"{model_type}.{model_name}", {})
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get the correct path for a model based on the downloaded structure.
        
        Args:
            model_name: Model name (e.g., 'numind/NuExtract-2.0-8B')
            
        Returns:
            Local path to the model
        """
        from pathlib import Path
        
        # Get base models directory
        models_dir = Path(self.get("global.cache_dir", "models"))
        
        # Convert model name to directory name
        model_dir_name = model_name.replace("/", "_")
        model_path = models_dir / model_dir_name
        
        if model_path.exists():
            return str(model_path)
        
        # Fallback to original name if directory doesn't exist
        return model_name
    
    def get_extraction_fields(self) -> Dict[str, list]:
        """Get field extraction configuration."""
        return self.get("extraction_fields", {})
    
    def get_medical_codes_config(self) -> Dict[str, Any]:
        """Get medical codes configuration."""
        return self.get("medical_codes", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security and compliance configuration."""
        return self.get("security", {})
    
    def get_model_path(self, model_identifier: str) -> Path:
        """
        Get the local path for a specific model.
        
        Args:
            model_identifier: Model identifier (e.g., 'echo840/MonkeyOCR', 'nanonets/Nanonets-OCR-s')
            
        Returns:
            Path to the local model directory
        """
        cache_dir = Path(self.get_cache_dir())
        
        # Convert model identifier to directory name
        # e.g., 'echo840/MonkeyOCR' -> 'echo840_MonkeyOCR'
        if '/' in model_identifier:
            dir_name = model_identifier.replace('/', '_')
        else:
            dir_name = model_identifier
            
        model_path = cache_dir / dir_name
        
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            
        return model_path
    
    def validate_model_paths(self) -> Dict[str, bool]:
        """
        Validate that all required models exist locally.
        
        Returns:
            Dictionary mapping model names to availability status
        """
        required_models = {
            'echo840/MonkeyOCR': 'echo840_MonkeyOCR',
            'nanonets/Nanonets-OCR-s': 'nanonets_Nanonets-OCR-s', 
            'numind/NuExtract-2.0-8B': 'numind_NuExtract-2.0-8B'
        }
        
        cache_dir = Path(self.get_cache_dir())
        results = {}
        
        for model_id, dir_name in required_models.items():
            model_path = cache_dir / dir_name
            results[model_id] = model_path.exists() and model_path.is_dir()
            
        return results
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance settings."""
        return self.get("performance", {})
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def update_config(self, key: str, value: Any) -> None:
        """
        Update configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites current file.
        """
        save_path = output_path or str(self.config_path)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.safe_dump(self.config, file, default_flow_style=False, indent=2)
                logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
