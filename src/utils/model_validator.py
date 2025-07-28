"""
Model Validation Utility

Provides comprehensive model validation and health checks for the extraction system.
"""

import asyncio
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class ModelValidator:
    """Comprehensive model validation and health checking."""
    
    def __init__(self, config: ConfigManager):
        """Initialize model validator."""
        self.config = config
        self.logger = get_logger(__name__)
        
    def validate_model_files(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required model files exist.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        model_path = Path(self.config.get_model_path(model_name))
        issues = []
        
        if not model_path.exists():
            issues.append(f"Model directory does not exist: {model_path}")
            return False, issues
        
        # Required files for most models
        required_files = [
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "model.safetensors.index.json"
        ]
        
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                issues.append(f"Required file missing: {file_path}")
        
        # Check for model weights files
        model_files = list(model_path.glob("model-*.safetensors"))
        if not model_files:
            issues.append(f"No model weight files found in {model_path}")
        
        # Check config.json integrity
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if not config:
                    issues.append("config.json is empty or invalid")
            except Exception as e:
                issues.append(f"Error reading config.json: {e}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def get_model_size(self, model_name: str) -> Optional[int]:
        """Get the total size of model files in bytes."""
        try:
            model_path = Path(self.config.get_model_path(model_name))
            if not model_path.exists():
                return None
            
            total_size = 0
            for file_path in model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
        except Exception as e:
            self.logger.error(f"Error calculating model size for {model_name}: {e}")
            return None
    
    def check_device_compatibility(self, model_name: str) -> Dict[str, any]:
        """Check if model can be loaded on available devices."""
        compatibility = {
            "cpu": True,
            "cuda": False,
            "mps": False,
            "issues": []
        }
        
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                compatibility["cuda"] = True
                device_props = torch.cuda.get_device_properties(0)
                compatibility["cuda_memory"] = device_props.total_memory
                compatibility["cuda_compute_capability"] = f"{device_props.major}.{device_props.minor}"
            else:
                compatibility["issues"].append("CUDA not available")
            
            # Check MPS availability
            if torch.backends.mps.is_available():
                compatibility["mps"] = True
            else:
                compatibility["issues"].append("MPS not available")
            
            # Check model size vs available memory
            model_size = self.get_model_size(model_name)
            if model_size:
                compatibility["model_size"] = model_size
                
                # Check if model fits in CPU memory (rough estimate)
                import psutil
                cpu_memory = psutil.virtual_memory().total
                if model_size > cpu_memory * 0.8:  # 80% of available memory
                    compatibility["issues"].append("Model may be too large for CPU memory")
                
                # Check if model fits in GPU memory
                if compatibility["cuda"] and model_size > compatibility["cuda_memory"] * 0.8:
                    compatibility["issues"].append("Model may be too large for GPU memory")
            
        except Exception as e:
            compatibility["issues"].append(f"Error checking device compatibility: {e}")
        
        return compatibility
    
    async def test_model_loading(self, model_name: str, engine_class) -> Tuple[bool, List[str]]:
        """
        Test actual model loading.
        
        Args:
            model_name: Name of the model to test
            engine_class: Engine class to use for testing
            
        Returns:
            Tuple of (success, list_of_issues)
        """
        issues = []
        
        try:
            # Create engine instance
            engine = engine_class(self.config)
            
            # Test model loading
            if hasattr(engine, 'load_models'):
                await engine.load_models()
                if hasattr(engine, 'models_loaded') and engine.models_loaded:
                    return True, issues
                else:
                    issues.append("Model loading completed but models_loaded flag is False")
            elif hasattr(engine, 'load_model'):
                await engine.load_model()
                if hasattr(engine, 'model') and engine.model is not None:
                    return True, issues
                else:
                    issues.append("Model loading completed but model is None")
            else:
                issues.append("Engine does not have load_models or load_model method")
                
        except Exception as e:
            issues.append(f"Model loading failed: {e}")
        
        return False, issues
    
    def validate_all_models(self) -> Dict[str, Dict]:
        """Validate all configured models."""
        results = {}
        
        # Models to validate
        models_to_check = [
            ("nanonets/Nanonets-OCR-s", "Nanonets OCR"),
            ("numind/NuExtract-2.0-8B", "NuExtract")
        ]
        
        for model_name, display_name in models_to_check:
            self.logger.info(f"Validating {display_name}...")
            
            # File validation
            is_valid, file_issues = self.validate_model_files(model_name)
            
            # Device compatibility
            compatibility = self.check_device_compatibility(model_name)
            
            # Model size
            model_size = self.get_model_size(model_name)
            
            results[model_name] = {
                "display_name": display_name,
                "file_validation": {
                    "valid": is_valid,
                    "issues": file_issues
                },
                "device_compatibility": compatibility,
                "model_size": model_size,
                "model_size_gb": model_size / (1024**3) if model_size else None
            }
        
        return results
    
    def print_validation_report(self, results: Dict[str, Dict]):
        """Print a comprehensive validation report."""
        print("🏥 Medical Superbill Extractor - Model Validation Report")
        print("=" * 80)
        
        for model_name, result in results.items():
            display_name = result["display_name"]
            file_validation = result["file_validation"]
            compatibility = result["device_compatibility"]
            model_size_gb = result["model_size_gb"]
            
            print(f"\n📋 {display_name} ({model_name})")
            print("-" * 60)
            
            # File validation
            if file_validation["valid"]:
                print("  ✅ File validation: PASSED")
            else:
                print("  ❌ File validation: FAILED")
                for issue in file_validation["issues"]:
                    print(f"    - {issue}")
            
            # Model size
            if model_size_gb:
                print(f"  📦 Model size: {model_size_gb:.2f} GB")
            
            # Device compatibility
            print("  🔧 Device compatibility:")
            print(f"    - CPU: {'✅' if compatibility['cpu'] else '❌'}")
            print(f"    - CUDA: {'✅' if compatibility['cuda'] else '❌'}")
            print(f"    - MPS: {'✅' if compatibility['mps'] else '❌'}")
            
            if compatibility["cuda"]:
                print(f"    - CUDA Memory: {compatibility['cuda_memory'] / (1024**3):.2f} GB")
                print(f"    - Compute Capability: {compatibility['cuda_compute_capability']}")
            
            # Issues
            if compatibility["issues"]:
                print("  ⚠️  Compatibility issues:")
                for issue in compatibility["issues"]:
                    print(f"    - {issue}")
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 Summary:")
        
        total_models = len(results)
        valid_models = sum(1 for r in results.values() if r["file_validation"]["valid"])
        
        print(f"  Total models: {total_models}")
        print(f"  Valid models: {valid_models}")
        print(f"  Invalid models: {total_models - valid_models}")
        
        if valid_models == total_models:
            print("  🎉 All models are valid!")
        else:
            print("  ⚠️  Some models have issues. Please check the details above.")


async def main():
    """Run model validation."""
    config = ConfigManager()
    validator = ModelValidator(config)
    
    # Validate all models
    results = validator.validate_all_models()
    
    # Print report
    validator.print_validation_report(results)
    
    return all(r["file_validation"]["valid"] for r in results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 