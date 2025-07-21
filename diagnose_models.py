"""
Model Diagnostic Tool for Medical Superbill Extraction System

Provides comprehensive diagnostics for model setup, loading, and functionality.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.models.standalone_model_manager import StandaloneModelManager
from src.processors.monkey_ocr_engine import MonkeyOCREngine
from src.processors.nanonets_ocr_engine import NanonetsOCREngine
from src.extractors.nuextract_engine import NuExtractEngine


class ModelDiagnostics:
    """Comprehensive model diagnostics system."""
    
    def __init__(self):
        """Initialize diagnostics."""
        self.logger = get_logger(__name__)
        try:
            self.config = ConfigManager()
            self.model_manager = StandaloneModelManager(self.config)
        except Exception as e:
            self.logger.error(f"Failed to initialize diagnostics: {e}")
            raise
    
    async def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete diagnostic suite."""
        self.logger.info("Starting comprehensive model diagnostics...")
        
        results = {
            "system_status": self._check_system_status(),
            "model_availability": self._check_model_availability(),
            "model_structure": self._check_model_structure(),
            "model_loading": await self._test_model_loading(),
            "basic_functionality": await self._test_basic_functionality(),
            "recommendations": self._get_recommendations()
        }
        
        self.logger.info("Diagnostics completed")
        return results
    
    def _check_system_status(self) -> Dict[str, Any]:
        """Check system requirements and resources."""
        self.logger.info("Checking system status...")
        
        import torch
        import psutil
        
        status = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            status["gpu_info"] = []
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                status["gpu_info"].append({
                    "name": gpu_props.name,
                    "memory_total": gpu_props.total_memory / 1e9,
                    "memory_free": (gpu_props.total_memory - torch.cuda.memory_allocated(i)) / 1e9
                })
        
        # Memory info
        memory = psutil.virtual_memory()
        status["ram"] = {
            "total": memory.total / 1e9,
            "available": memory.available / 1e9,
            "percent_used": memory.percent
        }
        
        status["cpu_cores"] = psutil.cpu_count()
        
        return status
    
    def _check_model_availability(self) -> Dict[str, Any]:
        """Check if models are available locally."""
        self.logger.info("Checking model availability...")
        
        availability = self.model_manager.check_model_availability()
        
        results = {}
        for model_key, available in availability.items():
            model_info = self.model_manager.models_info[model_key]
            model_path = Path(model_info.path)
            
            results[model_key] = {
                "available": available,
                "path": str(model_path),
                "exists": model_path.exists(),
                "size_gb": self._get_directory_size(model_path) / 1e9 if model_path.exists() else 0
            }
        
        return results
    
    def _check_model_structure(self) -> Dict[str, Any]:
        """Check model directory structure."""
        self.logger.info("Checking model structure...")
        
        results = {}
        
        for model_key in self.model_manager.models_info.keys():
            model_path = self.model_manager.get_model_path(model_key)
            
            if not model_path:
                results[model_key] = {"valid": False, "reason": "Model not found"}
                continue
            
            path = Path(model_path)
            structure_info = {"valid": False, "files": [], "directories": []}
            
            if path.exists():
                structure_info["files"] = [f.name for f in path.iterdir() if f.is_file()]
                structure_info["directories"] = [d.name for d in path.iterdir() if d.is_dir()]
                structure_info["valid"] = self.model_manager.validate_model_structure(model_key)
                
                # Check specific requirements
                if model_key == "monkey_ocr":
                    required_dirs = ["Recognition", "Structure", "Relation"]
                    structure_info["required_dirs"] = required_dirs
                    structure_info["missing_dirs"] = [d for d in required_dirs if d not in structure_info["directories"]]
                
                elif model_key in ["nanonets_ocr", "nuextract"]:
                    required_files = ["config.json", "tokenizer_config.json"]
                    structure_info["required_files"] = required_files
                    structure_info["missing_files"] = [f for f in required_files if f not in structure_info["files"]]
            
            results[model_key] = structure_info
        
        return results
    
    async def _test_model_loading(self) -> Dict[str, Any]:
        """Test loading each model."""
        self.logger.info("Testing model loading...")
        
        results = {}
        
        # Test MonkeyOCR
        try:
            monkey_engine = MonkeyOCREngine(self.config)
            await monkey_engine.load_models()
            results["monkey_ocr"] = {"loaded": True, "error": None}
        except Exception as e:
            results["monkey_ocr"] = {"loaded": False, "error": str(e)}
        
        # Test Nanonets OCR
        try:
            nanonets_engine = NanonetsOCREngine(self.config)
            await nanonets_engine.load_models()
            results["nanonets_ocr"] = {"loaded": True, "error": None}
        except Exception as e:
            results["nanonets_ocr"] = {"loaded": False, "error": str(e)}
        
        # Test NuExtract
        try:
            nuextract_engine = NuExtractEngine(self.config)
            await nuextract_engine.load_model()
            results["nuextract"] = {"loaded": True, "error": None}
        except Exception as e:
            results["nuextract"] = {"loaded": False, "error": str(e)}
        
        return results
    
    async def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic functionality of each model."""
        self.logger.info("Testing basic functionality...")
        
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        results = {}
        
        # Test MonkeyOCR functionality
        try:
            monkey_engine = MonkeyOCREngine(self.config)
            if not hasattr(monkey_engine, 'models_loaded') or not monkey_engine.models_loaded:
                await monkey_engine.load_models()
            
            ocr_result = await monkey_engine.extract_text(test_image)
            results["monkey_ocr"] = {
                "functional": True,
                "test_result": {
                    "text_length": len(ocr_result.text),
                    "confidence": ocr_result.confidence,
                    "processing_time": ocr_result.processing_time
                },
                "error": None
            }
        except Exception as e:
            results["monkey_ocr"] = {"functional": False, "error": str(e)}
        
        # Test Nanonets OCR functionality
        try:
            nanonets_engine = NanonetsOCREngine(self.config)
            if not hasattr(nanonets_engine, 'models_loaded') or not nanonets_engine.models_loaded:
                await nanonets_engine.load_models()
            
            ocr_result = await nanonets_engine.extract_text(test_image)
            results["nanonets_ocr"] = {
                "functional": True,
                "test_result": {
                    "text_length": len(ocr_result.text),
                    "confidence": ocr_result.confidence,
                    "processing_time": ocr_result.processing_time
                },
                "error": None
            }
        except Exception as e:
            results["nanonets_ocr"] = {"functional": False, "error": str(e)}
        
        # Test NuExtract functionality
        try:
            nuextract_engine = NuExtractEngine(self.config)
            if not hasattr(nuextract_engine, 'model') or nuextract_engine.model is None:
                await nuextract_engine.load_model()
            
            test_text = "Patient: John Doe, DOB: 1980-01-15, CPT: 99213"
            extraction_result = await nuextract_engine.extract_structured_data(
                test_text, template_name="patient_demographics"
            )
            results["nuextract"] = {
                "functional": True,
                "test_result": {
                    "extracted_fields": len(extraction_result),
                    "has_patient_name": "patient_name" in extraction_result
                },
                "error": None
            }
        except Exception as e:
            results["nuextract"] = {"functional": False, "error": str(e)}
        
        return results
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on diagnostic results."""
        return self.model_manager.get_recommended_actions()
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception:
            pass
        return total_size
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a human-readable summary of diagnostics."""
        print("\n" + "="*80)
        print("MEDICAL SUPERBILL EXTRACTION SYSTEM - MODEL DIAGNOSTICS")
        print("="*80)
        
        # System Status
        print("\n🖥️  SYSTEM STATUS:")
        system = results["system_status"]
        print(f"   Python: {system['python_version'].split()[0]}")
        print(f"   PyTorch: {system['torch_version']}")
        print(f"   CUDA Available: {'✓' if system['cuda_available'] else '✗'}")
        if system['cuda_available']:
            for i, gpu in enumerate(system['gpu_info']):
                print(f"   GPU {i}: {gpu['name']} ({gpu['memory_total']:.1f}GB total, {gpu['memory_free']:.1f}GB free)")
        print(f"   RAM: {system['ram']['available']:.1f}GB available / {system['ram']['total']:.1f}GB total")
        print(f"   CPU Cores: {system['cpu_cores']}")
        
        # Model Availability
        print("\n📦 MODEL AVAILABILITY:")
        availability = results["model_availability"]
        for model_key, info in availability.items():
            status = "✓" if info["available"] else "✗"
            size = f"({info['size_gb']:.1f}GB)" if info["available"] else ""
            print(f"   {model_key}: {status} {size}")
            if not info["available"]:
                print(f"      Expected at: {info['path']}")
        
        # Model Structure
        print("\n🔧 MODEL STRUCTURE:")
        structure = results["model_structure"]
        for model_key, info in structure.items():
            status = "✓" if info["valid"] else "✗"
            print(f"   {model_key}: {status}")
            if not info["valid"] and "missing_dirs" in info:
                print(f"      Missing directories: {info['missing_dirs']}")
            if not info["valid"] and "missing_files" in info:
                print(f"      Missing files: {info['missing_files']}")
        
        # Model Loading
        print("\n🚀 MODEL LOADING:")
        loading = results["model_loading"]
        for model_key, info in loading.items():
            status = "✓" if info["loaded"] else "✗"
            print(f"   {model_key}: {status}")
            if not info["loaded"]:
                print(f"      Error: {info['error']}")
        
        # Basic Functionality
        print("\n⚡ BASIC FUNCTIONALITY:")
        functionality = results["basic_functionality"]
        for model_key, info in functionality.items():
            status = "✓" if info["functional"] else "✗"
            print(f"   {model_key}: {status}")
            if info["functional"] and "test_result" in info:
                test = info["test_result"]
                if "text_length" in test:
                    print(f"      Extracted {test['text_length']} chars in {test['processing_time']:.2f}s")
                elif "extracted_fields" in test:
                    print(f"      Extracted {test['extracted_fields']} fields")
            elif not info["functional"]:
                print(f"      Error: {info['error']}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80)


async def main():
    """Run diagnostics."""
    try:
        diagnostics = ModelDiagnostics()
        results = await diagnostics.run_full_diagnostics()
        diagnostics.print_summary(results)
        
        # Return success/failure code
        all_loaded = all(
            info["loaded"] for info in results["model_loading"].values()
        )
        all_functional = all(
            info["functional"] for info in results["basic_functionality"].values()
        )
        
        if all_loaded and all_functional:
            print("\n🎉 All models are working correctly!")
            return 0
        else:
            print("\n⚠️  Some models have issues. Check the recommendations above.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
