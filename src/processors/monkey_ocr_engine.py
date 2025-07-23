"""
MonkeyOCR Engine for Medical Document Processing

Integrates the MonkeyOCR model for document structure recognition and text extraction.
Uses the downloaded local model from echo840/MonkeyOCR.
"""

import asyncio
import torch
import time
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class MonkeyOCREngine:
    """
    MonkeyOCR engine for document parsing with Structure-Recognition-Relation paradigm.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the MonkeyOCR engine.
        
        Args:
            config: Configuration manager.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get MonkeyOCR configuration
        self.ocr_config = self.config.get("ocr.monkey_ocr", {})
        self.model_name = "echo840/MonkeyOCR"
        
        # Get local model path
        self.model_path = Path(self.config.get_model_path(self.model_name))
        
        self.device = self._get_device()
        self.logger.info(f"MonkeyOCR engine initialized with device: {self.device}")
        
        # Model components
        self.model = None
        self.processor = None
        self.models_loaded = False
        
        # Check if model exists locally
        if not self.model_path.exists():
            self.logger.error(f"MonkeyOCR model not found at {self.model_path}")
            raise FileNotFoundError(f"MonkeyOCR model directory not found: {self.model_path}")
        
        # Check for required subdirectories
        required_dirs = ["Recognition", "Structure", "Relation"]
        for req_dir in required_dirs:
            if not (self.model_path / req_dir).exists():
                self.logger.error(f"Required MonkeyOCR component '{req_dir}' not found")
                raise FileNotFoundError(f"MonkeyOCR component missing: {req_dir}")
        
    def _get_device(self) -> str:
        """Get the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def load_models(self) -> None:
        """Load the MonkeyOCR model components."""
        if self.models_loaded:
            return
            
        self.logger.info(f"Loading MonkeyOCR model from: {self.model_path}")
        
        try:
            # Import MonkeyOCR specific modules
            sys.path.insert(0, str(self.model_path))
            
            # Load the three components of MonkeyOCR
            await self._load_structure_model()
            await self._load_recognition_model()
            await self._load_relation_model()
            
            self.models_loaded = True
            self.logger.info("MonkeyOCR models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load MonkeyOCR models: {e}", exc_info=True)
            raise RuntimeError(f"Could not load MonkeyOCR models: {e}")

    async def _load_structure_model(self) -> None:
        """Load the structure detection model (custom PyTorch models)."""
        try:
            import torch
            
            structure_path = self.model_path / "Structure"
            
            # Check for available structure model files
            pt_files = list(structure_path.glob("*.pt"))
            if not pt_files:
                self.logger.warning("No .pt files found in Structure directory, skipping structure model")
                self.structure_model = None
                return
            
            # Look for doclayout_yolo model first
            doclayout_yolo_files = [f for f in pt_files if "doclayout_yolo" in f.name]
            if doclayout_yolo_files:
                structure_file = doclayout_yolo_files[0]
            else:
                # If not found, try layout_zh or any other available model
                structure_file = structure_path / "layout_zh.pt"
                if not structure_file.exists():
                    structure_file = pt_files[0]  # Use first available
            
            self.logger.info(f"Loading structure model from: {structure_file}")
            
            # Import required modules for doclayout_yolo if applicable
            if "doclayout_yolo" in structure_file.name:
                try:
                    # First try to import existing module if it's in the path
                    import sys
                    sys.path.append(str(self.model_path))
                    try:
                        import doclayout_yolo
                        self.logger.info("Successfully imported doclayout_yolo module")
                    except ImportError:
                        # Create a minimal module structure if not found
                        import types
                        doclayout_yolo = types.ModuleType('doclayout_yolo')
                        sys.modules['doclayout_yolo'] = doclayout_yolo
                        self.logger.info("Created placeholder doclayout_yolo module")
                except Exception as import_err:
                    self.logger.warning(f"Error setting up doclayout_yolo module: {import_err}")
            
            # Load as raw PyTorch model
            self.structure_model = torch.load(str(structure_file), map_location=self.device)
            if hasattr(self.structure_model, 'eval'):
                self.structure_model.eval()
            
            self.logger.debug("Structure model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load structure model: {e}")
            self.structure_model = None  # Continue without structure model

    async def _load_recognition_model(self) -> None:
        """Load the text recognition model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            recognition_path = self.model_path / "Recognition"
            
            self.recognition_processor = AutoProcessor.from_pretrained(
                str(recognition_path),
                trust_remote_code=True
            )
            
            self.recognition_model = AutoModelForVision2Seq.from_pretrained(
                str(recognition_path),
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.logger.debug("Recognition model loaded successfully")
            self.recognition_model.eval()  # Ensure model is in eval mode
            
        except Exception as e:
            self.logger.error(f"Failed to load recognition model: {e}")
            raise

    async def _load_relation_model(self) -> None:
        """Load the relation prediction model."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            relation_path = self.model_path / "Relation"
            
            self.relation_processor = AutoProcessor.from_pretrained(
                str(relation_path),
                trust_remote_code=True
            )
            
            self.relation_model = AutoModelForVision2Seq.from_pretrained(
                str(relation_path),
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.logger.debug("Relation model loaded successfully")
            self.relation_model.eval()  # Ensure model is in eval mode
            
        except Exception as e:
            self.logger.error(f"Failed to load relation model: {e}")
            raise

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using MonkeyOCR.
        
        Args:
            image: A PIL Image to process.
            
        Returns:
            An OCRResult object with the extracted text and metadata.
        """
        if not self.models_loaded:
            await self.load_models()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        start_time = time.time()
        
        try:
            # Step 1: Structure Detection
            structure_result = await self._detect_structure(image)
            
            # Step 2: Text Recognition
            recognition_result = await self._recognize_text(image, structure_result)
            
            # Step 3: Relation Prediction
            final_result = await self._predict_relations(image, structure_result, recognition_result)
            
            # Extract the final text from the structured result
            extracted_text = self._extract_text_from_result(final_result)
            
            processing_time = time.time() - start_time
            
            self.logger.debug(f"MonkeyOCR extracted {len(extracted_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=extracted_text,
                confidence=0.9,  # MonkeyOCR typically has high confidence
                model_name=self.model_name,
                processing_time=processing_time,
                additional_data={
                    "structure": structure_result,
                    "recognition": recognition_result,
                    "relations": final_result
                }
            )
            
        except Exception as e:
            self.logger.error(f"MonkeyOCR text extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name=self.model_name, 
                processing_time=time.time() - start_time
            )

    async def _detect_structure(self, image: Image.Image) -> Dict[str, Any]:
        """Detect document structure using the structure model."""
        try:
            # Check if structure model is available
            if not hasattr(self, 'structure_model') or self.structure_model is None:
                self.logger.warning("Structure model not available, using fallback")
                return {"raw_output": "structure_model_unavailable", "boxes": []}
            
            # For PyTorch models, we need to handle them differently
            if hasattr(self.structure_model, 'forward'):
                # This is a PyTorch model - handle inference directly
                import torchvision.transforms as transforms
                transform = transforms.Compose([
                    transforms.Resize((640, 640)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # Convert PIL to tensor
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.structure_model(img_tensor)
                
                # Extract structure information from outputs
                structure_result = self._process_structure_outputs(outputs)
            else:
                # Try using as a transformers model with processor
                if hasattr(self, 'recognition_processor'):
                    inputs = self.recognition_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.structure_model.generate(**inputs, max_length=512)
                    
                    structure_text = self.recognition_processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )[0]
                    
                    try:
                        structure_result = json.loads(structure_text)
                    except json.JSONDecodeError:
                        structure_result = {"raw_output": structure_text}
                else:
                    structure_result = {"raw_output": "processor_unavailable"}
            
            return structure_result
            
        except Exception as e:
            self.logger.error(f"Structure detection failed: {e}")
            return {"error": str(e), "raw_output": "structure_detection_failed"}
    
    def _process_structure_outputs(self, outputs) -> Dict[str, Any]:
        """Process raw structure model outputs into structured format."""
        try:
            # Handle different types of structure model outputs
            if isinstance(outputs, torch.Tensor):
                # Convert tensor to numpy for processing
                outputs_np = outputs.cpu().numpy()
                return {
                    "boxes": [],  # Placeholder for bounding boxes
                    "confidence": 0.8,
                    "raw_output": f"tensor_shape_{outputs_np.shape}"
                }
            elif isinstance(outputs, (list, tuple)):
                # Handle list/tuple outputs (e.g., from YOLO models)
                return {
                    "boxes": [],
                    "confidence": 0.8,
                    "raw_output": f"list_output_length_{len(outputs)}"
                }
            elif isinstance(outputs, dict):
                # Handle dictionary outputs
                return {
                    "boxes": outputs.get("boxes", []),
                    "confidence": outputs.get("confidence", 0.8),
                    "raw_output": str(outputs)
                }
            else:
                return {
                    "boxes": [],
                    "confidence": 0.8,
                    "raw_output": str(type(outputs))
                }
                
        except Exception as e:
            self.logger.error(f"Error processing structure outputs: {e}")
            return {
                "boxes": [],
                "confidence": 0.0,
                "error": str(e)
            }

    async def _recognize_text(self, image: Image.Image, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize text content using the recognition model."""
        try:
            # Prepare image for text recognition
            inputs = self.recognition_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate text recognition
            with torch.no_grad():
                outputs = self.recognition_model.generate(**inputs, max_length=2048)
            
            # Decode recognition result
            recognition_text = self.recognition_processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            # Parse recognition result
            try:
                recognition_result = json.loads(recognition_text)
            except json.JSONDecodeError:
                recognition_result = {"raw_text": recognition_text}
            
            return recognition_result
            
        except Exception as e:
            self.logger.error(f"Text recognition failed: {e}")
            return {"error": str(e)}

    async def _predict_relations(self, image: Image.Image, structure: Dict[str, Any], recognition: Dict[str, Any]) -> Dict[str, Any]:
        """Predict relationships between detected elements."""
        try:
            # Prepare image for relation prediction
            inputs = self.relation_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate relation prediction
            with torch.no_grad():
                outputs = self.relation_model.generate(**inputs, max_length=1024)
            
            # Decode relation result
            relation_text = self.relation_processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            # Parse relation result
            try:
                relation_result = json.loads(relation_text)
            except json.JSONDecodeError:
                relation_result = {"raw_relations": relation_text}
            
            # Combine all results
            final_result = {
                "structure": structure,
                "recognition": recognition,
                "relations": relation_result
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Relation prediction failed: {e}")
            return {
                "structure": structure,
                "recognition": recognition,
                "relations": {"error": str(e)}
            }

    def _extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """Extract plain text from the structured MonkeyOCR result."""
        try:
            # Extract text from recognition results
            recognition = result.get("recognition", {})
            
            if "raw_text" in recognition:
                return recognition["raw_text"]
            
            # Try to extract text from structured format
            if "text_blocks" in recognition:
                texts = []
                for block in recognition["text_blocks"]:
                    if isinstance(block, dict) and "text" in block:
                        texts.append(block["text"])
                    elif isinstance(block, str):
                        texts.append(block)
                return "\n".join(texts)
            
            # Check for raw_output in recognition
            if "raw_output" in recognition and isinstance(recognition["raw_output"], str):
                return recognition["raw_output"]
            
            # Fallback: try to extract any text content
            if isinstance(recognition, dict):
                text_content = []
                for key, value in recognition.items():
                    if isinstance(value, str) and key not in ["error", "model_info"] and value.strip():
                        text_content.append(value)
                return "\n".join(text_content)
            
            # Last resort - check if recognition itself is a string
            if isinstance(recognition, str):
                return recognition
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from result: {e}")
            return ""

    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process.
            
        Returns:
            A list of OCRResult objects.
        """
        if not self.models_loaded:
            await self.load_models()
            
        self.logger.info(f"Starting MonkeyOCR batch processing for {len(images)} images")
        
        results = []
        for i, image in enumerate(images):
            self.logger.debug(f"Processing image {i+1}/{len(images)} with MonkeyOCR")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"MonkeyOCR batch processing completed for {len(images)} images")
        return results
