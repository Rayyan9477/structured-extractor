"""
Nanonets OCR Engine for Medical Document Processing

Integrates the Nanonets OCR model for advanced document understanding and text extraction.
Uses the downloaded local model from nanonets/Nanonets-OCR-s.
"""

import asyncio
import torch
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import base64
import io
from pathlib import Path

from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult, FieldConfidence


class NanonetsOCREngine:
    """
    Nanonets OCR engine for intelligent document understanding and markdown conversion.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the Nanonets OCR engine.
        
        Args:
            config: Configuration manager.
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get Nanonets configuration
        self.ocr_config = self.config.get("ocr.nanonets_ocr", {})
        self.model_name = "nanonets/Nanonets-OCR-s"
        
        # Get local model path
        self.model_path = Path(self.config.get_model_path(self.model_name))
        
        self.device = self._get_device()
        self.max_new_tokens = 15000  # Large context for complex documents
        
        self.logger.info(f"Nanonets OCR engine initialized with device: {self.device}")
        
        # Model components
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.models_loaded = False
        
        # Check if model exists locally
        if not self.model_path.exists():
            self.logger.error(f"Nanonets OCR model not found at {self.model_path}")
            raise FileNotFoundError(f"Nanonets OCR model directory not found: {self.model_path}")
        
    def _get_device(self) -> str:
        """Get the optimal device for model inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def load_models(self) -> None:
        """Load the Nanonets OCR model and processor with comprehensive error handling."""
        if self.models_loaded:
            return
            
        self.logger.info(f"Loading Nanonets OCR model from: {self.model_path}")
        
        # Validate model files exist
        if not self._validate_model_files():
            raise RuntimeError(f"Nanonets OCR model files not found or incomplete at {self.model_path}")
        
        try:
            # Load the model for Vision2Seq tasks with proper error handling
            self.logger.info("Loading Nanonets OCR model...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True if torch.cuda.is_available() else False
            )
            
            # Don't move model when using device_map="auto" - it handles device placement automatically
            if not torch.cuda.is_available():
                self.model.to(self.device)
            
            self.model.eval()
            self.logger.info("Nanonets OCR model loaded successfully")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            self.logger.info("Tokenizer loaded successfully")

            # Load processor with comprehensive error handling
            self.logger.info("Loading processor...")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True
                )
                self.logger.info("Processor loaded successfully")
            except Exception as processor_error:
                self.logger.warning(f"AutoProcessor loading failed: {processor_error}")
                # Try alternative processor loading
                try:
                    from transformers import Qwen2VLProcessor
                    self.processor = Qwen2VLProcessor.from_pretrained(
                        str(self.model_path),
                        trust_remote_code=True
                    )
                    self.logger.info("Alternative processor loaded successfully")
                except Exception as alt_error:
                    self.logger.error(f"Alternative processor loading also failed: {alt_error}")
                    raise RuntimeError(f"Failed to load processor: {processor_error}")
            
            self.models_loaded = True
            self.logger.info("Nanonets OCR model and components loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Nanonets OCR model: {e}", exc_info=True)
            # Clean up any partially loaded components
            self.model = None
            self.tokenizer = None
            self.processor = None
            self.models_loaded = False
            raise RuntimeError(f"Could not load Nanonets OCR model: {e}")

    def _validate_model_files(self) -> bool:
        """Validate that all required model files exist."""
        required_files = [
            "config.json",
            "tokenizer.json",
            "vocab.json",
            "model.safetensors.index.json"
        ]
        
        for file_name in required_files:
            file_path = self.model_path / file_name
            if not file_path.exists():
                self.logger.error(f"Required model file not found: {file_path}")
                return False
        
        # Check for model weights files
        model_files = list(self.model_path.glob("model-*.safetensors"))
        if not model_files:
            self.logger.error(f"No model weight files found in {self.model_path}")
            return False
        
        self.logger.info(f"Model validation passed. Found {len(model_files)} weight files.")
        return True

    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from a single image using Nanonets OCR.
        
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
            # Prepare the prompt for medical document processing
            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes. Focus on medical terminology, codes, patient information, and billing details."""
            
            # Prepare messages in the correct format
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in medical document processing."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]
                },
            ]
            
            # Apply chat template with error handling
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except AttributeError as e:
                # Handle the case where to_dict() is not available
                self.logger.warning(f"apply_chat_template failed: {e}, using fallback method")
                
                # Try direct attribute access or create a dictionary manually
                try:
                    # Convert message objects to dict format manually if to_dict() is missing
                    processed_messages = []
                    for msg in messages:
                        if "content" in msg and isinstance(msg["content"], list):
                            new_content = []
                            for item in msg["content"]:
                                if item.get("type") == "image":
                                    # Handle image specially - convert to tensor format the processor expects
                                    img = item.get("image")
                                    if isinstance(img, Image.Image):
                                        new_content.append({"type": "image", "image": img})
                                else:
                                    new_content.append(item)
                            msg = {**msg, "content": new_content}
                        processed_messages.append(msg)
                    
                    # Try using tokenizer's chat template directly
                    text = self.tokenizer.apply_chat_template(
                        processed_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as direct_err:
                    # Last resort fallback - hardcoded template
                    self.logger.warning(f"Direct attribute access failed: {direct_err}, using hardcoded template")
                    text = f"<|im_start|>system\nYou are a helpful assistant specialized in medical document processing.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Process image and text
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            
            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the generated tokens
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            # Decode the output
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            # Clean the extracted text
            cleaned_text = self._clean_text(output_text)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence based on output length and structure
            confidence = self._calculate_confidence(cleaned_text)
            
            self.logger.debug(f"Nanonets extracted {len(cleaned_text)} characters in {processing_time:.2f}s")
            
            return OCRResult(
                text=cleaned_text,
                confidence=confidence,
                model_name=self.model_name,
                processing_time=processing_time,
                additional_data={
                    "raw_output": output_text,
                    "prompt_used": prompt
                }
            )
            
        except Exception as e:
            self.logger.error(f"Nanonets OCR text extraction failed: {e}", exc_info=True)
            return OCRResult(
                text="", 
                confidence=0.0, 
                model_name=self.model_name, 
                processing_time=time.time() - start_time
            )

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text from Nanonets OCR.
        
        Args:
            text: The raw extracted text.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Remove any artifacts from the generation process
        text = text.strip()
        
        # Preserve structured elements like tables, checkboxes, etc.
        # The Nanonets model already formats these properly
        
        return text

    def _calculate_confidence(self, text: str) -> float:
        """
        Calculate confidence score based on text characteristics.
        
        Args:
            text: Extracted text
            
        Returns:
            Confidence score between 0 and 1
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        confidence = 0.7  # Base confidence
        
        # Increase confidence for medical document indicators
        medical_indicators = [
            "patient", "diagnosis", "cpt", "icd", "procedure", 
            "billing", "insurance", "provider", "npi", "date of service"
        ]
        
        text_lower = text.lower()
        found_indicators = sum(1 for indicator in medical_indicators if indicator in text_lower)
        
        # Boost confidence based on medical content
        if found_indicators > 0:
            confidence += min(0.2, found_indicators * 0.03)
        
        # Check for structured elements
        if any(marker in text for marker in ["<table>", "☐", "☑", "<img>", "<watermark>"]):
            confidence += 0.05
        
        # Penalize very short extractions
        if len(text.strip()) < 50:
            confidence *= 0.8
        
        return min(0.95, confidence)

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
            
        self.logger.info(f"Starting Nanonets OCR batch processing for {len(images)} images")
        
        # Process images individually for better memory management
        # Nanonets OCR handles complex documents better with individual processing
        results = []
        for i, image in enumerate(images):
            self.logger.debug(f"Processing image {i+1}/{len(images)} with Nanonets OCR")
            result = await self.extract_text(image)
            results.append(result)
        
        self.logger.info(f"Nanonets OCR batch processing completed for {len(images)} images")
        return results

    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return [
            "Medical superbills", "Insurance forms", "Clinical notes",
            "Lab reports", "Prescription forms", "Billing statements",
            "Patient records", "Referral letters", "Discharge summaries"
        ]

    def get_feature_capabilities(self) -> Dict[str, bool]:
        """Get feature capabilities of the Nanonets OCR engine."""
        return {
            "table_extraction": True,
            "checkbox_detection": True,
            "signature_detection": True,
            "watermark_extraction": True,
            "image_description": True,
            "latex_equations": True,
            "html_tables": True,
            "structured_output": True,
            "medical_terminology": True
        }
