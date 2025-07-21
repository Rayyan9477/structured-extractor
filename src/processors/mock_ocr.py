"""
Mock OCR Processor for Testing

Provides a mock OCR processor that returns placeholder text for testing
without requiring external services or APIs.
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional
from PIL import Image

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import OCRResult


class MockOCRProcessor:
    """
    Mock OCR processor that returns placeholder text for testing.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the mock OCR processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.models_loaded = False
        
        # Load configuration
        self.mock_config = self.config.get("ocr", {}).get("mock_ocr", {})
        self.delay = self.mock_config.get("delay", 0.5)  # Simulate processing time
        
        self.logger.info("Mock OCR processor initialized")
    
    async def load_models(self) -> None:
        """
        Simulate model loading.
        """
        # Simulate loading delay
        await asyncio.sleep(0.5)
        self.models_loaded = True
        self.logger.info("Mock OCR model loaded")
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """
        Extract text from an image using mock OCR.
        
        Args:
            image: PIL Image to process
            
        Returns:
            OCR result with mock text
        """
        if not self.models_loaded:
            await self.load_models()
        
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(self.delay)
        
        # Generate mock text based on image size
        width, height = image.size
        text_length = (width * height) // 10000  # Roughly scale text length to image size
        
        # Create dummy OCR text for testing
        mock_text = self._generate_mock_text(
            text_length, 
            has_patient_info=True,
            has_medical_codes=True
        )
        
        processing_time = time.time() - start_time
        
        self.logger.debug(f"Mock OCR extracted {len(mock_text)} characters in {processing_time:.2f}s")
        
        return OCRResult(
            text=mock_text,
            confidence=0.85,
            model_name="mock_ocr",
            processing_time=processing_time
        )
    
    async def extract_text_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Extract text from a batch of images.
        
        Args:
            images: A list of PIL Images to process
            
        Returns:
            A list of OCRResult objects
        """
        self.logger.info(f"Starting Mock OCR batch processing for {len(images)} images")
        
        results = []
        for i, image in enumerate(images):
            result = await self.extract_text(image)
            results.append(result)
            self.logger.debug(f"Processed image {i+1}/{len(images)}")
        
        self.logger.info(f"Completed Mock OCR batch processing for {len(images)} images")
        return results
    
    def _generate_mock_text(self, text_length: int, has_patient_info: bool = True, has_medical_codes: bool = True) -> str:
        """
        Generate mock OCR text.
        
        Args:
            text_length: Approximate length of text to generate
            has_patient_info: Include patient information
            has_medical_codes: Include medical codes
            
        Returns:
            Generated mock text
        """
        sections = []
        
        # Patient information section
        if has_patient_info:
            sections.append("""
PATIENT INFORMATION:
Name: Jane Smith
DOB: 04/15/1975
Gender: F
MRN: 12345678
Insurance: Blue Cross Blue Shield
Policy #: XYZ987654321
""")
        
        # Provider information
        sections.append("""
PROVIDER:
Dr. Michael Johnson
NPI: 1234567890
123 Medical Center Drive
Anytown, CA 90210
Phone: (555) 123-4567
""")
        
        # Date of service
        sections.append(f"""
DATE OF SERVICE: {random.choice(['04/10/2025', '06/15/2025', '08/22/2025'])}
""")
        
        # Diagnosis codes
        if has_medical_codes:
            sections.append("""
DIAGNOSIS CODES:
J45.909 - Unspecified asthma, uncomplicated
I10 - Essential (primary) hypertension
E11.9 - Type 2 diabetes mellitus without complications
""")
        
        # Procedure codes
        if has_medical_codes:
            sections.append("""
PROCEDURE CODES:
99214 - Office/outpatient visit, established patient - $120.00
93000 - Electrocardiogram, routine - $75.00
36415 - Collection of venous blood by venipuncture - $15.00
""")
        
        # Financial summary
        sections.append("""
FINANCIAL SUMMARY:
Total Charges: $210.00
Insurance Paid: $168.00
Patient Copay: $30.00
Balance Due: $12.00
""")
        
        # Notes section with Lorem Ipsum to pad the length
        lorem_ipsum = """
NOTES:
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor libero ac magna tincidunt, 
vel commodo nisl feugiat. Proin sodales metus non turpis sagittis, eu ultricies ligula pulvinar. 
Vivamus vel arcu nec dolor iaculis commodo. Fusce ullamcorper metus ut nisi facilisis, at ultrices 
augue faucibus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; 
Nulla facilisi. Curabitur ac vehicula tellus. Phasellus pharetra orci vel nisi congue, eget tincidunt 
odio vestibulum. Sed ullamcorper augue vel massa commodo, a commodo turpis varius.
"""
        
        sections.append(lorem_ipsum)
        
        # Combine sections and trim to desired length
        mock_text = "\n".join(sections)
        if len(mock_text) < text_length:
            # Pad with more lorem ipsum if needed
            mock_text += lorem_ipsum * ((text_length - len(mock_text)) // len(lorem_ipsum) + 1)
        
        return mock_text[:text_length] 