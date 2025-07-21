#!/usr/bin/env python3
"""
Debug OCR Output Script

This script helps debug what the OCR models are actually extracting from the superbills.
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger, get_logger
from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_engine import OCREngine


async def debug_ocr_extraction():
    """Debug OCR extraction for the sample superbills."""
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    # Initialize components
    config = ConfigManager()
    doc_processor = DocumentProcessor(config)
    ocr_engine = OCREngine(config)
    
    # Test different OCR models
    ocr_models = [
        "microsoft/trocr-base-handwritten",
        "microsoft/trocr-large-printed"
    ]
    
    # Get superbill files
    superbill_dir = Path("superbills")
    pdf_files = list(superbill_dir.glob("*.pdf"))[:1]  # Just test one file first
    
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_file.name}")
        logger.info(f"{'='*60}")
        
        # Process document to images
        images = await doc_processor.process_document(str(pdf_file))
        logger.info(f"Extracted {len(images)} pages")
        
        for model_name in ocr_models:
            logger.info(f"\n--- Testing OCR Model: {model_name} ---")
            
            # Update OCR model
            config.update_config("ocr.model_name", model_name)
            
            # Load model
            await ocr_engine.load_models()
            
            # Extract text from each page
            for i, image in enumerate(images):
                logger.info(f"\nPage {i + 1}:")
                
                ocr_result = await ocr_engine.extract_text(image)
                
                logger.info(f"Confidence: {ocr_result.confidence}")
                logger.info(f"Text length: {len(ocr_result.text)}")
                logger.info(f"Extracted text (first 500 chars):")
                logger.info("-" * 40)
                logger.info(ocr_result.text[:500])
                logger.info("-" * 40)
                
                # Save full text to file
                output_file = Path(f"debug_ocr_{pdf_file.stem}_page{i+1}_{model_name.replace('/', '_')}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Model: {model_name}\n")
                    f.write(f"Confidence: {ocr_result.confidence}\n")
                    f.write(f"Processing time: {ocr_result.processing_time}\n")
                    f.write(f"Text length: {len(ocr_result.text)}\n")
                    f.write("\n" + "="*60 + "\n")
                    f.write("EXTRACTED TEXT:\n")
                    f.write("="*60 + "\n")
                    f.write(ocr_result.text)
                
                logger.info(f"Saved full text to: {output_file}")


if __name__ == "__main__":
    asyncio.run(debug_ocr_extraction())
