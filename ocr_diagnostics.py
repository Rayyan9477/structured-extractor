#!/usr/bin/env python3
"""
Diagnostic Script for Superbill OCR Issues

This script investigates why the OCR models are extracting minimal text from the superbills
and provides detailed diagnostics to understand and fix the issue.
"""

import asyncio
import sys
from pathlib import Path
import time
from typing import List, Dict, Any
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.processors.document_processor import DocumentProcessor
from src.processors.ocr_engine import OCREngine
from src.core.config_manager import ConfigManager
from src.core.logger import setup_logger, get_logger


class OCRDiagnostics:
    """Comprehensive OCR diagnostics for superbill processing."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = get_logger(__name__)
        self.doc_processor = DocumentProcessor(self.config)
        
        # Create diagnostics output directory
        self.diagnostics_dir = Path("diagnostics")
        self.diagnostics_dir.mkdir(exist_ok=True)
    
    async def diagnose_superbill_ocr(self, pdf_path: Path):
        """Perform comprehensive OCR diagnostics on a superbill."""
        self.logger.info(f"🔍 Starting OCR diagnostics for: {pdf_path.name}")
        
        diagnostics = {
            'file_name': pdf_path.name,
            'file_size': pdf_path.stat().st_size,
            'timestamp': time.time(),
            'document_processing': {},
            'ocr_results': {},
            'issues_found': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Document processing diagnostics
            self.logger.info("📄 Analyzing document processing...")
            doc_results = await self._diagnose_document_processing(pdf_path)
            diagnostics['document_processing'] = doc_results
            
            if not doc_results['images']:
                diagnostics['issues_found'].append("No images extracted from PDF")
                diagnostics['recommendations'].append("Check PDF file integrity and format")
                return diagnostics
            
            # Step 2: OCR model diagnostics
            self.logger.info("🤖 Testing OCR models...")
            ocr_models = [
                'microsoft/trocr-base-handwritten',
                'microsoft/trocr-large-printed',
                'echo840/MonkeyOCR',
                'nanonets/Nanonets-OCR-s'
            ]
            
            for model_name in ocr_models:
                try:
                    self.logger.info(f"Testing {model_name}...")
                    ocr_result = await self._test_ocr_model(model_name, doc_results['images'][:1])  # Test first page only
                    diagnostics['ocr_results'][model_name] = ocr_result
                except Exception as e:
                    self.logger.error(f"Failed to test {model_name}: {e}")
                    diagnostics['ocr_results'][model_name] = {'error': str(e)}
            
            # Step 3: Analysis and recommendations
            diagnostics = self._analyze_results(diagnostics)
            
        except Exception as e:
            self.logger.error(f"Diagnostics failed: {e}")
            diagnostics['error'] = str(e)
        
        # Save diagnostics
        diag_file = self.diagnostics_dir / f"{pdf_path.stem}_diagnostics.json"
        with open(diag_file, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"💾 Diagnostics saved to: {diag_file}")
        return diagnostics
    
    async def _diagnose_document_processing(self, pdf_path: Path) -> Dict[str, Any]:
        """Diagnose document processing step."""
        try:
            start_time = time.time()
            images = await self.doc_processor.process_document(str(pdf_path))
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'processing_time': processing_time,
                'total_pages': len(images),
                'images': [],
                'page_info': []
            }
            
            # Analyze each page
            for i, image in enumerate(images):
                page_info = {
                    'page_number': i + 1,
                    'image_size': image.size,
                    'image_mode': image.mode,
                    'image_format': getattr(image, 'format', 'Unknown')
                }
                
                # Save sample image for manual inspection
                sample_path = self.diagnostics_dir / f"{pdf_path.stem}_page_{i+1}_sample.png"
                image.save(sample_path)
                page_info['sample_saved'] = str(sample_path)
                
                result['page_info'].append(page_info)
                result['images'].append(image)
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'images': [],
                'page_info': []
            }
    
    async def _test_ocr_model(self, model_name: str, images: List) -> Dict[str, Any]:
        """Test a specific OCR model."""
        # Update configuration
        self.config.update_config("ocr.model_name", model_name)
        
        # Create new OCR engine with updated config
        ocr_engine = OCREngine(self.config)
        
        try:
            start_time = time.time()
            await ocr_engine.load_models()
            load_time = time.time() - start_time
            
            # Test OCR on first image
            if images:
                ocr_start = time.time()
                ocr_result = await ocr_engine.extract_text(images[0])
                ocr_time = time.time() - ocr_start
                
                # Save extracted text
                text_file = self.diagnostics_dir / f"extracted_text_{model_name.replace('/', '_')}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Model: {model_name}\n")
                    f.write(f"Confidence: {ocr_result.confidence}\n")
                    f.write(f"Processing time: {ocr_result.processing_time}\n")
                    f.write(f"Text length: {len(ocr_result.text)}\n")
                    f.write("\n" + "="*60 + "\n")
                    f.write("EXTRACTED TEXT:\n")
                    f.write("="*60 + "\n")
                    f.write(ocr_result.text)
                
                return {
                    'success': True,
                    'model_load_time': load_time,
                    'ocr_processing_time': ocr_time,
                    'confidence': ocr_result.confidence,
                    'text_length': len(ocr_result.text),
                    'text_preview': ocr_result.text[:200],
                    'full_text': ocr_result.text,
                    'text_file_saved': str(text_file)
                }
            else:
                return {'success': False, 'error': 'No images to process'}
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_results(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results and provide recommendations."""
        issues = diagnostics['issues_found']
        recommendations = diagnostics['recommendations']
        
        # Check document processing
        doc_processing = diagnostics['document_processing']
        if not doc_processing.get('success'):
            issues.append("Document processing failed")
            recommendations.append("Check PDF file format and corruption")
        elif doc_processing.get('total_pages', 0) == 0:
            issues.append("No pages extracted from PDF")
            recommendations.append("PDF may be corrupted or password protected")
        
        # Check OCR results
        ocr_results = diagnostics['ocr_results']
        successful_models = []
        text_lengths = []
        
        for model_name, result in ocr_results.items():
            if result.get('success'):
                successful_models.append(model_name)
                text_lengths.append(result.get('text_length', 0))
        
        if not successful_models:
            issues.append("All OCR models failed")
            recommendations.append("Check model availability and CUDA/CPU compatibility")
        else:
            avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
            
            if avg_text_length < 100:
                issues.append("Very short text extraction (< 100 characters)")
                recommendations.extend([
                    "Images may be low quality or resolution",
                    "Try image preprocessing enhancement",
                    "Check if document contains actual text (not just images)",
                    "Consider using different OCR models or preprocessing"
                ])
            elif avg_text_length < 500:
                issues.append("Short text extraction (< 500 characters)")
                recommendations.append("Consider image quality improvements")
        
        # Performance analysis
        if successful_models:
            fastest_model = min(successful_models, 
                              key=lambda m: ocr_results[m].get('ocr_processing_time', float('inf')))
            most_confident = max(successful_models,
                               key=lambda m: ocr_results[m].get('confidence', 0))
            longest_text = max(successful_models,
                             key=lambda m: ocr_results[m].get('text_length', 0))
            
            diagnostics['recommendations'].extend([
                f"Fastest model: {fastest_model}",
                f"Most confident model: {most_confident}",
                f"Most text extracted: {longest_text}"
            ])
        
        return diagnostics
    
    def print_diagnostics_summary(self, diagnostics: Dict[str, Any]):
        """Print a summary of diagnostics results."""
        print("\n" + "="*80)
        print(f"🔍 OCR DIAGNOSTICS SUMMARY: {diagnostics['file_name']}")
        print("="*80)
        
        # Document processing
        doc_proc = diagnostics['document_processing']
        print(f"\n📄 DOCUMENT PROCESSING:")
        print(f"   Success: {doc_proc.get('success', False)}")
        print(f"   Pages extracted: {doc_proc.get('total_pages', 0)}")
        print(f"   Processing time: {doc_proc.get('processing_time', 0):.2f}s")
        
        if doc_proc.get('page_info'):
            print(f"   Page details:")
            for page in doc_proc['page_info']:
                print(f"      Page {page['page_number']}: {page['image_size']} ({page['image_mode']})")
        
        # OCR results
        ocr_results = diagnostics['ocr_results']
        print(f"\n🤖 OCR MODEL RESULTS:")
        
        for model_name, result in ocr_results.items():
            if result.get('success'):
                print(f"   ✅ {model_name}:")
                print(f"      Text length: {result['text_length']} characters")
                print(f"      Confidence: {result['confidence']:.3f}")
                print(f"      Processing time: {result['ocr_processing_time']:.3f}s")
                print(f"      Preview: {result['text_preview'][:100]}...")
            else:
                print(f"   ❌ {model_name}: {result.get('error', 'Unknown error')}")
        
        # Issues and recommendations
        if diagnostics['issues_found']:
            print(f"\n⚠️  ISSUES FOUND:")
            for issue in diagnostics['issues_found']:
                print(f"   • {issue}")
        
        if diagnostics['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in diagnostics['recommendations']:
                print(f"   • {rec}")
        
        print("="*80)


async def main():
    """Run OCR diagnostics on sample superbills."""
    # Setup logging
    setup_logger(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting OCR Diagnostics")
    
    # Initialize diagnostics
    diagnostics_tool = OCRDiagnostics()
    
    # Get superbill files
    superbill_dir = Path("superbills")
    pdf_files = list(superbill_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error("❌ No PDF files found in superbills directory")
        return
    
    logger.info(f"📁 Found {len(pdf_files)} superbill files")
    
    # Run diagnostics on each file
    all_diagnostics = []
    
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 DIAGNOSING: {pdf_file.name}")
        logger.info(f"{'='*60}")
        
        diagnostics = await diagnostics_tool.diagnose_superbill_ocr(pdf_file)
        all_diagnostics.append(diagnostics)
        
        # Print summary
        diagnostics_tool.print_diagnostics_summary(diagnostics)
    
    # Save consolidated report
    consolidated_path = Path("diagnostics") / "consolidated_diagnostics_report.json"
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(all_diagnostics),
            'timestamp': time.time(),
            'individual_diagnostics': all_diagnostics
        }, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n📊 Consolidated diagnostics saved to: {consolidated_path}")
    logger.info("✅ OCR Diagnostics completed!")


if __name__ == "__main__":
    asyncio.run(main())
