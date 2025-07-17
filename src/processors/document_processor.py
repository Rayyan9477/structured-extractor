"""
Document Processing Pipeline for Medical Superbill Extraction

Handles PDF to image conversion, image preprocessing, page segmentation,
and document orientation detection.
"""

import asyncio
import io
import tempfile
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pdf2image
from pdf2image import convert_from_path, convert_from_bytes

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize image preprocessor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get preprocessing settings
        self.settings = config.get("document_processing.image_preprocessing", {})
        self.resize_factor = self.settings.get("resize_factor", 1.0)
        self.denoise = self.settings.get("denoise", True)
        self.enhance_contrast = self.settings.get("enhance_contrast", True)
        self.binarize = self.settings.get("binarize", False)
        self.rotation_correction = self.settings.get("rotation_correction", True)
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing operations to an image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        self.logger.debug("Starting image preprocessing")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if specified
        if self.resize_factor != 1.0:
            new_size = (
                int(image.width * self.resize_factor),
                int(image.height * self.resize_factor)
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image to {new_size}")
        
        # Convert to OpenCV format for advanced processing
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Rotation correction
        if self.rotation_correction:
            cv_image = self._correct_rotation(cv_image)
        
        # Denoising
        if self.denoise:
            cv_image = self._denoise_image(cv_image)
        
        # Contrast enhancement
        if self.enhance_contrast:
            cv_image = self._enhance_contrast(cv_image)
        
        # Binarization (optional)
        if self.binarize:
            cv_image = self._binarize_image(cv_image)
        
        # Convert back to PIL
        processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        self.logger.debug("Image preprocessing completed")
        return processed_image
    
    def _correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image rotation.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Rotation-corrected image
        """
        try:
            # Convert to grayscale for rotation detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Hough lines to detect text orientation
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:20]:  # Use first 20 lines
                    angle = theta * 180 / np.pi
                    # Convert to rotation angle
                    if angle > 90:
                        angle = angle - 180
                    angles.append(angle)
                
                # Find median angle
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant
                if abs(median_angle) > 1:
                    self.logger.debug(f"Correcting rotation by {median_angle:.2f} degrees")
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                         borderMode=cv2.BORDER_REPLICATE)
            
        except Exception as e:
            self.logger.warning(f"Rotation correction failed: {e}")
        
        return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Denoised image
        """
        # Use Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge channels back
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to binary (black and white).
        
        Args:
            image: OpenCV image array
            
        Returns:
            Binary image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to 3-channel
        binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return binary_3ch


class PageSegmenter:
    """Handles page segmentation for multi-patient documents."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize page segmenter.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Get patient detection settings
        self.patient_config = config.get("patient_detection", {})
        self.max_patients = self.patient_config.get("max_patients_per_document", 10)
        self.separation_keywords = self.patient_config.get("separation_keywords", [])
    
    def segment_page(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Segment page into potential patient sections.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of segments with bounding boxes and metadata
        """
        self.logger.debug("Starting page segmentation")
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        segments = []
        
        # Method 1: Detect horizontal lines that might separate patients
        horizontal_segments = self._detect_horizontal_separators(gray)
        
        # Method 2: Detect text blocks that might indicate patient sections
        text_block_segments = self._detect_text_blocks(gray)
        
        # Method 3: Detect form-like structures
        form_segments = self._detect_form_structures(gray)
        
        # Combine and validate segments
        all_segments = horizontal_segments + text_block_segments + form_segments
        validated_segments = self._validate_segments(all_segments, image.size)
        
        # If no segments found, treat entire page as single patient
        if not validated_segments:
            validated_segments = [{
                'bbox': (0, 0, image.width, image.height),
                'confidence': 1.0,
                'method': 'full_page',
                'patient_index': 0
            }]
        
        self.logger.debug(f"Found {len(validated_segments)} segments")
        return validated_segments
    
    def _detect_horizontal_separators(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect horizontal lines that separate patient sections."""
        segments = []
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for significant horizontal lines
        height, width = gray_image.shape
        significant_lines = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > width * 0.3 and h < 10:  # Wide but thin lines
                significant_lines.append(y)
        
        # Create segments between lines
        significant_lines = sorted(set(significant_lines))
        
        if significant_lines:
            # Add top and bottom boundaries
            boundaries = [0] + significant_lines + [height]
            
            for i in range(len(boundaries) - 1):
                top = boundaries[i]
                bottom = boundaries[i + 1]
                
                if bottom - top > 50:  # Minimum segment height
                    segments.append({
                        'bbox': (0, top, width, bottom),
                        'confidence': 0.7,
                        'method': 'horizontal_separator',
                        'patient_index': i
                    })
        
        return segments
    
    def _detect_text_blocks(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect distinct text blocks that might represent patients."""
        segments = []
        
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to connect text
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        height, width = gray_image.shape
        min_area = (width * height) * 0.02  # At least 2% of page
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append((x, y, w, h))
        
        # Sort by vertical position
        text_regions.sort(key=lambda r: r[1])
        
        # Create segments
        for i, (x, y, w, h) in enumerate(text_regions):
            segments.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 0.6,
                'method': 'text_block',
                'patient_index': i
            })
        
        return segments
    
    def _detect_form_structures(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect form-like structures that might indicate patient sections."""
        segments = []
        
        # Detect rectangles and boxes
        contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray_image.shape
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for rectangular shapes
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by size (should be substantial but not entire page)
                if area > (width * height * 0.05) and area < (width * height * 0.8):
                    segments.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.5,
                        'method': 'form_structure',
                        'patient_index': len(segments)
                    })
        
        return segments
    
    def _validate_segments(self, segments: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Validate and filter segments."""
        if not segments:
            return []
        
        width, height = image_size
        validated = []
        
        # Remove overlapping segments (keep highest confidence)
        segments.sort(key=lambda s: s['confidence'], reverse=True)
        
        for segment in segments:
            bbox = segment['bbox']
            
            # Check if segment is reasonable size
            seg_width = bbox[2] - bbox[0]
            seg_height = bbox[3] - bbox[1]
            
            if seg_width < width * 0.1 or seg_height < height * 0.05:
                continue  # Too small
            
            # Check for significant overlap with existing segments
            overlap = False
            for existing in validated:
                if self._calculate_overlap(bbox, existing['bbox']) > 0.5:
                    overlap = True
                    break
            
            if not overlap:
                validated.append(segment)
        
        # Limit to max patients
        validated = validated[:self.max_patients]
        
        # Re-index patients
        for i, segment in enumerate(validated):
            segment['patient_index'] = i
        
        return validated
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize document processor.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize sub-processors
        self.preprocessor = ImagePreprocessor(config)
        self.segmenter = PageSegmenter(config)
        
        # Get PDF processing settings
        self.pdf_settings = config.get("document_processing.pdf", {})
        self.dpi = self.pdf_settings.get("dpi", 300)
        self.format = self.pdf_settings.get("format", "RGB")
        self.first_page = self.pdf_settings.get("first_page")
        self.last_page = self.pdf_settings.get("last_page")
    
    async def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract images with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed images with metadata
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = await self._pdf_to_images(pdf_path)
            
            # Process each image
            processed_images = []
            for i, image in enumerate(images):
                self.logger.debug(f"Processing page {i + 1}")
                
                # Preprocess image
                preprocessed = self.preprocessor.preprocess_image(image)
                
                # Segment page
                segments = self.segmenter.segment_page(preprocessed)
                
                # Store processed page data
                page_data = {
                    'page_number': i + 1,
                    'image': preprocessed,
                    'segments': segments,
                    'original_size': image.size,
                    'processed_size': preprocessed.size
                }
                processed_images.append(page_data)
            
            self.logger.info(f"Successfully processed {len(processed_images)} pages")
            return processed_images
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    async def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """
        Convert PDF to list of PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of PIL Images
        """
        try:
            # Use pdf2image for conversion
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.format,
                first_page=self.first_page,
                last_page=self.last_page,
                thread_count=4
            )
            
            self.logger.debug(f"Converted PDF to {len(images)} images")
            return images
            
        except Exception as e:
            self.logger.warning(f"pdf2image failed, trying PyMuPDF: {e}")
            
            # Fallback to PyMuPDF
            try:
                doc = fitz.open(pdf_path)
                images = []
                
                start_page = (self.first_page - 1) if self.first_page else 0
                end_page = self.last_page if self.last_page else doc.page_count
                
                for page_num in range(start_page, min(end_page, doc.page_count)):
                    page = doc[page_num]
                    
                    # Render page to image
                    mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("ppm")
                    image = Image.open(io.BytesIO(img_data))
                    
                    if image.mode != self.format:
                        image = image.convert(self.format)
                    
                    images.append(image)
                
                doc.close()
                self.logger.debug(f"PyMuPDF converted PDF to {len(images)} images")
                return images
                
            except Exception as e2:
                self.logger.error(f"Both PDF conversion methods failed: {e2}")
                raise
    
    async def process_images_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of image files.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of processed images with metadata
        """
        self.logger.info(f"Processing batch of {len(image_paths)} images")
        
        processed_images = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess image
                preprocessed = self.preprocessor.preprocess_image(image)
                
                # Segment page
                segments = self.segmenter.segment_page(preprocessed)
                
                # Store processed image data
                page_data = {
                    'page_number': i + 1,
                    'source_file': image_path,
                    'image': preprocessed,
                    'segments': segments,
                    'original_size': image.size,
                    'processed_size': preprocessed.size
                }
                processed_images.append(page_data)
                
            except Exception as e:
                self.logger.error(f"Error processing image {image_path}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_images)} images")
        return processed_images

    async def process_document(self, file_path: str) -> List[Image.Image]:
        """
        Process a document file (PDF or image) with memory optimization.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of processed images
        """
        self.logger.info(f"Processing document: {file_path}")
        file_path = str(file_path)  # Ensure string path
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            # Determine file type
            if file_ext in ['.pdf']:
                # Process PDF
                processed_data = await self._process_pdf_with_memory_optimization(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
                # Process single image
                image = Image.open(file_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                preprocessed = self.preprocessor.preprocess_image(image)
                processed_data = [preprocessed]
                
                # Clean up original image
                image.close()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            self.logger.info(f"Document processing completed: {len(processed_data)} pages")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def _process_pdf_with_memory_optimization(self, pdf_path: str) -> List[Image.Image]:
        """
        Process PDF with memory optimization for large documents.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed images
        """
        self.logger.debug(f"Processing PDF with memory optimization: {pdf_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Determine page range
            start_page = (self.first_page - 1) if self.first_page else 0
            end_page = self.last_page if self.last_page else doc.page_count
            
            processed_images = []
            
            # Process pages one by one to conserve memory
            for page_num in range(start_page, min(end_page, doc.page_count)):
                self.logger.debug(f"Processing PDF page {page_num + 1}")
                
                # Extract image from PDF page
                page = doc[page_num]
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess image
                preprocessed = self.preprocessor.preprocess_image(image)
                processed_images.append(preprocessed)
                
                # Clean up to free memory
                image.close()
                pix = None
                page = None
                
                # Force garbage collection after each page
                if page_num % 5 == 0:  # Every 5 pages
                    gc.collect()
            
            # Clean up
            doc.close()
            gc.collect()
            
            return processed_images
            
        except Exception as e:
            self.logger.error(f"PDF processing with memory optimization failed: {e}")
            
            # Fallback to regular method
            self.logger.warning("Falling back to standard PDF processing")
            images = await self._pdf_to_images(pdf_path)
            processed_images = [self.preprocessor.preprocess_image(img) for img in images]
            
            return processed_images
