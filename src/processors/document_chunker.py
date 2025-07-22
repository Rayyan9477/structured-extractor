"""
Adaptive Document Chunker for Large Medical Documents

Provides specialized chunking of large medical documents and PDFs to ensure optimal processing
by VLM models without exceeding token limits or memory constraints.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import logging
from pathlib import Path

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger


class DocumentChunker:
    """
    Handles intelligent chunking of large documents and pages to optimize VLM processing.
    Uses layout analysis and content density estimation to create optimally-sized chunks.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize document chunker.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load chunking configuration
        chunking_config = config.get("document_processing.chunking", {})
        self.max_width = chunking_config.get("max_width", 1200)
        self.max_height = chunking_config.get("max_height", 1600)
        self.overlap_percent = chunking_config.get("overlap_percent", 10)
        self.max_tokens_per_chunk = chunking_config.get("max_tokens_per_chunk", 8000)
        self.estimated_tokens_per_pixel = chunking_config.get("estimated_tokens_per_pixel", 0.0003)
        self.use_layout_detection = chunking_config.get("use_layout_detection", True)
        
    def chunk_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Chunk a large image into optimally-sized pieces for VLM processing.
        
        Args:
            image: PIL Image to chunk
            
        Returns:
            List of chunk dictionaries with image data and metadata
        """
        self.logger.debug(f"Chunking image of size {image.width}x{image.height}")
        
        # Check if image needs chunking
        if (image.width <= self.max_width and 
            image.height <= self.max_height and
            self._estimate_tokens(image) <= self.max_tokens_per_chunk):
            # Image is small enough to process as-is
            return [{
                'image': image,
                'bbox': (0, 0, image.width, image.height),
                'is_full_page': True,
                'chunk_index': 0,
                'total_chunks': 1,
                'estimated_tokens': self._estimate_tokens(image)
            }]
        
        # Image needs chunking
        chunks = []
        
        if self.use_layout_detection:
            # Use layout detection to create smart chunks
            chunks = self._layout_based_chunking(image)
        
        # If layout detection fails or is disabled, fall back to grid-based chunking
        if not chunks:
            chunks = self._grid_based_chunking(image)
        
        self.logger.debug(f"Created {len(chunks)} chunks from image")
        return chunks
    
    def _estimate_tokens(self, image: Image.Image) -> int:
        """
        Estimate the number of tokens that would be generated from an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Estimated token count
        """
        # Basic estimation based on image size and a density factor
        # This is a heuristic and can be improved with more sophisticated models
        pixel_count = image.width * image.height
        text_density = self._estimate_text_density(image)
        
        estimated_tokens = int(pixel_count * self.estimated_tokens_per_pixel * text_density)
        return max(100, estimated_tokens)  # Minimum token count of 100
    
    def _estimate_text_density(self, image: Image.Image) -> float:
        """
        Estimate the text density of an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Text density factor (0.0-1.0)
        """
        try:
            # Convert to OpenCV format and grayscale
            cv_image = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding to identify potential text areas
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Use morphological operations to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Count non-zero pixels after morphological operations
            text_pixels = cv2.countNonZero(morph)
            total_pixels = image.width * image.height
            
            # Calculate density and adjust it
            raw_density = text_pixels / total_pixels
            
            # Apply non-linear scaling to better represent token consumption
            # Higher density means more text content
            adjusted_density = min(1.0, 0.3 + (raw_density * 1.5))
            
            return adjusted_density
            
        except Exception as e:
            self.logger.warning(f"Error estimating text density: {e}")
            return 0.5  # Default medium density
    
    def _layout_based_chunking(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Create chunks based on document layout analysis.
        
        Args:
            image: PIL Image
            
        Returns:
            List of chunk dictionaries
        """
        try:
            # Convert to OpenCV format
            cv_image = np.array(image.convert('RGB'))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use layout analysis to detect logical blocks in the document
            # This could be tables, paragraphs, headers, etc.
            blocks = self._detect_layout_blocks(gray)
            
            if not blocks:
                self.logger.debug("No layout blocks detected, falling back to grid chunking")
                return []
            
            # Group blocks into chunks that fit within token limits
            chunks = []
            current_blocks = []
            current_token_estimate = 0
            
            for block in blocks:
                x, y, w, h = block
                
                # Estimate tokens for this block
                block_image = image.crop((x, y, x+w, y+h))
                block_tokens = self._estimate_tokens(block_image)
                
                # If this block alone exceeds token limit, it needs internal chunking
                if block_tokens > self.max_tokens_per_chunk:
                    # If we have accumulated blocks, create a chunk from them first
                    if current_blocks:
                        chunk_image, bbox = self._create_chunk_from_blocks(image, current_blocks)
                        chunks.append({
                            'image': chunk_image,
                            'bbox': bbox,
                            'is_full_page': False,
                            'chunk_index': len(chunks),
                            'total_chunks': 0,  # Will update later
                            'estimated_tokens': current_token_estimate
                        })
                        current_blocks = []
                        current_token_estimate = 0
                    
                    # Now handle the oversized block
                    block_chunks = self._chunk_large_block(image, block)
                    chunks.extend(block_chunks)
                    continue
                
                # If adding this block would exceed token limit, create a chunk from current blocks
                if current_token_estimate + block_tokens > self.max_tokens_per_chunk and current_blocks:
                    chunk_image, bbox = self._create_chunk_from_blocks(image, current_blocks)
                    chunks.append({
                        'image': chunk_image,
                        'bbox': bbox,
                        'is_full_page': False,
                        'chunk_index': len(chunks),
                        'total_chunks': 0,  # Will update later
                        'estimated_tokens': current_token_estimate
                    })
                    current_blocks = []
                    current_token_estimate = 0
                
                # Add block to current batch
                current_blocks.append(block)
                current_token_estimate += block_tokens
            
            # Add any remaining blocks as a final chunk
            if current_blocks:
                chunk_image, bbox = self._create_chunk_from_blocks(image, current_blocks)
                chunks.append({
                    'image': chunk_image,
                    'bbox': bbox,
                    'is_full_page': False,
                    'chunk_index': len(chunks),
                    'total_chunks': 0,  # Will update later
                    'estimated_tokens': current_token_estimate
                })
            
            # Update total chunks count
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk['total_chunks'] = total_chunks
            
            return chunks
            
        except Exception as e:
            self.logger.warning(f"Layout-based chunking failed: {e}")
            return []
    
    def _detect_layout_blocks(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect layout blocks in the document.
        
        Args:
            gray_image: Grayscale OpenCV image
            
        Returns:
            List of block bounding boxes (x, y, width, height)
        """
        blocks = []
        
        # Method 1: Use connected components to find text blocks
        thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Use morphological operations to connect text within same blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Connect words horizontally
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Connect lines vertically but less aggressively
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
        
        # Find contours which correspond to blocks
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray_image.shape
        min_area = (width * height) * 0.001  # Minimum block size (0.1% of page)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if area > min_area:
                # Expand rectangle slightly to ensure full content is captured
                x = max(0, x - 5)
                y = max(0, y - 5)
                w = min(width - x, w + 10)
                h = min(height - y, h + 10)
                
                blocks.append((x, y, w, h))
        
        # Sort blocks top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b[1], b[0]))
        
        return blocks
    
    def _create_chunk_from_blocks(self, image: Image.Image, blocks: List[Tuple[int, int, int, int]]) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Create a chunk image from a list of blocks.
        
        Args:
            image: Original PIL Image
            blocks: List of block bounding boxes
            
        Returns:
            Tuple of (chunk image, bounding box)
        """
        # Find the bounding box that encompasses all blocks
        min_x = min(block[0] for block in blocks)
        min_y = min(block[1] for block in blocks)
        max_x = max(block[0] + block[2] for block in blocks)
        max_y = max(block[1] + block[3] for block in blocks)
        
        # Add some padding
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.width, max_x + padding)
        max_y = min(image.height, max_y + padding)
        
        # Crop the image to create the chunk
        chunk_image = image.crop((min_x, min_y, max_x, max_y))
        bbox = (min_x, min_y, max_x, max_y)
        
        return chunk_image, bbox
    
    def _chunk_large_block(self, image: Image.Image, block: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
        """
        Chunk a large block that exceeds token limits.
        
        Args:
            image: Original PIL Image
            block: Block bounding box
            
        Returns:
            List of chunk dictionaries
        """
        x, y, w, h = block
        
        # Crop the block
        block_image = image.crop((x, y, x+w, y+h))
        
        # Use grid chunking on this block
        grid_chunks = self._grid_based_chunking(block_image)
        
        # Adjust bounding boxes to be relative to original image
        for chunk in grid_chunks:
            chunk_bbox = chunk['bbox']
            adjusted_bbox = (
                x + chunk_bbox[0],
                y + chunk_bbox[1],
                x + chunk_bbox[2],
                y + chunk_bbox[3]
            )
            chunk['bbox'] = adjusted_bbox
        
        return grid_chunks
    
    def _grid_based_chunking(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Create chunks based on a grid layout.
        
        Args:
            image: PIL Image
            
        Returns:
            List of chunk dictionaries
        """
        # Calculate grid dimensions based on max dimensions
        cols = math.ceil(image.width / self.max_width)
        rows = math.ceil(image.height / self.max_height)
        
        # If the image is still too dense (text-heavy), increase the grid density
        estimated_tokens = self._estimate_tokens(image)
        estimated_tokens_per_chunk = estimated_tokens / (rows * cols)
        
        if estimated_tokens_per_chunk > self.max_tokens_per_chunk:
            # Increase grid density to reduce tokens per chunk
            token_ratio = estimated_tokens_per_chunk / self.max_tokens_per_chunk
            area_increase = math.sqrt(token_ratio)
            
            cols = math.ceil(cols * area_increase)
            rows = math.ceil(rows * area_increase)
        
        # Calculate chunk size
        chunk_width = image.width / cols
        chunk_height = image.height / rows
        
        # Calculate overlap
        overlap_x = int(chunk_width * (self.overlap_percent / 100))
        overlap_y = int(chunk_height * (self.overlap_percent / 100))
        
        chunks = []
        chunk_index = 0
        
        for row in range(rows):
            for col in range(cols):
                # Calculate chunk coordinates with overlap
                x1 = max(0, int(col * chunk_width) - (overlap_x if col > 0 else 0))
                y1 = max(0, int(row * chunk_height) - (overlap_y if row > 0 else 0))
                x2 = min(image.width, int((col + 1) * chunk_width) + (overlap_x if col < cols - 1 else 0))
                y2 = min(image.height, int((row + 1) * chunk_height) + (overlap_y if row < rows - 1 else 0))
                
                # Crop the chunk
                chunk_image = image.crop((x1, y1, x2, y2))
                
                # Create chunk metadata
                chunks.append({
                    'image': chunk_image,
                    'bbox': (x1, y1, x2, y2),
                    'is_full_page': False,
                    'chunk_index': chunk_index,
                    'total_chunks': rows * cols,
                    'estimated_tokens': self._estimate_tokens(chunk_image)
                })
                
                chunk_index += 1
        
        return chunks
