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
