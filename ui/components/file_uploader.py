"""
File Uploader Component for Medical Superbill Extractor UI
"""
import streamlit as st
import os
from pathlib import Path
from PIL import Image

# Try to import magic, but fallback gracefully if not available
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    st.warning("⚠️ python-magic not available. Using basic file type detection.")

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_DIMENSION = 4096
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'image/jpeg', 
    'image/png',
    'image/tiff'
}

class FileValidator:
    """Comprehensive file validation for security."""
    
    def validate_file(self, file_data: bytes, filename: str) -> dict:
        """Validate uploaded file comprehensively."""
        results = {
            'valid': False,
            'errors': [],
            'file_type': None,
            'warnings': []
        }
        
        # Size validation
        if len(file_data) > MAX_FILE_SIZE:
            results['errors'].append(f"File too large: {len(file_data)/1024/1024:.1f}MB (max: {MAX_FILE_SIZE/1024/1024}MB)")
            return results
        
        if len(file_data) < 100:  # Minimum viable file size
            results['errors'].append("File appears to be empty or corrupted")
            return results
        
        # MIME type validation using python-magic for security
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(file_data, mime=True)
            except:
                # Fallback to basic validation if magic fails
                mime_type = self._guess_mime_type(filename, file_data)
        else:
            # Use basic validation if python-magic is not available
            mime_type = self._guess_mime_type(filename, file_data)
        
        if mime_type not in ALLOWED_MIME_TYPES:
            results['errors'].append(f"Invalid file type: {mime_type}")
            return results
        
        results['file_type'] = mime_type
        
        # Format-specific validation
        if mime_type == 'application/pdf':
            pdf_valid, pdf_message = self._validate_pdf(file_data)
            if not pdf_valid:
                results['errors'].append(pdf_message)
                return results
        elif mime_type.startswith('image/'):
            img_valid, img_message = self._validate_image(file_data)
            if not img_valid:
                results['errors'].append(img_message)
                return results
        
        results['valid'] = True
        return results
    
    def _guess_mime_type(self, filename: str, data: bytes) -> str:
        """Fallback MIME type detection."""
        # Basic file signature detection
        if data.startswith(b'%PDF'):
            return 'application/pdf'
        elif data.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif data.startswith(b'\x89PNG'):
            return 'image/png'
        elif data.startswith(b'II*\x00') or data.startswith(b'MM\x00*'):
            return 'image/tiff'
        else:
            return 'application/octet-stream'  # Unknown type
    
    def _validate_pdf(self, data: bytes) -> tuple:
        """Validate PDF structure and content."""
        try:
            import io
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(data))
            
            # Check if PDF has pages
            if len(pdf_reader.pages) == 0:
                return False, "PDF file contains no pages"
            
            # Check for reasonable page count (prevent DoS)
            if len(pdf_reader.pages) > 100:
                return False, "PDF file has too many pages (max: 100)"
            
            return True, "PDF validation successful"
        except Exception as e:
            return False, "Invalid or corrupted PDF file"
    
    def _validate_image(self, data: bytes) -> tuple:
        """Validate image format and dimensions."""
        try:
            import io
            image = Image.open(io.BytesIO(data))
            
            # Check dimensions
            if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                return False, f"Image too large: {image.width}x{image.height} (max: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            
            # Check for reasonable file size vs dimensions ratio (detect potential zip bombs)
            pixel_count = image.width * image.height
            if len(data) > pixel_count * 10:  # Reasonable compression ratio
                return False, "Image file appears to be corrupted or malicious"
            
            return True, "Image validation successful"
        except Exception as e:
            return False, "Invalid or corrupted image file"


def render_file_uploader():
    """
    Render the file uploader component with drag and drop functionality and security validation.
    
    Returns:
        The uploaded file object if a file was uploaded and validated, otherwise None
    """
    st.markdown("""
    <div class="card">
        <div class="card-header">Upload Document</div>
    """, unsafe_allow_html=True)
    
    # Initialize validator
    validator = FileValidator()
    
    # File types explanation
    file_col, format_col = st.columns([3, 2])
    
    with file_col:
        # Add some descriptive text
        st.markdown("Upload a medical superbill document to extract data.")
        st.markdown("⚠️ **Security Notice**: Files are validated for safety before processing.")
        
        # Create the file uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "jpg", "jpeg", "png", "tiff", "tif"],
            label_visibility="collapsed"
        )
        
        # Validate uploaded file
        if uploaded_file is not None:
            file_data = uploaded_file.getbuffer()
            validation_result = validator.validate_file(bytes(file_data), uploaded_file.name)
            
            if not validation_result['valid']:
                st.error("File validation failed:")
                for error in validation_result['errors']:
                    st.error(f"• {error}")
                uploaded_file = None  # Reject invalid file
            else:
                st.success(f"✅ File validated successfully ({validation_result['file_type']})")
                if validation_result.get('warnings'):
                    for warning in validation_result['warnings']:
                        st.warning(f"⚠️ {warning}")
    
    with format_col:
        st.markdown("**Supported Formats:**")
        st.markdown("- PDF Documents (.pdf)")
        st.markdown("- Images (.jpg, .png, .tiff)")
        st.markdown("---")
        st.markdown("**PDF Processing Features:**")
        st.markdown("✅ Page counting & optimization")
        st.markdown("✅ VLM token limit handling")  
        st.markdown("✅ Multi-patient detection")
        st.markdown("✅ Adaptive batch processing")
    
    # Display preview if file is uploaded
    if uploaded_file is not None:
        # Get file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Show file info
        st.markdown(f"**File:** {uploaded_file.name}")
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "File Size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        # Display file details in a cleaner format
        details_md = ""
        for key, value in file_details.items():
            details_md += f"**{key}:** {value}  \n"
        st.markdown(details_md)
        
        # Preview for images
        if file_extension in ['.jpg', '.jpeg', '.png']:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Document Preview", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
        
        # For PDFs, show enhanced info message
        elif file_extension == '.pdf':
            st.success("📄 PDF document loaded successfully!")
            st.info("The system will automatically:\n"
                   "• Count total pages\n"
                   "• Optimize each page for VLM processing\n" 
                   "• Detect and differentiate multiple patients\n"
                   "• Handle token limits intelligently\n\n"
                   "Click 'Extract Data' to begin processing.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return uploaded_file
