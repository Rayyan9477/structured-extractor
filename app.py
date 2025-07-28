import streamlit as st
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from pdf2image import convert_from_path
from PIL import Image
import json
import torch
import logging
import tempfile
import os
def process_pdf(pdf_file):
    """Convert PDF to images and extract data from each page."""
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            tmp_path = tmp_file.name
        
        try:
            pages = convert_from_path(tmp_path)
            results = []
            for i, page in enumerate(pages): pdf2image import convert_from_path
from PIL import Image
import json
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set device for model inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading functions with caching
@st.cache_resource
def load_ocr_model():
    """Load the OCR model, tokenizer, and processor."""
    try:
        model = AutoModelForImageTextToText.from_pretrained("nanonets/Nanonets-OCR-s").to(device)
        tokenizer = AutoTokenizer.from_pretrained("nanonets/Nanonets-OCR-s")
        processor = AutoProcessor.from_pretrained("nanonets/Nanonets-OCR-s")
        logger.info("OCR model loaded successfully.")
        return model, tokenizer, processor
    except Exception as e:
        logger.error(f"Failed to load OCR model: {str(e)}")
        raise

@st.cache_resource
def load_extractor_model():
    """Load the data extractor model and tokenizer."""
    try:
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained("numind/NuExtract-2.0-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "numind/NuExtract-2.0-8B",
            config=config,
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "numind/NuExtract-2.0-8B",
            trust_remote_code=True
        )
        logger.info("Extractor model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load extractor model: {str(e)}")
        raise

# Initialize models
try:
    ocr_model, ocr_tokenizer, ocr_processor = load_ocr_model()
    extractor_model, extractor_tokenizer = load_extractor_model()
except Exception as e:
    st.error(f"Model initialization failed: {str(e)}")
    st.stop()

# OCR extraction function
def extract_text_from_image(image, model, processor, tokenizer):
    """Extract text from an image with detailed formatting instructions."""
    prompt = """
    Extract all visible text from the document with high accuracy. Include:
    - Headers and footers as plain text, prefixed with "[Header]" or "[Footer]".
    - Watermarks in [brackets], e.g., [Watermark: Confidential].
    - Tables in markdown format with clear column alignment.
    - Equations in LaTeX notation, e.g., \\( x^2 + y^2 = z^2 \\).
    - Brief descriptions of images or their captions if present, e.g., [Image: Chart showing patient vitals].
    Ensure no text is missed, even if faint or rotated.
    """
    messages = [
        {"role": "system", "content": "You are an expert OCR assistant optimized for document analysis."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info("Text extracted successfully from image.")
        return generated_text[0]
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        return f"Error: {str(e)}"

# Medical field extraction function
def extract_medical_fields(text, model, tokenizer):
    """Extract structured medical data from text into JSON format."""
    fields = ["patient_name", "age", "gender", "diagnosis", "treatment", "date_of_visit"]
    prompt = f"""
    You are a medical data extraction expert. Analyze the provided text and extract the following fields: 
    {', '.join(fields)}. 
    - Use contextual clues (e.g., "Patient: John Doe" or "Dx: Hypertension") to identify values.
    - For missing fields, return an empty string ("").
    - For dates, prefer formats like "YYYY-MM-DD" if possible.
    - Output the result as a valid JSON object.

    Text to analyze:
    {text}

    Output in JSON format:
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=1024)
        extracted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Try to extract JSON from the output, even if there is extra text
        import re
        match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            extracted_data = json.loads(json_str)
            logger.info("Medical fields extracted successfully.")
            return extracted_data
        else:
            logger.error("No valid JSON found in model output.")
            return {"error": "No valid JSON found in model output", "raw_output": extracted_text}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {str(e)}")
        return {"error": "Invalid JSON output from model", "raw_output": extracted_text}
    except Exception as e:
        logger.error(f"Field extraction failed: {str(e)}")
        return {"error": str(e)}

# Process PDF and extract data
def process_pdf(pdf_file):
    """Convert PDF to images and extract data from each page."""
    temp_path = None
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            temp_path = tmp_file.name
        
        # Convert PDF pages to images
        pages = convert_from_path(temp_path)
        results = []
        for i, page in enumerate(pages):
            with st.spinner(f"Extracting text from page {i+1}..."):
                text = extract_text_from_image(page, ocr_model, ocr_processor, ocr_tokenizer)
            with st.spinner(f"Extracting medical fields from page {i+1}..."):
                data = extract_medical_fields(text, extractor_model, extractor_tokenizer)
                results.append({"page": i+1, "text": text, "extracted_data": data})
        return results
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")

# Streamlit UI
def main():
    st.title("Medical Data Extraction Tool")
    st.markdown("""
    Upload a PDF file containing medical records. Each page is processed as a separate patient's data, 
    extracting key fields like name, age, gender, diagnosis, treatment, and date of visit.
    """)

    pdf_file = st.file_uploader("Upload Medical PDF", type=["pdf"])

    if pdf_file:
        with st.spinner("Analyzing PDF..."):
            try:
                results = process_pdf(pdf_file)
                st.success(f"Processed {len(results)} patient records successfully!")
                
                # Display results
                patient_options = [f"Patient {r['page']}" for r in results]
                selected_patient = st.selectbox("Select Patient", patient_options)
                selected_index = patient_options.index(selected_patient)
                
                st.subheader(f"Extracted Data for {selected_patient}")
                extracted_data = results[selected_index]["extracted_data"]
                if "error" in extracted_data:
                    st.warning(f"Extraction error: {extracted_data['error']}")
                    st.text_area("Raw Model Output", extracted_data.get("raw_output", ""), height=200)
                else:
                    st.json(extracted_data)
                st.subheader("Raw Extracted Text")
                st.text_area("Text", results[selected_index]["text"], height=200)
                
                # Download all data
                all_data = json.dumps([r["extracted_data"] for r in results], indent=4)
                st.download_button(
                    label="Download All Extracted Data",
                    data=all_data,
                    file_name="medical_data.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()