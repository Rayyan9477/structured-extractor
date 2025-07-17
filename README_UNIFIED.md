# Medical Superbill Data Extraction System - Unified Application

This is a complete, unified application that runs the entire Medical Superbill Data Extraction project in a single file, including both CLI and UI/UX interfaces.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python test_system.py
```

### 3. Run the Application

#### Streamlit UI (Default)
```bash
python app_main.py
```
Then open your browser to: http://localhost:8501

#### CLI Mode
```bash
# Extract from a single file
python app_main.py --cli superbill.pdf

# Batch processing
python app_main.py --cli *.pdf --format csv --output-dir results/

# With custom configuration
python app_main.py --cli file.pdf --config custom_config.yaml
```

#### Demo Mode
```bash
python app_main.py --demo
```

## 📁 File Structure

```
structured-extractor/
├── app_main.py              # 🎯 MAIN UNIFIED APPLICATION
├── test_system.py           # System test script
├── requirements.txt         # Dependencies
├── config/
│   └── config.yaml          # Configuration file
└── temp/                    # Temporary files (auto-created)
```

## 🔧 Features

### Core Functionality
- **OCR Processing**: Tesseract + EasyOCR fallback
- **Field Extraction**: Regex-based pattern matching
- **Data Validation**: CPT/ICD-10 code validation
- **Multi-format Export**: JSON, CSV, Excel
- **Batch Processing**: Process multiple files at once

### UI Features (Streamlit)
- **Drag & Drop Upload**: Easy file uploading
- **Real-time Processing**: Live extraction results
- **Interactive Results**: Expandable patient details
- **Export Options**: Download in multiple formats
- **Configuration Panel**: Adjust extraction settings
- **Progress Tracking**: Visual processing indicators

### CLI Features
- **Batch Processing**: Handle multiple files
- **Flexible Output**: Multiple export formats
- **Configuration**: Custom config file support
- **Logging**: Detailed processing logs
- **Error Handling**: Robust error recovery

## 📊 Supported File Types

- **PDF Documents**: Medical superbills and forms
- **Images**: JPG, PNG, TIFF (scanned documents)
- **Multi-page**: Automatic page processing

## 🔍 Extracted Data

### Patient Information
- First Name, Last Name, Middle Name
- Date of Birth
- Patient ID / Account Number
- Contact Information

### Medical Codes
- **CPT Codes**: Procedure codes with descriptions
- **ICD-10 Codes**: Diagnosis codes
- **HCPCS Codes**: Healthcare procedure codes

### Service Information
- Date of Service
- Provider Information
- NPI Numbers
- Facility Details

### Financial Data
- Charges and Fees
- Copay amounts
- Deductible information
- Insurance details

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
# OCR Settings
models:
  ocr:
    primary: "tesseract"
    confidence_threshold: 0.7

# Processing Options
document_processing:
  pdf:
    dpi: 300
  image_preprocessing:
    enhance_contrast: true
    denoise: true

# Field Extraction
extraction_fields:
  patient_info:
    - "first_name"
    - "last_name"
    - "date_of_birth"
  medical_codes:
    - "cpt_codes"
    - "icd10_codes"

# Export Settings
export:
  csv:
    include_headers: true
    flatten_codes: true
```

## 🎯 Usage Examples

### Basic UI Usage
1. Run: `python app_main.py`
2. Upload a medical superbill document
3. Click "Extract Data"
4. Review and export results

### CLI Usage Examples

```bash
# Single file extraction
python app_main.py --cli medical_bill.pdf

# Batch processing with CSV output
python app_main.py --cli *.pdf --format csv --output-dir ./results/

# JSON output with custom config
python app_main.py --cli document.pdf --format json --config my_config.yaml

# All formats
python app_main.py --cli file.pdf --format all
```

### Advanced CLI Options

```bash
# Verbose logging
python app_main.py --cli file.pdf --verbose

# Custom confidence threshold
python app_main.py --cli file.pdf --confidence-threshold 0.8

# Batch size for performance
python app_main.py --cli *.pdf --batch-size 8
```

## 📈 Performance Tips

1. **Image Quality**: Higher DPI for better OCR results
2. **Batch Size**: Adjust based on available memory
3. **GPU Support**: Enable if CUDA is available
4. **File Size**: Optimize images before processing

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
```

**OCR Issues**
```bash
# Install Tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Tesseract (macOS)
brew install tesseract

# Install Tesseract (Ubuntu)
sudo apt install tesseract-ocr
```

**GPU Issues**
```bash
# For CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Getting Help

1. **Test System**: Run `python test_system.py`
2. **Demo Mode**: Run `python app_main.py --demo`
3. **Verbose Logging**: Add `--verbose` flag
4. **Configuration**: Check `config/config.yaml`

## 🔒 Security & Privacy

- **PHI Handling**: Configurable anonymization
- **Data Storage**: Temporary files auto-cleaned
- **Audit Logging**: Track data access
- **HIPAA Compliance**: Secure processing pipeline

## 📝 Output Formats

### JSON Export
```json
{
  "success": true,
  "patients": [
    {
      "first_name": "John",
      "last_name": "Smith",
      "patient_id": "12345",
      "cpt_codes": [
        {
          "code": "99213",
          "description": "Office visit",
          "confidence": {"overall": 0.85}
        }
      ]
    }
  ]
}
```

### CSV Export
```csv
Patient_Number,First_Name,Last_Name,Patient_ID,CPT_Codes,Charges
1,John,Smith,12345,99213,$150.00
```

## 🧪 Testing

```bash
# Run system tests
python test_system.py

# Test specific functionality
python app_main.py --demo

# CLI test
python app_main.py --cli test_file.pdf --verbose
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Run system tests: `python test_system.py`
3. Use demo mode: `python app_main.py --demo`
4. Enable verbose logging: `--verbose`

---

**Note**: This unified application (`app_main.py`) contains the complete functionality of the entire project in a single file, making it easy to deploy and run without complex setup procedures.
