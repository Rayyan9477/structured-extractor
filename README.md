# Medical Superbill Data Extraction System

A comprehensive Python project for extracting structured data from medical superbills using advanced OCR and NLP models from Hugging Face.

## Features

- **Multi-Model OCR**: Combines MonkeyOCR for handwriting and Nanonets-OCR for printed text
- **Intelligent Field Extraction**: Uses NuExtract-2.0-4B for structured data extraction
- **Multi-Patient Support**: Handles multiple patients in single documents
- **HIPAA Compliance**: PHI detection and optional anonymization
- **Flexible Export**: CSV and JSON output formats
- **Medical Code Validation**: CPT and ICD-10 code verification
- **High Performance**: GPU acceleration and batch processing
- **Modern UI**: Intuitive web interface built with Streamlit

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rayyan9477/structured-extractor.git
cd structured-extractor

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process a single PDF file
python main.py input_superbill.pdf --output results.json

# Process multiple files
python main.py *.pdf --output-dir ./results/ --format csv

# Enable PHI anonymization
python main.py input.pdf --anonymize-phi --output results.json

# Launch the web UI
python run_ui.py
```

### Using the Web UI

The web UI provides an intuitive interface for all extraction features:

```bash
# Start the web interface
python run_ui.py
```

This will open a browser window with the Medical Superbill Extractor UI, where you can:

- Upload and process individual files
- Batch process multiple documents
- Configure extraction parameters
- Export results in various formats
- Validate extraction results

## Project Structure

```
structured-extractor/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── config_manager.py     # Configuration management
│   │   ├── logger.py             # HIPAA-compliant logging
│   │   └── data_schema.py        # Data models and validation
│   ├── processors/               # Document processing
│   ├── models/                   # Model management
│   ├── extractors/               # Field extraction
│   ├── validators/               # Data validation
│   └── exporters/                # Output generation
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── docs/                        # Documentation
├── tests/                       # Test files
├── examples/                    # Usage examples
├── data/                        # Sample data
├── main.py                      # Application entry point
└── requirements.txt             # Dependencies
```

## Configuration

The system uses YAML configuration files for flexible setup:

```yaml
# Example configuration
models:
  ocr:
    monkey_ocr:
      model_name: "echo840/MonkeyOCR"
      confidence_threshold: 0.7
    nanonets_ocr:
      model_name: "nanonets/Nanonets-OCR-s"
      confidence_threshold: 0.8
  
  extraction:
    nuextract:
      model_name: "numind/NuExtract-2.0-4B"
      temperature: 0.1

extraction_fields:
  required_fields:
    - "cpt_codes"
    - "diagnosis_codes"
    - "patient_name"
    - "date_of_service"
```

## Extracted Fields

The system extracts the following key fields from superbills:

### Patient Information
- Patient name, DOB, address, phone
- Insurance information
- Account/patient ID numbers

### Medical Codes
- **CPT Codes**: 5-digit procedure codes
- **ICD-10 Codes**: Diagnosis codes
- **Modifiers**: Procedure modifiers

### Service Information
- Date of service
- Claim date
- Place of service
- Visit type and duration

### Financial Information
- Charges and fees
- Copayments
- Outstanding balances
- Payment information

### Provider Information
- Provider name and NPI
- Practice information
- Referring providers

## Command Line Options

```bash
# Input and Output
python main.py input.pdf --output results.json
python main.py *.pdf --output-dir ./results/
python main.py input.pdf --format csv

# Processing Options
python main.py input.pdf --batch-size 8 --max-workers 4
python main.py input.pdf --gpu  # Force GPU usage
python main.py input.pdf --cpu-only  # Force CPU usage

# PHI and Security
python main.py input.pdf --anonymize-phi
python main.py input.pdf --detect-phi-only

# Quality Control
python main.py input.pdf --confidence-threshold 0.8
python main.py input.pdf --validate-codes
python main.py input.pdf --skip-validation

# Logging and Debug
python main.py input.pdf --verbose
python main.py input.pdf --debug
python main.py input.pdf --log-file extraction.log
```

## Output Formats

### JSON Output
```json
{
  "metadata": {
    "document_id": "doc_001",
    "source_file": "superbill.pdf",
    "processing_timestamp": "2025-07-17T10:30:00Z"
  },
  "patients": [
    {
      "first_name": "John",
      "last_name": "Doe",
      "date_of_birth": "1980-01-15",
      "cpt_codes": [
        {
          "code": "99213",
          "description": "Office visit",
          "charge": 150.00
        }
      ],
      "icd10_codes": [
        {
          "code": "M54.5",
          "description": "Low back pain"
        }
      ]
    }
  ]
}
```

### CSV Output
Patient-separated CSV with all extracted fields in tabular format.

## HIPAA Compliance

The system includes comprehensive HIPAA compliance features:

- **PHI Detection**: Automatic identification of protected health information
- **Audit Logging**: Complete audit trails for all operations
- **Data Security**: Secure handling and optional anonymization
- **Access Controls**: User authentication and authorization
- **Encryption**: Data encryption in transit and at rest

## Performance

- **Processing Speed**: <30 seconds per page
- **Accuracy**: >95% for critical fields
- **Memory Usage**: <8GB for standard documents
- **Concurrent Processing**: Up to 4 documents simultaneously

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
# Linting
flake8 src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

## API Usage

The system also provides a REST API for programmatic access:

```python
import requests

# Upload and process a file
with open('superbill.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/extract',
        files={'file': f},
        data={'format': 'json', 'anonymize_phi': True}
    )

result = response.json()
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Contact the development team
- Check the documentation in the `docs/` directory

## Acknowledgments

- Hugging Face for providing the ML models
- Medical coding communities for validation resources
- HIPAA compliance guidelines and best practices
