# Medical Superbill Data Extraction System

A comprehensive Python project designed to automate the extraction of structured data from medical superbills using advanced OCR and NLP models from Hugging Face.

## 🚀 Features

- **Multi-Model OCR Pipeline**: Supports handwritten and printed text using echo840/MonkeyOCR and nanonets/Nanonets-OCR-s
- **Intelligent Data Extraction**: Utilizes numind/NuExtract-2.0-4B for structured medical data extraction
- **Multi-Patient Document Handling**: Automatically detects and separates multiple patients within single documents
- **HIPAA-Compliant Processing**: Built-in PHI detection and anonymization capabilities
- **Comprehensive Validation**: CPT code, ICD-10 code, and medical data validation
- **Multiple Export Formats**: CSV, JSON, and Excel export with customizable formatting
- **Async Processing**: High-performance async/await architecture for batch processing
- **Configurable Pipeline**: YAML-based configuration for easy customization

## 📋 Extracted Data Fields

- **Patient Information**: Names, DOB, Patient ID, Contact details
- **Medical Codes**: CPT codes, ICD-10 diagnosis codes with descriptions
- **Service Information**: Date of service, Provider details, NPI numbers
- **Financial Data**: Charges, payments, insurance information
- **PHI Detection**: Automatic identification of protected health information

## 🏗️ Architecture

```
src/
├── core/                    # Core configuration and data models
│   ├── config_manager.py    # YAML configuration management
│   ├── logger.py           # HIPAA-compliant logging
│   └── data_schema.py      # Pydantic data models
├── processors/             # Document and OCR processing
│   ├── document_processor.py  # PDF/image preprocessing
│   └── ocr_engine.py          # Multi-model OCR pipeline
├── models/                 # Model management
│   └── model_manager.py    # Hugging Face model handling
├── extractors/             # Data extraction engines
│   ├── field_detector.py   # Regex-based field detection
│   ├── nuextract_engine.py # NuExtract integration
│   └── multi_patient_handler.py # Multi-patient processing
├── validators/             # Data validation
│   └── data_validator.py   # CPT/ICD-10/date validation
├── exporters/              # Export functionality
│   └── data_exporter.py    # CSV/JSON/Excel export
└── extraction_engine.py    # Main extraction pipeline
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd structured-extractor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system** (optional):
   Edit `config/config.yaml` to customize settings.

### Basic Usage

```python
import asyncio
from src.extraction_engine import ExtractionEngine

async def extract_superbill():
    # Initialize the extraction engine
    engine = ExtractionEngine()
    
    # Extract data from a file
    results = await engine.extract_from_file("path/to/superbill.pdf")
    
    # Print results
    print(f"Found {results.total_patients} patients")
    for patient in results.patients:
        print(f"Patient: {patient.first_name} {patient.last_name}")
        print(f"CPT Codes: {[cpt.code for cpt in patient.cpt_codes]}")
        print(f"ICD-10 Codes: {[icd.code for icd in patient.icd10_codes]}")
    
    # Export to CSV
    engine.export_to_csv(results, "output/extracted_data.csv")

# Run the extraction
asyncio.run(extract_superbill())
```

### Text Extraction

```python
async def extract_from_text():
    engine = ExtractionEngine()
    
    text = """
    Patient: John Doe
    DOB: 01/15/1980
    CPT: 99213 - Office visit
    ICD-10: I10 - Essential hypertension
    """
    
    results = await engine.extract_from_text(text)
    return results
```

### Batch Processing

```python
async def batch_process():
    engine = ExtractionEngine()
    
    file_paths = ["file1.pdf", "file2.pdf", "file3.pdf"]
    results_list = await engine.extract_batch(file_paths)
    
    # Export all results
    for i, results in enumerate(results_list):
        engine.export_to_json(results, f"output/results_{i}.json")
```

## 🧪 Testing

Run the example script to test the system:

```bash
python examples/usage_example.py
```

The script demonstrates:
- Basic file extraction
- Text-based extraction
- Data validation
- Multi-format export
- Batch processing
- PHI anonymization

## 📁 Project Structure

```
structured-extractor/
├── config/
│   └── config.yaml          # Main configuration file
├── src/                     # Source code
│   ├── core/               # Core components
│   ├── processors/         # Document processing
│   ├── models/             # Model management
│   ├── extractors/         # Data extraction
│   ├── validators/         # Data validation
│   ├── exporters/          # Export functionality
│   └── extraction_engine.py # Main engine
├── examples/               # Usage examples
│   ├── usage_example.py    # Comprehensive example
│   └── sample_superbill.pdf # Sample file (add your own)
├── docs/                   # Documentation
├── tests/                  # Unit tests
├── output/                 # Output directory
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔧 System Requirements

- **Python**: 3.8+
- **Memory**: 8GB RAM minimum (16GB recommended for large batches)
- **Storage**: 5GB for models and dependencies
- **GPU**: Optional but recommended for faster processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For providing the transformer models
- **echo840/MonkeyOCR**: Handwriting OCR model
- **numind/NuExtract-2.0-4B**: Structured data extraction model
- **nanonets/Nanonets-OCR-s**: Printed text OCR model

---

**Note**: This system is designed for research and development purposes. Ensure compliance with HIPAA and other healthcare regulations when processing real patient data.
