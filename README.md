# Unified Structured Extraction System

A powerful document extraction system that combines multiple OCR and extraction models to achieve high accuracy in extracting structured data from documents.

## Models Used

This system leverages three powerful models for optimal extraction performance:

1. **NumInd NuExtract-2.0-8B** - A state-of-the-art language model specialized for extracting structured information from raw text. It converts document text into well-formatted structured data according to provided schemas.

2. **Monkey OCR** - An advanced OCR engine optimized for complex document layouts. It excels at processing documents with mixed formats, tables, and varying font styles.

3. **Nanonets OCR** - A specialized OCR model with high accuracy for document understanding. It has particularly strong performance for detecting form fields and structured content.

The system integrates these models in a carefully designed pipeline:
- Both OCR models process the document images in parallel
- Results are ensembled using a weighted voting mechanism
- The combined OCR text is fed into NuExtract for structured extraction
- Confidence scores from all models contribute to the final quality assessment

## Features

- **Multi-model OCR ensemble** - Combines Monkey OCR and Nanonets OCR results for improved text recognition
- **Structured data extraction** - Uses NumInd NuExtract 8B to convert text to structured data
- **Customizable templates** - Define custom schemas for different document types
- **Confidence scoring** - Provides detailed confidence scores for extracted data
- **Multiple export formats** - Export results to JSON, CSV, XML, or plain text
- **Batch processing** - Process multiple documents efficiently
- **Command-line interface** - Easy to use from the command line
- **Programmatic API** - Integrate with other applications

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Sufficient disk space for model weights (approximately 16GB for all models)
- GPU recommended for faster processing with NuExtract 8B

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/structured-extractor.git
   cd structured-extractor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the system:
   ```
   python src/cli.py config create
   ```
   
4. Set up API keys in the configuration file:
   - For Monkey OCR: Set your API key in the `ocr.monkey_ocr.api_key` field
   - For Nanonets OCR: Set your API key in the `ocr.nanonets_ocr.api_key` field

## Usage

### Command Line Interface

The system provides a powerful command-line interface:

```bash
# Extract from a file using all three models
python src/cli.py extract-file document.pdf -o results.json

# Extract using a specific template
python src/cli.py extract-file document.pdf -t medical -o results.json

# Batch processing
python src/cli.py batch documents_folder/ -o output_folder/

# Show help
python src/cli.py -h
```

### Python API

You can also use the Python API in your applications:

```python
import asyncio
from src.unified_extraction_system import UnifiedExtractionSystem

async def extract_example():
    # Initialize the extraction system
    extractor = UnifiedExtractionSystem()
    
    # Extract from a file
    result = await extractor.extract_from_file(
        "document.pdf",
        output_path="results.json"
    )
    
    print(f"Extraction confidence: {result.overall_confidence}")
    print(f"OCR confidence: {result.ocr_confidence}")
    print(f"NuExtract confidence: {result.extraction_confidence}")
    print(result.structured_data)

# Run the example
asyncio.run(extract_example())
```

## Example Usage

Run the included example script:

```bash
python examples/extract_document.py --file your_document.pdf
```

### More Examples:

```bash
# Extract medical document using the medical template
python examples/extract_document.py --medical medical_report.pdf

# Process a batch of documents
python examples/extract_document.py --batch documents_folder/

# Extract with custom schema
python examples/extract_document.py --custom invoice.pdf

# Export to multiple formats
python examples/extract_document.py --formats document.pdf --output exports/
```

## Configuration

The system can be configured by editing `config/config.yaml`:

### Key Configuration Options:

1. **OCR Model Settings**:
   - Enable/disable specific OCR models
   - Adjust model weights in the ensemble
   - Configure API endpoints and keys

2. **NuExtract Settings**:
   - Adjust temperature and other generation parameters
   - Define custom extraction templates
   - Configure confidence thresholds

3. **Export Settings**:
   - Configure output formats
   - Set default export directory

## Folder Structure

```
structured-extractor/
├── config/                # Configuration files
├── examples/              # Example usage scripts
├── output/                # Default output directory
├── src/                   # Source code
│   ├── core/              # Core modules and utilities
│   ├── processors/        # Document and OCR processors
│   │   ├── monkey_ocr.py  # Monkey OCR integration
│   │   ├── nanonets_ocr.py # Nanonets OCR integration
│   │   └── ocr_ensemble.py # OCR ensemble engine
│   ├── extractors/        # Structured data extractors
│   │   └── nuextract_structured_extractor.py # NuExtract integration
│   ├── exporters/         # Export formatters
│   ├── cli.py             # Command line interface
│   └── unified_extraction_system.py  # Main system
└── tests/                 # Test suite
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
