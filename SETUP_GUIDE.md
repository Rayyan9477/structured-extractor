# Medical Superbill Extraction System - Setup and Troubleshooting Guide

## Quick Start

### 1. Verify Model Setup
```bash
python diagnose_models.py
```
This will check if all models are properly installed and working.

### 2. Test the System
```bash
python test_standalone_system.py
```
This runs comprehensive tests of all components.

### 3. Run Extraction
```bash
python main.py path/to/your/medical_document.pdf
```

## Model Requirements

The system uses three locally downloaded models:

### MonkeyOCR (echo840/MonkeyOCR)
- **Location**: `models/echo840_MonkeyOCR/`
- **Required subdirectories**: 
  - `Recognition/` - Text recognition model
  - `Structure/` - Document structure detection
  - `Relation/` - Relationship prediction
- **Size**: ~3-4GB
- **Purpose**: Document structure analysis and OCR

### Nanonets OCR (nanonets/Nanonets-OCR-s)
- **Location**: `models/nanonets_Nanonets-OCR-s/`
- **Required files**: `config.json`, `tokenizer_config.json`, model files
- **Size**: ~6-7GB  
- **Purpose**: High-quality OCR with medical terminology support

### NuExtract (numind/NuExtract-2.0-8B)
- **Location**: `models/numind_NuExtract-2.0-8B/`
- **Required files**: `config.json`, `tokenizer_config.json`, model files
- **Size**: ~15-16GB
- **Purpose**: Structured data extraction from OCR text

## System Requirements

### Minimum Requirements
- **RAM**: 16GB (32GB recommended)
- **Storage**: 25GB free space for models
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher

### GPU Requirements (Optional but Recommended)
- **VRAM**: 8GB minimum (16GB recommended)
- **CUDA**: Compatible GPU with CUDA 11.8+
- **Driver**: Latest NVIDIA drivers

### For CPU-Only Operation
- **RAM**: 32GB minimum (64GB recommended)
- **CPU**: 8+ cores recommended
- Expect slower processing times (5-10x slower than GPU)

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository_url>
cd structured-extractor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Models
Models should be downloaded to the `models/` directory with the following structure:
```
models/
├── echo840_MonkeyOCR/
│   ├── Recognition/
│   ├── Structure/
│   ├── Relation/
│   └── README.md
├── nanonets_Nanonets-OCR-s/
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model files...
└── numind_NuExtract-2.0-8B/
    ├── config.json
    ├── tokenizer_config.json
    └── model files...
```

### 4. Verify Installation
```bash
python diagnose_models.py
```

## Troubleshooting

### Common Issues

#### 1. "Model not found" Errors
**Problem**: Models are not in the expected directory structure.

**Solution**:
- Ensure models are in `models/` directory
- Check that directory names match exactly:
  - `echo840_MonkeyOCR`
  - `nanonets_Nanonets-OCR-s` 
  - `numind_NuExtract-2.0-8B`

#### 2. "CUDA out of memory" Errors
**Problem**: GPU doesn't have enough VRAM.

**Solutions**:
- Force CPU operation by setting in config:
  ```yaml
  performance:
    force_cpu: true
  ```
- Reduce batch size in config:
  ```yaml
  performance:
    batch_size: 1
  ```
- Close other GPU-using applications

#### 3. "Invalid model structure" Errors
**Problem**: Model files are incomplete or corrupted.

**Solution**:
- Re-download the affected model
- Verify all required files are present
- Check file permissions

#### 4. Import Errors
**Problem**: Missing Python dependencies.

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

#### 5. Slow Performance
**Problem**: Models running on CPU or insufficient resources.

**Solutions**:
- Verify GPU is being used: Check diagnostic output
- Increase available RAM by closing other applications
- Consider using smaller batch sizes
- For CPU-only: Ensure you have sufficient RAM (32GB+)

### Performance Tuning

#### GPU Optimization
```yaml
performance:
  gpu_memory_fraction: 0.8  # Use 80% of GPU memory
  enable_mixed_precision: true  # Use FP16 for faster inference
  batch_size: 2  # Adjust based on VRAM
```

#### CPU Optimization
```yaml
performance:
  force_cpu: true
  max_workers: 4  # Match CPU cores
  batch_size: 1  # Lower batch size for CPU
```

### Diagnostic Commands

#### Check System Resources
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

#### Test Individual Models
```bash
# Test each model individually
python -c "
import asyncio
from src.processors.monkey_ocr_engine import MonkeyOCREngine
from src.core.config_manager import ConfigManager

async def test():
    config = ConfigManager()
    engine = MonkeyOCREngine(config)
    await engine.load_models()
    print('MonkeyOCR loaded successfully')

asyncio.run(test())
"
```

#### Memory Usage Monitoring
```python
import psutil
import torch

# RAM usage
memory = psutil.virtual_memory()
print(f"RAM: {memory.percent}% used")

# GPU memory (if available)
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB used")
```

## Configuration Options

### OCR Engine Selection
```yaml
ocr:
  ensemble:
    use_models: ["nanonets_ocr", "monkey_ocr"]  # Which models to use
    method: "best_confidence"  # Selection method
    weights:
      nanonets_ocr: 1.3
      monkey_ocr: 1.2
```

### Model Paths
```yaml
global:
  cache_dir: "models"  # Base directory for models

ocr:
  monkey_ocr:
    local_path: "models/echo840_MonkeyOCR"
  nanonets_ocr:
    local_path: "models/nanonets_Nanonets-OCR-s"

extraction:
  nuextract:
    local_path: "models/numind_NuExtract-2.0-8B"
```

## Support and Debugging

### Enable Debug Logging
```yaml
logging:
  level: "DEBUG"
  file: "logs/debug.log"
```

### Generate Diagnostic Report
```bash
python diagnose_models.py > diagnostic_report.txt
```

### Performance Benchmarking
```bash
python test_standalone_system.py
```

## Expected Processing Times

### GPU (RTX 3080/4080 class)
- **OCR per page**: 2-5 seconds
- **Extraction**: 1-3 seconds
- **Total per document**: 5-10 seconds

### CPU (8+ cores)
- **OCR per page**: 10-30 seconds
- **Extraction**: 5-15 seconds  
- **Total per document**: 30-60 seconds

### Memory Usage
- **MonkeyOCR**: ~2GB VRAM/RAM
- **Nanonets OCR**: ~4GB VRAM/RAM
- **NuExtract**: ~8GB VRAM/RAM
- **Total system**: ~16GB VRAM or 32GB RAM

## Success Indicators

When everything is working correctly, you should see:

1. **Diagnostic output**: All models show ✓ status
2. **Test results**: All tests pass
3. **Processing**: Documents are processed without errors
4. **Output**: Structured JSON with extracted medical data

Example successful extraction:
```json
{
  "patients": [
    {
      "patient_info": {
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": "1980-01-15"
      },
      "medical_codes": {
        "cpt_codes": [{"code": "99213", "description": "Office Visit"}],
        "icd10_codes": [{"code": "M54.5", "description": "Low back pain"}]
      },
      "financial_info": {
        "total_charges": 150.00
      }
    }
  ]
}
```
