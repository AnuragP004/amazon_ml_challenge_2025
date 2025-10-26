# Source Code

This directory contains the main source code for the project.

## Files

### `inference.py`
Production-ready inference script for making predictions on new data.

**Usage:**
```bash
python src/inference.py --input data/test.csv --output predictions.csv
```

**Features:**
- Batch processing for efficiency
- Support for both text and image inputs
- Automatic model loading
- Progress tracking

### `utils.py`
Utility functions used throughout the project.

**Contains:**
- Data loading and preprocessing functions
- Image processing utilities
- Text tokenization helpers
- SMAPE calculation
- Visualization functions

## Requirements

All dependencies are listed in `requirements.txt` at the project root.
