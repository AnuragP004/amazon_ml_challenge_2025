# Quick Start Guide

This guide will help you set up and run the Amazon ML Challenge project.

## Prerequisites

- Python 3.8 or higher
- Git
- 8GB+ RAM (16GB recommended for training)
- CUDA-capable GPU (optional but recommended for training)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/amazon-ml-challenge.git
cd amazon-ml-challenge
```

### 2. Set Up Virtual Environment

**On Linux/MacOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Data

1. Download the dataset from [Amazon ML Challenge](https://www.hackerearth.com/challenges/competitive/amazon-ml-challenge/)
2. Place the files in the `data/` directory:
   - `train.csv`
   - `test.csv`
   - `images/` (folder containing product images)

### 5. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Usage

### Exploratory Data Analysis

```bash
jupyter notebook notebooks/eda.ipynb
```

### Training the Model

```bash
jupyter notebook notebooks/smart_pricing_multimodal.ipynb
```

Follow the cells in order to:
1. Load and preprocess data
2. Train the multimodal model
3. Evaluate performance
4. Save the best model

### Making Predictions

After training, use the inference script:

```bash
python src/inference.py --input data/test.csv --output predictions.csv
```

## Project Structure

```
amazon-ml-challenge/
├── data/              # Dataset files (gitignored)
├── models/            # Trained models (gitignored)
├── notebooks/         # Jupyter notebooks for EDA and training
├── src/               # Source code
│   ├── inference.py   # Inference script
│   └── utils.py       # Utility functions
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Common Issues

### CUDA Out of Memory
- Reduce batch size in training notebook
- Use CPU instead: set `device = 'cpu'`

### Missing Images
- Ensure images are in `data/images/` directory
- Check image paths in CSV files match actual filenames

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Next Steps

1. Run `notebooks/eda.ipynb` to understand the data
2. Run `notebooks/eda_preprocessing.ipynb` for feature engineering
3. Train your model with `notebooks/smart_pricing_multimodal.ipynb`
4. Generate predictions with `src/inference.py`

## Support

For questions or issues:
- Check existing [Issues](https://github.com/yourusername/amazon-ml-challenge/issues)
- Create a new issue with detailed description

## License

MIT License - see [LICENSE](LICENSE) file for details.
