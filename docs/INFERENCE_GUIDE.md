# Inference Guide - Using the Trained Model

This guide explains how to use the trained multimodal model for price prediction.

## Overview

The `sample_code.py` has been updated with the complete multimodal approach:
- **Image Features**: MobileNetV2 CNN
- **Text Features**: DistilBERT transformer
- **IPQ Feature**: Item Pack Quantity extracted from text
- **Fusion**: Neural network combining all features

## Prerequisites

1. **Trained Model**: Ensure you have trained the model and saved it as `src/best_model.pth`
   - Train using: `smart_pricing_multimodal.ipynb`
   
2. **Downloaded Images**: All product images should be in `dataset/images/`
   - Use the `download_images()` function from `utils.py`
   
3. **Dependencies**: Install required packages
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run the Complete Script

```bash
cd student_resource
python sample_code.py
```

This will:
1. Load the trained model from `src/best_model.pth`
2. Read `dataset/test.csv`
3. Generate predictions for all test samples
4. Save results to `dataset/test_out.csv`

### Option 2: Use the Predictor Function Directly

```python
from sample_code import predictor, load_model_components

# Load model once
load_model_components()

# Predict for a single sample
sample_id = "ABC123"
catalog_content = "Item Pack Quantity: 2 | Brand: Example | Description: Great product..."
image_link = "https://example.com/image.jpg"

price = predictor(sample_id, catalog_content, image_link)
print(f"Predicted price: ${price:.2f}")
```

## File Structure

```
amazon-ml-challenge/
├── student_resource/
│   ├── sample_code.py           # Updated inference script
│   ├── dataset/
│   │   ├── test.csv            # Test data
│   │   ├── test_out.csv        # Output predictions
│   │   └── images/             # Downloaded product images
│   └── src/
│       ├── best_model.pth      # Trained model weights
│       ├── smart_pricing_multimodal.ipynb  # Training notebook
│       └── eda_preprocessing.ipynb         # EDA notebook
├── requirements.txt
└── INFERENCE_GUIDE.md          # This file
```

## Model Components

### 1. Image Processing
- **Model**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 224x224 pixels
- **Normalization**: ImageNet mean and std
- **Missing Images**: Black placeholder image

### 2. Text Processing
- **Model**: DistilBERT (distilbert-base-uncased)
- **Max Length**: 128 tokens
- **Processing**: Lowercase, whitespace normalization
- **Output**: CLS token embeddings (768-dim → 64-dim)

### 3. Feature Engineering
- **IPQ (Item Pack Quantity)**:
  - Extracted using regex patterns
  - Log-transformed: `log(IPQ + 1)`
  - Normalized using StandardScaler
  - Default value: 1.0 if not found

### 4. Fusion Network
- Concatenates: [Image (64-dim), Text (64-dim), IPQ (1-dim)]
- Architecture: 
  ```
  Input (129-dim) → 64 → ReLU → Dropout → 32 → ReLU → 1 (price)
  ```

## Output Format

The output file `test_out.csv` contains:
```csv
sample_id,price
sample1,25.99
sample2,149.50
sample3,8.75
...
```

## Expected Performance

### Speed
- **With GPU**: ~0.01-0.05 seconds per sample
- **With CPU**: ~0.1-0.5 seconds per sample
- **Full Test Set**: 
  - GPU: 2-10 minutes for 100K samples
  - CPU: 20-60 minutes for 100K samples

### Accuracy
- Target metric: **SMAPE** (Symmetric Mean Absolute Percentage Error)
- Expected SMAPE on validation: 20-30%
- The model learns from patterns in:
  - Product descriptions and keywords
  - Visual appearance of products
  - Pack quantity and bulk pricing
  - Product categories and brands

## Troubleshooting

### Issue: Model file not found
```
⚠️  Warning: No trained model found at src/best_model.pth
```
**Solution**: Train the model first using `smart_pricing_multimodal.ipynb`

### Issue: Images not found
**Solution**: Download images using:
```python
from utils import download_images
import pandas as pd

df = pd.read_csv('dataset/test.csv')
image_links = df['image_link'].dropna().unique().tolist()
download_images(image_links, 'dataset/images/')
```

### Issue: CUDA out of memory
**Solution**: 
1. Model uses frozen pretrained weights (minimal GPU memory)
2. If still issues, reduce batch size in data loading
3. Or run on CPU (slower but works)

### Issue: ImportError for packages
**Solution**: 
```bash
pip install torch torchvision transformers pillow scikit-learn pandas numpy
```

## Advanced Usage

### Batch Prediction with Progress Bar

```python
import pandas as pd
from tqdm import tqdm
from sample_code import predictor, load_model_components

# Load model once
load_model_components()

# Read data
test_df = pd.read_csv('dataset/test.csv')

# Predict with progress bar
predictions = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    price = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
    predictions.append(price)

test_df['price'] = predictions
test_df[['sample_id', 'price']].to_csv('dataset/test_out.csv', index=False)
```

### Custom Scaler Fitting

For more accurate predictions, fit the scaler on training data:

```python
import pandas as pd
import numpy as np
from sample_code import extract_ipq, SCALER, load_model_components

# Load model
load_model_components()

# Read training data
train_df = pd.read_csv('dataset/train.csv')

# Extract and normalize IPQ
train_df['ipq'] = train_df['catalog_content'].apply(extract_ipq)
train_df['ipq_log'] = np.log1p(train_df['ipq'])

# Fit scaler
SCALER.fit(train_df[['ipq_log']])

print("✓ Scaler fitted on training data")

# Now predictions will be more accurate
```

## Performance Optimization Tips

1. **Batch Processing**: Process multiple samples at once (modify code to support batch inference)
2. **GPU Usage**: Ensure CUDA is available with `torch.cuda.is_available()`
3. **Image Caching**: Images are loaded on-demand; consider preloading for speed
4. **Mixed Precision**: Already enabled in training; inference uses FP32 by default
5. **Model Quantization**: For deployment, consider INT8 quantization

## Model Architecture Summary

```
Input:
  ├─ Image (224x224x3)
  ├─ Text (128 tokens)
  └─ IPQ (1 float)

Processing:
  ├─ Image → MobileNetV2 → 64-dim embedding
  ├─ Text → DistilBERT → 64-dim embedding  
  └─ IPQ → log transform → normalize → 1-dim

Fusion:
  └─ Concat [64 + 64 + 1 = 129] → MLP → Price

Output:
  └─ Price (float)
```

## Citation

If you use this approach, please cite:
- **DistilBERT**: Sanh et al., 2019
- **MobileNetV2**: Sandler et al., 2018
- **PyTorch**: Paszke et al., 2019

## Support

For issues or questions:
1. Check the notebook: `smart_pricing_multimodal.ipynb`
2. Review EDA insights: `eda_preprocessing.ipynb`
3. Verify all dependencies are installed: `pip install -r requirements.txt`

---

**Happy Predicting! 🚀**
