import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler

# Global variables for model components
MODEL = None
TOKENIZER = None
SCALER = None
TRANSFORM = None
DEVICE = None
IMAGE_FOLDER = 'dataset/images/'
IMG_SIZE = 224
MAX_TEXT_LENGTH = 128
IMAGE_EMBED_DIM = 64
TEXT_EMBED_DIM = 64


class MultimodalPricePredictor(nn.Module):
    """
    Multimodal model for price prediction using MobileNetV2 (images), 
    DistilBERT (text), and IPQ feature.
    """
    
    def __init__(self, image_embed_dim=64, text_embed_dim=64, dropout=0.3):
        super(MultimodalPricePredictor, self).__init__()
        
        # Image Tower - MobileNetV2
        self.image_model = models.mobilenet_v2(pretrained=True)
        for param in self.image_model.parameters():
            param.requires_grad = False
        
        num_features = self.image_model.classifier[1].in_features
        self.image_model.classifier = nn.Identity()
        
        self.image_projection = nn.Sequential(
            nn.Linear(num_features, image_embed_dim),
            nn.ReLU()
        )
        
        # Text Tower - DistilBERT
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.text_projection = nn.Sequential(
            nn.Linear(768, text_embed_dim),
            nn.ReLU()
        )
        
        # Fusion Head
        fusion_dim = image_embed_dim + text_embed_dim + 1  # +1 for IPQ
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, image, input_ids, attention_mask, ipq):
        # Image features
        image_features = self.image_model(image)
        image_embed = self.image_projection(image_features)
        
        # Text features
        with torch.no_grad():
            text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # CLS token
        text_embed = self.text_projection(text_features)
        
        # Concatenate all features
        ipq = ipq.unsqueeze(1) if ipq.dim() == 1 else ipq
        combined = torch.cat([image_embed, text_embed, ipq], dim=1)
        
        # Predict price
        price = self.fusion_head(combined)
        return price.squeeze(1)


def extract_ipq(text):
    """Extract Item Pack Quantity from text."""
    if pd.isna(text) or not isinstance(text, str):
        return 1.0
    
    patterns = [
        r'Item Pack Quantity[:\s]+([0-9]+(?:\.[0-9]+)?)',
        r'Pack of ([0-9]+)',
        r'([0-9]+)[\\s-]*Pack',
        r'Value[:\s]+([0-9]+(?:\.[0-9]+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    return 1.0


def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_model_components():
    """Load model, tokenizer, scaler, and transforms."""
    global MODEL, TOKENIZER, SCALER, TRANSFORM, DEVICE
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load tokenizer
    TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load model
    MODEL = MultimodalPricePredictor(
        image_embed_dim=IMAGE_EMBED_DIM,
        text_embed_dim=TEXT_EMBED_DIM,
        dropout=0.3
    )
    
    # Load trained weights if available
    model_path = 'src/best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print(f"⚠️  Warning: No trained model found at {model_path}")
        print("   Using randomly initialized weights")
    
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Initialize scaler (will be fitted on training data)
    SCALER = StandardScaler()
    
    # Image transforms
    TRANSFORM = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ All model components loaded")


def predictor(sample_id, catalog_content, image_link):
    '''
    Predict product price using multimodal ML model.
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    global MODEL, TOKENIZER, SCALER, TRANSFORM, DEVICE
    
    # Load components if not already loaded
    if MODEL is None:
        load_model_components()
    
    # Extract IPQ feature
    ipq = extract_ipq(catalog_content)
    ipq_log = np.log1p(ipq)
    ipq_normalized = (ipq_log - 0.5) / 1.0  # Approximate normalization
    
    # Clean text
    text = clean_text(catalog_content)
    
    # Tokenize text
    encoded = TOKENIZER(
        text,
        max_length=MAX_TEXT_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    
    # Load image
    default_image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
    
    if pd.notna(image_link) and isinstance(image_link, str) and image_link.strip() != '':
        filename = Path(image_link).name
        image_path = os.path.join(IMAGE_FOLDER, filename)
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                image = default_image
        except Exception:
            image = default_image
    else:
        image = default_image
    
    # Transform image
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    # Prepare IPQ tensor
    ipq_tensor = torch.tensor([ipq_normalized], dtype=torch.float32).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        prediction = MODEL(image_tensor, input_ids, attention_mask, ipq_tensor)
        price = prediction.item()
    
    # Ensure positive price
    price = max(0.01, price)
    
    return round(price, 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    print("="*60)
    print("AMAZON ML CHALLENGE - MULTIMODAL PRICE PREDICTION")
    print("="*60)
    
    # Read test data
    print("\nLoading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Test samples: {len(test)}")
    
    # Load model components once
    print("\nInitializing model components...")
    load_model_components()
    
    # Apply predictor function to each row
    print("\nGenerating predictions...")
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n✓ Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"\nSample predictions:")
    print(output_df.head(10))
    
    # Statistics
    print(f"\nPrediction Statistics:")
    print(f"  Min price: ${output_df['price'].min():.2f}")
    print(f"  Max price: ${output_df['price'].max():.2f}")
    print(f"  Mean price: ${output_df['price'].mean():.2f}")
    print(f"  Median price: ${output_df['price'].median():.2f}")
    
    print("\n" + "="*60)
    print("✓ PREDICTION COMPLETE")
    print("="*60)

