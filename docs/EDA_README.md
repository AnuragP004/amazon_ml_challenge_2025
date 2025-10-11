# EDA & Feature Engineering Notebook

## Overview
This notebook performs comprehensive **Exploratory Data Analysis (EDA)** and **Feature Engineering** for the Amazon ML Challenge dataset.

## 📋 What This Notebook Does

### 1. **Data Exploration**
- Load and inspect training and test datasets
- Analyze missing values and data types
- Generate descriptive statistics

### 2. **Target Variable Analysis (Price)**
- Distribution analysis (histogram, box plot, Q-Q plot)
- Log-scale transformation analysis
- Identify outliers and percentiles
- Skewness and kurtosis analysis

### 3. **Text Analysis**
- Text length and word count statistics
- Correlation with price
- Word cloud visualization
- Bullet points and descriptions analysis

### 4. **Feature Engineering**
Extracts the following features from `catalog_content`:

| Feature | Description | Type |
|---------|-------------|------|
| `ipq` | Item Pack Quantity | Numerical |
| `unit_type` | Unit of measurement (Count, Weight, Volume, Length) | Categorical |
| `brand` | Extracted brand name | Categorical |
| `bullet_count` | Number of bullet points | Numerical |
| `has_description` | Whether product has description | Binary |
| `text_length` | Character count | Numerical |
| `word_count` | Word count | Numerical |
| `has_image` | Image availability | Binary |

### 5. **Feature Analysis**
- IPQ distribution and correlation with price
- Unit type distribution and price patterns
- Bullet points impact on pricing
- Image availability analysis

### 6. **Correlation Analysis**
- Heatmap of feature correlations
- Feature importance ranking
- Identify strongest predictors

### 7. **Visualizations**
Generates the following plots:
- Price distribution (raw and log-scale)
- Text length vs price
- IPQ analysis
- Unit type distribution
- Bullet points analysis
- Correlation matrix heatmap
- Word cloud
- Feature importance chart

### 8. **Data Preprocessing**
- Apply all feature engineering to test set
- Save preprocessed data with new features
- Generate summary report

## 🚀 How to Use

### 1. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook eda_preprocessing.ipynb
```

### 3. Run All Cells
Execute all cells sequentially from top to bottom.

## 📊 Output Files

### Preprocessed Data
- `../dataset/preprocessed/train_preprocessed.csv` - Training data with engineered features
- `../dataset/preprocessed/test_preprocessed.csv` - Test data with engineered features

### Visualizations
All plots are saved in the current directory:
- `price_distribution.png` - Price analysis plots
- `text_length_analysis.png` - Text length distributions
- `text_price_correlation.png` - Text features vs price
- `ipq_analysis.png` - IPQ feature analysis
- `unit_type_analysis.png` - Unit type distributions
- `bullet_description_analysis.png` - Bullet points analysis
- `correlation_matrix.png` - Feature correlation heatmap
- `wordcloud.png` - Most common words visualization
- `feature_importance.png` - Feature correlation with price

## 🔍 Key Insights

### Top Features for Price Prediction
1. **IPQ (Item Pack Quantity)** - Strongest numerical predictor
2. **Text Length** - Longer descriptions correlate with price
3. **Word Count** - Similar to text length
4. **Bullet Count** - Number of features listed
5. **Unit Type** - Different units have different pricing

### Data Characteristics
- **Price**: Right-skewed distribution (log transformation recommended)
- **Text**: Varies widely in length (mean ~1500 characters)
- **Images**: High availability (~95%+ in both train/test)
- **IPQ**: Most products are single items (IPQ=1), but packs exist

### Recommendations
1. Use **multimodal model** (text + images)
2. Apply **log transformation** to IPQ
3. **Normalize** text length features
4. Include **all engineered features** in the model
5. Use **MAE or Huber loss** for SMAPE optimization

## 📈 Feature Statistics

### Numerical Features
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| price | ~$30-50 | Variable | >$0 | >$1000 |
| ipq | ~2-5 | Variable | 1 | 100+ |
| text_length | ~1500 | Variable | 100 | 5000+ |
| word_count | ~250 | Variable | 20 | 800+ |
| bullet_count | ~4-5 | 1-2 | 0 | 8 |

### Categorical Features
- **Unit Type**: Count, Weight, Volume, Length, Unknown
- **Has Description**: Binary (0/1)
- **Has Image**: Binary (0/1)

## 🎯 Integration with Main Model

The preprocessed data can be directly used in the main training notebook:

```python
# Load preprocessed data
train_df = pd.read_csv('../dataset/preprocessed/train_preprocessed.csv')
test_df = pd.read_csv('../dataset/preprocessed/test_preprocessed.csv')

# Features are already extracted and ready to use
# Just add them to your model input alongside text and image features
```

## 💡 Tips

1. **Run this notebook first** before training the main model
2. **Review all visualizations** to understand data patterns
3. **Check the summary report** at the end for key statistics
4. **Use preprocessed data** for faster model training
5. **Experiment with feature combinations** in your model

## 🔧 Customization

You can easily add more features by:
1. Creating new extraction functions
2. Adding them to the feature engineering section
3. Updating the correlation analysis

Example:
```python
def extract_custom_feature(text):
    # Your custom logic here
    return feature_value

train_df['custom_feature'] = train_df['catalog_content'].apply(extract_custom_feature)
```

## 📚 Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- wordcloud >= 1.9.0
- scipy >= 1.10.0

## ⏱️ Execution Time

- **With full dataset**: ~5-10 minutes
- **With sampled data**: ~1-2 minutes

Word cloud generation uses only first 1000 samples for speed.

## 🎨 Visualization Examples

The notebook generates beautiful, publication-ready visualizations:
- Clean, professional style
- Clear labels and titles
- Saved at high resolution (150 DPI)
- Color-coded for easy interpretation

## 📝 Notes

- All features are extracted using regex patterns
- Missing values are handled with sensible defaults
- The notebook is self-contained and documented
- All intermediate steps are explained

## 🤝 Contributing

Feel free to:
- Add more feature extraction functions
- Create additional visualizations
- Improve existing analysis
- Share insights discovered

---

**Happy Exploring! 🚀📊**
