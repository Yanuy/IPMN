# Anti-Money Laundering Feature Engineering Library

View the details: https://yanuy.github.io/IPMN1/
A comprehensive machine learning system for detecting financial fraud and money laundering in transaction data. This system implements advanced feature engineering techniques, graph-based analysis, and multiple classification algorithms to identify fraudulent patterns in financial transactions.

## 🚀 Features

### Core Capabilities
- **Multi-task Learning**: Supports both binary fraud detection and multi-class laundering type classification
- **Advanced Feature Engineering**: Implements sliding window algorithms for temporal features without data leakage
- **Graph-based Analysis**: Calculates in-degree, out-degree, and network connectivity features
- **Temporal Pattern Detection**: Extracts time-based patterns and cyclical features
- **Memory Optimization**: Efficient processing of large datasets with chunked loading
- **Comprehensive Evaluation**: Detailed model performance analysis with visualizations

### Key Technical Features
- **Sliding Window Algorithm**: Leak-proof temporal feature computation
- **Imbalanced Data Handling**: SMOTE oversampling and class weight balancing
- **Feature Caching**: Persistent storage of computed features for faster iterations
- **Multiple ML Algorithms**: XGBoost, LightGBM, Random Forest, Logistic Regression
- **Cross-border Detection**: Automatic identification of international transactions
- **Risk Scoring**: Rule-based risk assessment integration

## 🗂️ Output Files

The system generates several output directories:

```
project/
├── eda_plots/              # Exploratory data analysis visualizations
│   ├── basic_statistics.png
│   ├── temporal_patterns.png
│   ├── amount_distribution.png
│   └── correlation_analysis.png
├── evaluation_plots/       # Model evaluation visualizations
│   ├── binary_roc_curves.png
│   └── binary_confusion_matrices.png
├── enhanced_evaluation/    # Advanced evaluation metrics
├── models/                 # Saved model files
│   ├── binary_XGBoost.joblib
│   ├── multiclass_XGBoost_Multi.joblib
│   ├── encoders.joblib
│   └── scalers.joblib
└── features_cache.csv      # Cached features for reuse
```

## 📋 Requirements

```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```python
from fraud_detection_system import FraudDetectionSystem
system = FraudDetectionSystem()
print("Installation successful!")
```

## 📊 Data Format

The system expects transaction data in CSV format with the following required columns:

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Transaction date |
| `Time` | time | Transaction time |
| `Amount` | float | Transaction amount |
| `Sender_account` | string | Sender account identifier |
| `Receiver_account` | string | Receiver account identifier |
| `Sender_bank_location` | string | Sender bank location |
| `Receiver_bank_location` | string | Receiver bank location |
| `Payment_currency` | string | Payment currency code |
| `Received_currency` | string | Received currency code |
| `Payment_type` | string | Type of payment method |
| `Is_laundering` | int | Binary fraud label (0/1) |
| `Laundering_type` | string | Specific type of laundering |

### Sample Data Structure
```csv
Date,Time,Amount,Sender_account,Receiver_account,Sender_bank_location,Receiver_bank_location,Payment_currency,Received_currency,Payment_type,Is_laundering,Laundering_type
2023-01-01,09:30:00,1500.50,ACC001,ACC002,USA,USA,USD,USD,Wire Transfer,0,Normal Transaction
2023-01-01,14:22:15,25000.00,ACC003,ACC004,USA,CHE,USD,CHF,Cash Deposit,1,Structuring
```

## 🚀 Quick Start

### Basic Usage
```python
from fraud_detection_system import FraudDetectionSystem

# Initialize the system
fraud_system = FraudDetectionSystem(
    data_path="your_data.csv",
    feature_cache_path="features_cache.csv"
)

# Run the complete pipeline
fraud_system.run_full_pipeline(
    data_path="SAML-D.csv",
    sample_ratio=1.0,  # Use full dataset
    force_recreate_features=False
)
```

### Step-by-step Execution
```python
# 1. Load and preprocess data
fraud_system.load_data(sample_ratio=0.1)  # Use 10% for testing

# 2. Create or load features
fraud_system.load_or_create_features(force_recreate=False)

# 3. Perform exploratory data analysis
fraud_system.exploratory_data_analysis()

# 4. Prepare features for modeling
fraud_system.prepare_features_for_modeling()

# 5. Split data by time
fraud_system.split_data_by_month(train_ratio=0.7)

# 6. Train models
fraud_system.train_models(task_mode='both')  # 'binary', 'multiclass', or 'both'

# 7. Evaluate models
fraud_system.evaluate_models()

# 8. Save trained models
fraud_system.save_models(path="models/")
```

## 🔧 Advanced Configuration

### Sliding Window Parameters
```python
# Configure sliding window for temporal features
fraud_system.set_window_parameters(
    window_days=30,  # Look back 30 days
    step_days=7      # Step forward 7 days
)
```

### Model Selection
```python
# Train only specific models
fraud_system.use_all_binary_models = True  # Enable all binary models
fraud_system.train_models(task_mode='binary')
```

### Feature Engineering Options
```python
# Create advanced degree features
fraud_system.create_advanced_degree_features()

# Get feature statistics
feature_summary = fraud_system.get_feature_summary()
degree_stats = fraud_system.get_degree_statistics()
```

## 📈 Pipeline Components

### 1. Data Loading & Preprocessing
- **Chunked Loading**: Efficiently handles large CSV files
- **Memory Optimization**: Smart sampling while preserving all fraud cases
- **Data Validation**: Automatic data type conversion and cleaning

### 2. Feature Engineering

#### Basic Features
- Amount transformations (log, sqrt, binning)
- Cross-border transaction detection
- Currency mismatch identification
- Round amount pattern detection

#### Temporal Features
- Cyclical time encoding (hour, day, month)
- Business hour classification
- Weekend/weekday patterns
- Monthly and quarterly trends

#### Graph Features (Sliding Window)
- **Sender Statistics**: Send amount, count, frequency
- **Receiver Statistics**: Receive amount, count, frequency
- **Degree Features**: In-degree, out-degree calculations
- **Pair Statistics**: Account-to-account transaction patterns
- **Activity Metrics**: Total activity counts per account

#### Risk Scoring
- Rule-based risk assessment
- Pattern-based anomaly detection
- Behavioral risk indicators

### 3. Model Training
- **Binary Classification**: Fraud vs. Normal transactions
- **Multi-class Classification**: Specific laundering type identification
- **Data Leakage Prevention**: Strict temporal validation
- **Class Imbalance Handling**: SMOTE and class weighting

### 4. Model Evaluation
- ROC/AUC analysis
- Precision-Recall curves
- Confusion matrices
- Feature importance analysis
- Cross-validation metrics

## 🤖 Supported Models

| Model | Type | Key Features |
|-------|------|--------------|
| **XGBoost** | Gradient Boosting | Built-in class weighting, high performance |
| **LightGBM** | Gradient Boosting | Memory efficient, fast training |
| **Random Forest** | Ensemble | Good baseline, interpretable |
| **Logistic Regression** | Linear | Fast, interpretable, good with SMOTE |

## 📊 Evaluation Metrics

### Binary Classification
- **AUC-ROC**: Area under ROC curve
- **Precision/Recall**: For imbalanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/False positive and negative analysis

### Multi-class Classification
- **Accuracy**: Overall classification accuracy
- **Fraud Detection Rate**: Binary fraud detection within multi-class
- **Class-specific Metrics**: Per-class precision and recall

## 🔮 Prediction

### Using Trained Models
```python
# Load trained models
fraud_system.load_models(path="models/")

# Make predictions on new data
new_data = pd.read_csv("new_transactions.csv")
predictions = fraud_system.predict(new_data, model_name='XGBoost')

# Access results
is_fraud = predictions['is_laundering_prediction']
fraud_probability = predictions['is_laundering_probability']
laundering_type = predictions['laundering_type_prediction']
```

## ⚙️ Configuration Options

### Memory Management
```python
# For large datasets
fraud_system = FraudDetectionSystem(
    data_path="large_dataset.csv",
    feature_cache_path="features_cache.csv"
)

# Use sampling for initial exploration
fraud_system.load_data(sample_ratio=0.1, chunk_size=50000)
```

### Feature Engineering
```python
# Customize sliding window
fraud_system.window_days = 60  # 60-day lookback
fraud_system.step_days = 14    # 14-day steps

# Enable all model types
fraud_system.use_all_binary_models = True
```

## 🐛 Troubleshooting

### Common Issues

**Memory Errors**
```python
# Reduce sample ratio
fraud_system.load_data(sample_ratio=0.05)

# Use smaller chunk size
fraud_system.load_data(chunk_size=5000)
```

**Feature Cache Issues**
```python
# Force feature recreation
fraud_system.load_or_create_features(force_recreate=True)
```

**Model Training Errors**
```python
# Train individual task types
fraud_system.train_models(task_mode='binary')
fraud_system.train_models(task_mode='multiclass')
```

## 📝 Performance Tips

1. **Use Feature Caching**: Save time on repeated runs
2. **Start with Sampling**: Test pipeline with small data subset
3. **Monitor Memory Usage**: Use task manager during large dataset processing
4. **Parallel Processing**: Enable n_jobs=-1 for compatible models
5. **Feature Selection**: Remove low-importance features for faster training


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub

## 🚀 Quick Example

```python
# Complete fraud detection in 5 lines
from fraud_detection_system import FraudDetectionSystem

system = FraudDetectionSystem()
system.run_full_pipeline("your_data.csv", sample_ratio=0.1)
print("Fraud detection system ready!")
```
