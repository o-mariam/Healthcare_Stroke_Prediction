# Stroke Prediction using Random Forest

## 🌐 Overview
This project predicts whether a patient has experienced a stroke based on health-related features using a Random Forest classifier. The complete pipeline includes:

1. **Data Preprocessing**:
   - Handling missing values in the BMI column using:
     - Row dropping
     - Mean imputation
     - K-Nearest Neighbors (KNN) imputation
   - One-hot encoding for categorical features (e.g., smoking_status)

2. **Feature Engineering**:
   - Addressing class imbalance with SMOTE
   - Feature selection (all features used except stroke and BMI)

3. **Model Training & Evaluation**:
   - Random Forest classifier implementation
   - Performance evaluation using:
     - Confusion matrices
     - Classification reports (precision, recall, F1-score)

## Dataset
- **File**: `healthcare-dataset-stroke-data.csv`
- **Description**: Contains patient information with risk factors and stroke indicator (1 = stroke, 0 = no stroke)

##  Implementation Steps

### 1. Data Preprocessing
- Perform Exploratory Data Analysis (EDA)
- Handle missing values through:
  - Column removal (if excessive missing values)
  - Mean imputation
  - Linear regression imputation
  - k-NN imputation

### 2. Model Building
- 75/25 train-test split
- Random Forest classifier training
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/o-mariam/Healthcare_Stroke_Prediction
```

# Stroke Prediction with Random Forest

## 📊 Results & Performance
- **Accuracy**: 96-97% with Random Forest
- **Missing Data Handling**:
  - ✅ Mean Imputation
  - ✅ KNN Imputation
  - ✅ Row Dropping
- **Class Balance**: SMOTE applied successfully
- **Key Takeaway**: Preprocessing and tuning are critical

## 🛠 Technologies
```python
# Core Stack
"Python" : {
  "Data Processing": ["Pandas", "NumPy"],
  "ML Framework": "Scikit-Learn",
  "Visualization": ["Matplotlib", "Seaborn"]
}
