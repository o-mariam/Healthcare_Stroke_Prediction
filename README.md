# Stroke Prediction using Random Forest

## Overview
This project aims to predict whether a patient has experienced a stroke based on health-related features. A Random Forest classifier is used for classification. The pipeline includes the following steps:

1. Data Preprocessing:
    Handling Missing Data: Missing values in the bmi column are handled using different techniques:
    
    Dropping rows with missing values.
    
    Imputing missing values using the mean of the column.
    
    Imputing using K-Nearest Neighbors (KNN) for better accuracy.
    
    One-Hot Encoding: Categorical features, such as smoking_status, are converted into numerical values using one-hot encoding.

2. Feature Engineering:
    Class Imbalance: SMOTE is applied to generate synthetic samples for the minority class (stroke cases), addressing the class imbalance.

    Feature Selection: All features are used except for the target variable (stroke) and bmi.

3. Model Training & Evaluation:
    Model: A Random Forest classifier is trained on the processed data.
    
    Evaluation: The model is evaluated using confusion matrices and classification reports, with performance metrics like precision, recall,        and F1-score.

4. Key Steps:
    Data Cleaning: Missing values are imputed (using mean, KNN, or dropped).
    
    SMOTE: Applied to handle class imbalance.
    
    Modeling: A Random Forest classifier is trained and evaluated.

5. Final Outcome:
    The project provides a complete pipeline for stroke prediction, including data preprocessing, model training, and performance evaluation.

## Dataset
- **File:** `healthcare-dataset-stroke-data.csv`
- **Description:** This dataset contains patient information, including risk factors and a column indicating whether the patient had a stroke (1 for stroke, 0 otherwise).

## Steps to Follow

### 1. Data Preprocessing
- Perform **Exploratory Data Analysis (EDA)** to understand the dataset.
- Handle missing values using different techniques:
  - Remove columns with excessive missing values.
  - Fill missing values with mean, linear regression, or k-NN imputation.

### 2. Model Building
- Split the dataset into **75% training** and **25% testing**.
- Train a **Random Forest classifier** to predict stroke occurrences.
- Evaluate model performance using accuracy, precision, recall, and F1-score.

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/o-mariam/Healthcare_Stroke_Prediction
   cd stroke-prediction
2. Set up a Python virtual environment (optional but recommended):

  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
### 3.Execution
  Run the following command to train and test the model:
      python stroke_prediction.py

## Results & Performance
 - The Random Forest model is highly effective in predicting stroke occurrences with an impressive accuracy rate of 96-97%.
  
 - The handling of missing data using different imputation methods (mean, KNN, and dropping) did not drastically affect the performance, indicating that the model is robust to various preprocessing strategies.
  
 - Class imbalance was successfully addressed using SMOTE, which helped improve model performance in predicting the minority class (stroke cases).
  
 - This project highlights the importance of data preprocessing and model tuning in achieving high accuracy in real-world predictive tasks.


## Technologies Used
Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn
  
