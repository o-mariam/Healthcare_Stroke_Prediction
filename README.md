# Stroke Prediction using Random Forest

## Overview
This project focuses on predicting the likelihood of a stroke based on patient health data using a **Random Forest model**. The dataset includes various health indicators such as age, hypertension, heart disease, and more.

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
   git clone <repository-url>
   cd stroke-prediction
2. Set up a Python virtual environment (optional but recommended):

  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
### 3.Execution
  Run the following command to train and test the model:
      python stroke_prediction.py

## Results & Performance
The model is evaluated using accuracy, confusion matrix, and classification reports.

Key performance metrics and visualizations are generated for better understanding.

## Technologies Used
Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn
  
