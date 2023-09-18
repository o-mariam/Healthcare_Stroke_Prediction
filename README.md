# Healthcare Stroke Prediction & Spam Email Classification

This repository contains two distinct tasks related to machine learning:

1. Stroke Prediction using a Random Forest Model.
2. Spam Email Classification using a Neural Network.

---

## Task 1: Stroke Prediction

### Dataset

- `healthcare-dataset-stroke-data.csv`: Contains patient information. It also includes a column indicating the existence of a stroke episode (1 for stroke, 0 otherwise).

### Steps to Follow

#### 1A. Dataset Analysis & Graphical Representation

- Start by analyzing the dataset.
- Create visualizations to understand the distributions and relationships between different variables.

#### 1B. Handle Missing Values

Before feeding the data to a Random Forest model, it's essential to handle missing values. Utilize the following methods:

- **Remove a column**: If a column has too many missing values and isn't significant, consider removing it.
  
- **Fill using Mean**: Fill the missing values with the mean value of that column.
  
- **Linear Regression**: Use linear regression to predict and fill the missing values. This can be done by treating the column with missing values as the dependent variable and other columns as independent variables.
  
- **k-Nearest Neighbors (k-NN)**: Use k-NN to impute missing values.

#### Model Training

- Split the dataset into a 75%-25% training-test set.
- Train a Random Forest model on the training set and evaluate its performance on the test set.

---

## Task 2: Spam Email Classification

### Dataset

- `spam_or_not_spam.csv`: Contains two columns. The first column holds the text from various emails. The second column indicates whether the email was spam (1 for spam, 0 otherwise).

### Script

- `neural.py`: This script focuses on using a neural network to predict whether an email is spam or not.

### Steps to Follow

- Start by preprocessing the email texts. This includes tokenizing, removing stop words, and converting texts to sequences.
- Build a neural network model to predict the spam label based on the email text.
- Split the dataset, train the model, and evaluate its performance.

---

## Installation & Setup

1. Clone this repository.
2. Set up a Python environment (preferably using `virtualenv` or `conda`).
3. Install the required libraries using `pip install -r requirements.txt`.

## Execution

For Task 1:

```bash
python stroke_prediction.py
```

For Task 2:

```bash
python neural.py
```

---
