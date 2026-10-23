# Credit Card Fraud Detection Using Apache Spark

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using **Apache Spark and Machine Learning**.

The project demonstrates how PySpark can be used for data preprocessing, machine learning, model evaluation, and fraud prediction. The final results are also visualized using **Power BI**.

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle, provided by the Machine Learning Group – ULB.

- **Original transactions:** 284,807
- **Features:** 31
- **Fraudulent transactions:** 492
- **Genuine transactions:** 284,315
- **Target column:** `Class`
  - `0` = Genuine
  - `1` = Fraud

The dataset contains `Time`, `Amount`, `Class`, and anonymized PCA-transformed features (`V1`–`V28`).

> **Note:** The original `creditcard.csv` dataset is not included in this repository because of its large file size. It can be downloaded separately from Kaggle.

## 🛠️ Technologies Used

- Apache Spark
- PySpark
- Spark MLlib
- Python
- Pandas
- Power BI
- VS Code
- Java 21

## 🔄 Project Workflow

```text
Credit Card Dataset
        ↓
Data Loading using PySpark
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Preparation
        ↓
Train/Test Split
        ↓
Machine Learning Models
        ↓
Model Evaluation
        ↓
Fraud Prediction
        ↓
Power BI Dashboard
```

## 🧹 Data Preprocessing

During preprocessing:

- Duplicate transactions were identified and removed.
- The dataset was prepared for machine learning.
- Features were assembled for Spark ML.
- The data was divided into training and testing sets.

After removing duplicates:

- **283,726 records** remained.
- **473 fraud records** remained.
- **283,253 genuine records** remained.
- **227,168 records** were used for training.
- **56,558 records** were used for testing.

## 🤖 Machine Learning Models

Two classification models were implemented using **Spark ML**:

### 1. Logistic Regression

Used as a classification model to identify fraudulent transactions.

### 2. Random Forest

Used as a tree-based classification model for detecting fraudulent transactions.

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 99.91% | 87.10% | 56.84% | 68.79% | 96.61% | 69.43% |
| Random Forest | 99.93% | 99.92% | 99.93% | 99.93% | 96.47% | 77.09% |

Based on the obtained results, **Random Forest performed better overall** than Logistic Regression.

## 📁 Repository Files

- `fraud_detection.py` — PySpark data processing, machine learning, and evaluation code
- `fraud_predictions.csv` — fraud prediction results
- `model_comparison.csv` — comparison of the machine learning models
- `Credit_Card_Fraud_Dashboard.pbix` — Power BI dashboard
- `.gitignore` — specifies files excluded from Git

## 📊 Power BI Dashboard

The Power BI dashboard presents:

- Total transactions
- Fraudulent transactions
- Fraud detection rate
- Genuine vs Fraud transaction distribution
- Model comparison
- F1 Score comparison
- PR-AUC comparison
- Accuracy comparison
- Average fraud probability

## 🎯 Objective

The main objective of this project is to demonstrate the use of **Apache Spark for large-scale data processing and machine learning-based credit card fraud detection**, followed by visualization of the analytical results using Power BI.



## 📌 Dataset Source

Credit Card Fraud Detection dataset by Machine Learning Group – ULB, available through Kaggle.
