# Smoking Prediction Using ML/DL

A comprehensive Machine Learning and Deep Learning project for predicting smoking status and health classification using biometric data.

---

## Overview

This project implements various ML and DL techniques to:
- Predict smoking status from health biometrics
- Predict age using regression models
- Classify health status using clustering
- Recognize ASL (American Sign Language) signs using CNN

---

## Dataset

**Smoking Dataset:** 55,692 samples with 26 health biometric features including gender, age, height, weight, blood pressure, cholesterol, hemoglobin, and more. Target variable is binary smoking status (0/1).

**ASL Dataset:** 29 classes (A-Z + del, nothing, space) of RGB images for sign language recognition.

---

## Tasks Completed

**1. Data Preparation:** Missing value analysis, categorical encoding, feature scaling (StandardScaler), data balancing (SMOTE)

**2. Classification (Smoking):** KNN with optimal K selection, Logistic Regression, Gaussian Naive Bayes with confusion matrices and metrics

**3. Regression (Age):** Linear Regression, Random Forest, Gradient Boosting

**4. Clustering (Health Status):** K-Means with K=3 producing three health levels (Bon, Normal, Faible)

**5. Deep Learning (MLP):** Neural network with Keras testing 5, 10, 20 neurons using tanh and sigmoid activations

**6. Deep Learning (CNN):** Simple and Deep CNN architectures for ASL recognition with data augmentation and error analysis

---

## Results

**Classification:** KNN (K=1) achieved 78% accuracy with F1-score of 0.72. Logistic Regression reached 91% recall. Naive Bayes achieved 70% accuracy.

**Regression:** Random Forest performed best with R²=0.616 and MAE=5.5 years.

**MLP:** All configurations (5, 10, 20 neurons) achieved approximately 76% accuracy.

**Key Insights:** Men smoke 13x more than women (55.4% vs 4.2%). Top predictors are gender, hemoglobin, Gtp, triglyceride, and height.

---

## Technologies Used

Python 3.8+, pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, TensorFlow, Keras

---

## Installation

```bash
git clone https://github.com/mr-taha-saqib/ml-dl-smoking-health-analysis.git
cd smoking-prediction-ml-dl
pip install -r requirements.txt
```

---

## Usage

Run locally with `jupyter notebook exam_ml_dl_complete.ipynb` or upload to Google Colab with the dataset.

For ASL section, update the ZIP_PATH variable to your dataset location.

---

## License

MIT License
