# Fake-Review-Detection-with-Machine-Learning

This repository contains four models to detect fake product reviews using traditional machine learning and deep learning approaches: Logistic Regression, Multi-Layer Perceptron (MLP), Naïve Bayes, and BERT.

## File Overview

- `data_processing.py` – Preprocesses raw reviews into structured feature datasets  
- `logistic_reg.py` – Logistic Regression model  
- `mlp.py` – Multi-Layer Perceptron model  
- `naive_bayes.py` – Naïve Bayes model  
- `bert.py` – BERT-based model using raw review text  
- `reviews_dataset.csv` – Raw dataset  
- `processed_reviews_dataset.csv` – Dataset with extracted features  
- `train_dataset.csv`, `dev_dataset.csv`, `test_dataset.csv` – Dataset splits  
- `confusion_matrix_*.png` – Saved evaluation confusion matrices  
- `requirements.txt` – Python dependencies  

## Install Dependencies
pip install -r requirements.txt

## Run Each Model
- Logistic Regression: python logistic_reg.py
- Multi-Layer Perceptron (MLP): python mlp.py
- Naïve Bayes: python naive_bayes.py
- BERT: python bert.py

