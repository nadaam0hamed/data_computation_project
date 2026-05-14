# Human Activity Recognition Using SVM

Final Project – Data Computation Spring 2026  
Alexandria National University  
Computers and Data Science

---

# Project Overview

This project applies Support Vector Machine (SVM) techniques to classify human activities using smartphone sensor data.

The dataset contains accelerometer and gyroscope measurements collected from participants performing daily activities such as walking, sitting, and standing.

The project includes:
- Exploratory Data Analysis (EDA)
- Data Cleaning
- Feature Scaling
- Dimensionality Reduction using PCA
- SVM Model Development
- Hyperparameter Tuning
- Model Evaluation
- Streamlit Deployment (Bonus)

---

# Dataset Information

Dataset: Human Activity Recognition (HAR)

## Dataset Characteristics
- Total Records: 10,299
- Total Features: 561 sensor features
- Classes: 6 human activities

## Activities
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

# Project Structure

```text
project/
│
├── train.csv
├── test.csv
├── notebook.ipynb
├── app.py
├── svm_har_model.pkl
├── requirements.txt
└── README.md