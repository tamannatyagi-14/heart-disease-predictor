# ❤️ Heart Disease Predictor

A Streamlit web app that predicts the risk of heart disease from a patient's clinical details, and explains **which factors pushed the result** up or down.

> This is a student machine learning project, not a medical diagnosis tool.

## Features

- Simple form with 11 clinical inputs (age, blood pressure, cholesterol, chest pain type, etc.)
- Risk prediction with probability using a trained Logistic Regression model
- "What pushed this result?" chart showing the top factors behind each prediction (based on the model's coefficients)

## How it works

1. The user's inputs are converted into the same 20 one-hot encoded columns used during training.
2. The values are scaled with the saved `StandardScaler`-style scaler.
3. The Logistic Regression model predicts the class and probability.
4. Each feature's contribution (`coefficient × scaled value`) is plotted to explain the result.

## Tech stack

Python, pandas, scikit-learn, joblib, Streamlit

## Model

- Algorithm: Logistic Regression (best accuracy among the classifiers I compared)
- Dataset: Heart Failure Prediction dataset (`heart.csv`, 918 rows)
- Accuracy: **ADD YOUR ACCURACY HERE**

## Project structure

```
├── app.py                          # Streamlit app
├── heart.ipynb                     # Model training notebook
├── heart.csv                       # Dataset
├── LogisticRegression_heart.pkl    # Trained model
├── scaler_heart.pkl                # Fitted scaler
├── columns_heart.pkl               # Column order used in training
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Tamanna Tyagi
