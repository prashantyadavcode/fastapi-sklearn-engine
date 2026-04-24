Customer Churn Prediction System

Problem - 

Customer churn is a critical business problem where companies lose customers over time. Predicting churn helps organizations take proactive actions such as targeted retention campaigns, personalized offers, and improved customer experience.

The objective of this project is to build a machine learning system that can predict whether a customer is likely to churn based on historical data.

Approach - 

The problem is treated as a supervised binary classification task.

The workflow followed:

- Data cleaning and preprocessing on real-world Telco customer dataset
- Handling missing values and inconsistent data types
- Feature transformation using encoding and scaling
- Model training using multiple algorithms
- Hyperparameter tuning using GridSearchCV
- Evaluation using multiple metrics beyond accuracy
- Experiment tracking using MLflow
- Deployment using FastAPI for real-time predictions

Pipeline - 

A complete scikit-learn pipeline is implemented to ensure reproducibility and scalability.

Preprocessing - 

- Numerical Features:
  - Imputation using median strategy
  - Standard scaling

- Categorical Features:
  - Imputation using most frequent values
  - One-hot encoding

Models Used - 

- Logistic Regression
- Random Forest Classifier

Pipeline Flow - 

Data → Preprocessing → Model → Prediction

Results - 

The model performance is evaluated using multiple metrics:

- Accuracy
- Classification Report (Precision, Recall, F1-score)
- ROC-AUC Score
- Confusion Matrix

Artifacts generated:

- Confusion Matrix (CSV)
- Feature Importance (CSV)
- Logistic Coefficients (CSV)
- ROC-AUC Score (CSV)

API - 

A FastAPI service is built to serve predictions in real time.

Endpoint - 

POST /predict

Sample Input - 

{
  "tenure": 12,
  "MonthlyCharges": 70,
  "TotalCharges": 840,
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}

Output - 

{
  "prediction": 0
}

MLflow - 

MLflow is used for experiment tracking and model management.

Features Used - 

- Tracking metrics such as accuracy and ROC-AUC
- Logging hyperparameters from GridSearchCV
- Saving trained models as artifacts
- Storing evaluation outputs (CSV files)

Project Structure - 

scikit-learn-pipeline/
│
├── data/
├── ml-api/
├── outputs/
├── training/
├── advanced-sklearn-pipeline.py
├── mlflow.db
├── requirements.txt
└── README.txt

Key Highlights - 

- End-to-end ML pipeline with real dataset
- Proper preprocessing using ColumnTransformer
- Hyperparameter tuning using GridSearchCV
- MLflow-based experiment tracking
- API deployment using FastAPI
- Structured output artifacts for analysis

Conclusion - 

This project demonstrates a complete machine learning workflow from data processing to deployment. It emphasizes reproducibility, scalability, and real-world applicability.
