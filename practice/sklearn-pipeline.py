# Pipeline - Automated ML Workflow

# Imported Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a realistic dataset

data = {
    "Age": [25, 45, 35, 23, 52, 40, 28, 33, 48, 50, 37, 29, 41, 30, 55],
    "Salary": [50000, 80000, 60000, 30000, 90000, 70000, 45000, 52000, 85000, 95000, 62000, 48000, 78000, 54000, 100000],
    "City": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai", "Delhi", "Chennai", "Delhi", "Mumbai", "Mumbai", "Delhi", "Chennai", "Mumbai", "Delhi", "Mumbai"],
    "Gender": ["M", "F", "M", "F", "F", "M", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "Churn": [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Split Features & Target

X = df.drop('Churn', axis = 1)
y = df['Churn']

# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42
)

# Feature Types

num_features = ['Age', 'Salary']
cat_features = ['City', 'Gender']

# Preprocessor 

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), cat_features)
    ]
)

# Pipeline

pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('model', LogisticRegression())
    ]
)

# Train Model

pipeline.fit(X_train, y_train)

# Prediction

y_pred = pipeline.predict(X_test)

# Evaluation

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report: ', classification_report(y_test, y_pred))


