# Advanced Scikit-Learn Pipeline

# Import Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Dataset

data = {
    "Age": [25, 45, 35, 23, 52, 40, 28, 33, 48, 50, 37, 29, 41, 30, 55],
    "Salary": [50000, 80000, 60000, 30000, 90000, 70000, 45000, 52000, 85000, 95000, 62000, 48000, 78000, 54000, 100000],
    "City": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai", "Delhi", "Chennai", "Delhi", "Mumbai", "Mumbai", "Delhi", "Chennai", "Mumbai", "Delhi", "Mumbai"],
    "Gender": ["M", "F", "M", "F", "F", "M", "M", "F", "M", "F", "M", "F", "M", "F", "M"],
    "Churn": [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop('Churn', axis = 1)
y = df['Churn']

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state= 42
)

# Preprocessing

num_features = ['Age', 'Salary']
cat_features = ['City', 'Gender']

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

# Pipeline (Initial Model)

pipeline = Pipeline(steps = [
    ('preprocessing', preprocessor),
    ('model', LogisticRegression())
])

# Cross Validation

cv_scores = cross_val_score(pipeline, X, y, cv = 5)

print('Cross-Validation Scores', cv_scores)
print('Mean CV Score: ', np.mean(cv_scores))

# Grid Search (Hyperparameter Tuning)
param_grid = [
    {
    'model': [LogisticRegression()],
    # Logistic Regression params
    'model__C': [0.1, 1, 10]
    },
    # Random Forest Params
    {
    'model': [RandomForestClassifier()],
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, None]
    }
]

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv = 3,
    scoring = 'accuracy'
)

grid.fit(X_train, y_train)

print('Best Model: ', grid.best_estimator_)
print('Best Parameters:', grid.best_params_)

# Final Prediction

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

print('Final Accuracy: ', accuracy_score(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))




# Extracting Feature Names

feature_names = best_model.named_steps['preprocessing'].get_feature_names_out()

model = best_model.named_steps['model']

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by = 'importance', ascending = False)

    print('Feature Importance:', feature_importance_df)


if hasattr(model, 'coef_'):
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    }).sort_values(by = 'coefficient', ascending=False)

    print('Feature Coefficients:', coef_df)

import joblib 

joblib.dump(best_model, 'churn_model.pkl')

loaded_model = joblib.load('churn_model.pkl')

