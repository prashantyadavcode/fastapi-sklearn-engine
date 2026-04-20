# =====================================
# ADVANCED SKLEARN PIPELINE WITH MLFLOW
# =====================================




# IMPORT LIBRARIES
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline

import joblib

# MLflow
import mlflow
import mlflow.sklearn

# Tracking URI of MLFLOW

print("Tracking URI:", mlflow.get_tracking_uri())

# =====================================
# DATASET
# =====================================
data = pd.read_csv('data/telco_customer_churn.csv')

df = pd.DataFrame(data)



df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df = df.dropna()

df = df.drop('customerID', axis = 1)

X = df.drop("Churn", axis=1)
y = df["Churn"].map({'Yes': 1, 'No': 0})

# =====================================
# TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# PREPROCESSING
# =====================================
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cat_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

num_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

cat_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ]
)

# =====================================
# PIPELINE
# =====================================
pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])

# =====================================
# CROSS VALIDATION
# =====================================
cv_scores = cross_val_score(pipeline, X, y, cv=5)

print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# =====================================
# PARAM GRID
# =====================================
param_grid = [
    {
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": [0.1, 1, 10]
    },
    {
        "model": [RandomForestClassifier()],
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5, None]
    }
]

# =====================================
# MLFLOW EXPERIMENT
# =====================================
mlflow.set_experiment("Churn Prediction Pipeline")

with mlflow.start_run():

    # GRID SEARCH
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("\nBest Model:", best_model)
    print("Best Parameters:", grid.best_params_)

    # =====================================
    # EVALUATION
    # =====================================
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nFinal Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # =====================================
    # LOGGING TO MLFLOW
    # =====================================
    mlflow.log_metric("accuracy", acc)

    safe_params = {k: str(v) for k, v in grid.best_params_.items()}
    mlflow.log_params(safe_params)

    mlflow.sklearn.log_model(best_model, "model")

    # =====================================
    # FEATURE IMPORTANCE / COEFFICIENTS
    # =====================================
    feature_names = best_model.named_steps["preprocessing"].get_feature_names_out()
    model = best_model.named_steps["model"]

    # Random Forest
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        print("\nFeature Importance:\n", feature_importance_df)

    # Logistic Regression
    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": model.coef_[0]
        }).sort_values(by="coefficient", ascending=False)

        print("\nFeature Coefficients:\n", coef_df)

# =====================================
# SAVE MODEL LOCALLY 
# =====================================
# joblib.dump(best_model, "churn_model.pkl")

# print("\nModel saved as churn_model.pkl")

