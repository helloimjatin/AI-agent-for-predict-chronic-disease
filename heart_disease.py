import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import make_pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Split features & target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split (holdout 10%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=33, stratify=y
)

# Build pipeline: imputation + (no scaling since IBM pipeline disables it) + LR
pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    LogisticRegression(
        class_weight="balanced",
        max_iter=999,
        multi_class="ovr",
        n_jobs=-1,
        random_state=33
    )
)

# Train
pipeline.fit(X_train, y_train)

# Predict & metrics
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nMetrics on holdout test set:")
print(f"Accuracy:   {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall:     {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:   {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:    {roc_auc_score(y_test, y_prob):.4f}")

print("\nConfusion matrix (rows: true, columns: pred):")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# Example predictions for first 5 X_test rows
print("Predictions for the first 5 X_test rows:")
print(pipeline.predict(X_test.iloc[:5]))

# Save the pipeline
joblib.dump(pipeline, "heart_disease_lr_pipeline.joblib")
print("\nTrained model saved as 'heart_disease_lr_pipeline.joblib'")