import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the dataset
data = pd.read_csv("diabetes_dataset.csv")

# Step 2: Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Step 3: Reproduce train/test split as in notebook (holdout_size=0.1, random_state=33)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=33, stratify=y
)

# Step 4: Build the pipeline as per notebook
# The notebook's pipeline applies the following:
# - Column selector: select all columns (already done)
# - Float casting: all values already float
# - Replace missing: treat as np.nan (use the SimpleImputer)
# - Imputation: median
# - StandardScaler: not used (use_scaler_flag=False), so skip for strict reproduction
# - Feature engineering: add squared features for all columns
def add_squared_features(X):
    X = np.asarray(X, dtype=np.float32)
    squared = np.square(X)
    return np.concatenate([X, squared], axis=1)

import sklearn.base as base

class FeatureEngineer(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return add_squared_features(X)

# - FS1: select original features (first 8 columns); in full pipeline, this keeps main features after engineering
class FeatureSelector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, keep=8):
        self.keep = keep
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, :self.keep]

# - PCA: fits on engineered features, default components
# - FS1: select original features again
# - LogisticRegression: as in notebook

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Median imputation
    ("fe", FeatureEngineer()),                     # Add squared features
    ("fs1", FeatureSelector(keep=8)),              # Select first 8 columns
    ("pca", PCA()),                                # PCA on selected columns
    ("fs2", FeatureSelector(keep=8)),              # Select again (workaround: keep after PCA)
    ("clf", LogisticRegression(
        class_weight="balanced", max_iter=999, multi_class="ovr", n_jobs=1, random_state=33
    )),
])

# Step 5: Fit the pipeline
pipeline.fit(X_train.values, y_train.values)

# Step 6: Evaluate on holdout (test) set
y_pred = pipeline.predict(X_test.values)
score = accuracy_score(y_test.values, y_pred)
print(f"Holdout accuracy: {score:.4f}")

# Step 7: Save the trained model
joblib.dump(pipeline, "diabetes_logreg_pipeline.joblib")
print("Model saved to diabetes_logreg_pipeline.joblib")
