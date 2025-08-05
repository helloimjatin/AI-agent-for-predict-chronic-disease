# incremental_learning_script.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==== STEP 1: CONFIGURATION ====
data_path = 'hypertension_dataset_clean.csv'
chunk_size = 848
target_column = 'prevalentHyp'
classes = np.array([0, 1])

# ==== STEP 2: LOAD/MOCK PIPELINE ====
def load_pipeline():
    # Use log_loss so we can have probability predictions
    from sklearn.linear_model import SGDClassifier
    pipeline = SGDClassifier(loss="log_loss", max_iter=1000)  # logistic regression with SGD
    return pipeline

pipeline_model = load_pipeline()

# ==== STEP 3: BATCHED DATA READING ====
reader = pd.read_csv(data_path, chunksize=chunk_size)

partial_fit_scores = []
fit_times = []
import time

# ==== STEP 4: INCREMENTAL LEARNING ====
for i, batch_df in enumerate(reader):
    batch_df.dropna(subset=[target_column], inplace=True)
    X = batch_df.drop([target_column], axis=1).values
    y = batch_df[target_column].values

    if i == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=33
        )
        start_time = time.time()
        pipeline_model.partial_fit(X_train, y_train, classes=classes)
        fit_times.append(time.time() - start_time)
        y_pred = pipeline_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        partial_fit_scores.append(score)
        print(f'Batch {i+1} - Accuracy: {score:.4f}')
    else:
        start_time = time.time()
        pipeline_model.partial_fit(X, y)
        fit_times.append(time.time() - start_time)
        y_pred = pipeline_model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        partial_fit_scores.append(score)
        print(f'Batch {i+1} - Accuracy: {score:.4f}')

# ==== STEP 5: FINAL EVALUATION ====
print('\nFinal evaluation on test set:')
print('Accuracy:', accuracy_score(y_test, pipeline_model.predict(X_test)))
print('Confusion Matrix:')
print(confusion_matrix(y_test, pipeline_model.predict(X_test)))
print('\nClassification Report:')
print(classification_report(y_test, pipeline_model.predict(X_test)))

# ==== STEP 6: SAVE MODEL LOCALLY ====
output_model_path = 'hypertension_logreg_pipeline.joblib'
joblib.dump(pipeline_model, output_model_path)
print(f'Model saved to {output_model_path}')

# ==== OPTIONAL: PLOT LEARNING CURVE ====
plt.figure(figsize=(8, 4))
plt.plot(partial_fit_scores, marker='o', label='Accuracy per batch')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Incremental Learning Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
