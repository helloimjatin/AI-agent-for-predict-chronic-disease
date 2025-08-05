# streamlit_app.py
import streamlit as st
import numpy as np
import joblib
import random
import sklearn.base as base

# =========================
# Custom classes used in training
# =========================
class FeatureEngineer(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        squared = np.square(X)
        return np.concatenate([X, squared], axis=1)

class FeatureSelector(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, keep=8):
        self.keep = keep
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, :self.keep]

# =========================
# Load trained models
# =========================
diabetes_model = joblib.load("diabetes_logreg_pipeline.joblib")
heart_model = joblib.load("heart_disease_lr_pipeline.joblib")
hypertension_model = joblib.load("hypertension_logreg_pipeline.joblib")

# =========================
# Default values
# =========================
default_values = {
    "diabetes": {
        "Pregnancies": 3, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 80, "BMI": 32, "DiabetesPedigreeFunction": 0.5, "Age": 33
    },
    "heart": {
        "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 246, "fbs": 0,
        "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    },
    "hypertension": {
        "age": 50, "education": 2, "sex": 1, "is_smoking": 0, "cigsPerDay": 5,
        "BPMeds": 0, "prevalentStroke": 0, "diabetes": 0, "totChol": 200,
        "sysBP": 120, "diaBP": 80, "BMI": 25, "heartRate": 75, "glucose": 90
    }
}

# =========================
# Recommendations
# =========================
recommendations = {
    "diabetes": {
        "high": [
            "Exercise at least 30 minutes daily.",
            "Limit sugar and refined carbs.",
            "Monitor blood sugar regularly.",
            "Maintain a healthy weight.",
            "Consult your healthcare provider regularly."
        ],
        "low": [
            "Keep up with healthy eating habits.",
            "Stay physically active.",
            "Have regular health check-ups."
        ]
    },
    "heart": {
        "high": [
            "Reduce saturated fats and salt intake.",
            "Monitor blood pressure regularly.",
            "Exercise at least 150 minutes per week.",
            "Quit smoking immediately.",
            "Manage stress effectively."
        ],
        "low": [
            "Maintain a balanced diet.",
            "Stay physically active.",
            "Continue regular heart check-ups."
        ]
    },
    "hypertension": {
        "high": [
            "Reduce salt in your diet.",
            "Check your blood pressure daily.",
            "Limit alcohol and quit smoking.",
            "Exercise regularly to manage weight.",
            "Consult a doctor for medication advice."
        ],
        "low": [
            "Maintain healthy lifestyle habits.",
            "Monitor your blood pressure regularly.",
            "Stay active and eat a balanced diet."
        ]
    }
}

# =========================
# Prediction function
# =========================
def predict_disease(disease, inputs):
    if disease == "Diabetes":
        model = diabetes_model
        feature_order = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        defaults = default_values["diabetes"]
        recs = recommendations["diabetes"]
    elif disease == "Heart Disease":
        model = heart_model
        feature_order = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        defaults = default_values["heart"]
        recs = recommendations["heart"]
    else:
        model = hypertension_model
        feature_order = ["age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds", "prevalentStroke", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
        defaults = default_values["hypertension"]
        recs = recommendations["hypertension"]

    values = [float(inputs.get(f, defaults[f])) if inputs.get(f, "") != "" else defaults[f] for f in feature_order]
    values_array = np.array(values).reshape(1, -1)
    pred = model.predict(values_array)[0]
    prob = model.predict_proba(values_array)[0][int(pred)]

    risk = "High" if pred == 1 else "Low"
    advice_list = recs["high"] if pred == 1 else recs["low"]
    random.shuffle(advice_list)
    selected_advice = "\n".join([f"- {a}" for a in advice_list[:3]])

    return f"**{risk} Risk of {disease}** — {prob*100:.2f}% probability", selected_advice

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Chronic Disease Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Chronic Disease Prediction AI")
st.write("Select a disease, fill in the essential details, and get prediction results with doctor's advice.")

disease_choice = st.selectbox("Choose Disease", ["Diabetes", "Heart Disease", "Hypertension"])

user_inputs = {}

# Dynamic form rendering
st.subheader("Patient Information")

if disease_choice == "Diabetes":
    user_inputs["Glucose"] = st.number_input("Glucose", min_value=0.0, step=1.0)
    user_inputs["BloodPressure"] = st.number_input("Blood Pressure", min_value=0.0, step=1.0)
    user_inputs["Insulin"] = st.number_input("Insulin", min_value=0.0, step=1.0)
    user_inputs["Age"] = st.number_input("Age", min_value=0, step=1)

elif disease_choice == "Heart Disease":
    user_inputs["age"] = st.number_input("Age", min_value=0, step=1)
    user_inputs["sex"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    user_inputs["cp"] = st.number_input("Chest Pain Type (cp)", min_value=0, step=1)
    user_inputs["chol"] = st.number_input("Cholesterol", min_value=0.0, step=1.0)
    user_inputs["fbs"] = st.number_input("Fasting Blood Sugar (fbs)", min_value=0, max_value=1, step=1)

elif disease_choice == "Hypertension":
    user_inputs["age"] = st.number_input("Age", min_value=0, step=1)
    user_inputs["education"] = st.number_input("Education Level", min_value=0, step=1)
    user_inputs["sex"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    user_inputs["is_smoking"] = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
    user_inputs["diabetes"] = st.number_input("Diabetes Status", min_value=0, max_value=1, step=1)
    user_inputs["heartRate"] = st.number_input("Heart Rate", min_value=0.0, step=1.0)
    user_inputs["glucose"] = st.number_input("Glucose", min_value=0.0, step=1.0)

if st.button("🔍 Predict"):
    result, advice = predict_disease(disease_choice, user_inputs)
    st.subheader("Prediction Result")
    st.markdown(result)
    st.markdown("### 🩺 Doctor's Advice")
    st.info(advice)
