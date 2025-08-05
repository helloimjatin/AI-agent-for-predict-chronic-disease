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
# Load all trained models
# =========================
diabetes_model = joblib.load("diabetes_logreg_pipeline.joblib")
heart_model = joblib.load("heart_disease_lr_pipeline.joblib")
hypertension_model = joblib.load("hypertension_logreg_pipeline.joblib")

# =========================
# Default values for features
# =========================
default_values = {
    "diabetes": {
        "Pregnancies": 3,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 32,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 33
    },
    "heart": {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 246,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 2
    },
    "hypertension": {
        "age": 50,
        "education": 2,
        "sex": 1,
        "is_smoking": 0,
        "cigsPerDay": 5,
        "BPMeds": 0,
        "prevalentStroke": 0,
        "diabetes": 0,
        "totChol": 200,
        "sysBP": 120,
        "diaBP": 80,
        "BMI": 25,
        "heartRate": 75,
        "glucose": 90
    }
}

# =========================
# Recommendations list
# =========================
recommendations = {
    "diabetes": {
        "high": [
            "Limit refined sugars and processed foods.",
            "Exercise at least 30 minutes a day.",
            "Maintain a healthy weight.",
            "Monitor blood sugar levels regularly.",
            "Eat more vegetables, lean proteins, and whole grains.",
            "Consult your healthcare provider for a checkup."
        ],
        "low": [
            "Maintain a balanced diet with fruits, vegetables, and whole grains.",
            "Stay active with regular exercise.",
            "Keep a healthy weight.",
            "Get routine checkups to monitor blood sugar."
        ]
    },
    "heart": {
        "high": [
            "Consult a cardiologist for a heart health evaluation.",
            "Reduce saturated fats, trans fats, and salt in your diet.",
            "Eat more vegetables and omega‑3‑rich foods.",
            "Exercise at least 150 minutes a week.",
            "Avoid smoking and limit alcohol."
        ],
        "low": [
            "Continue a heart‑healthy diet rich in vegetables and lean proteins.",
            "Do regular cardio exercise.",
            "Monitor cholesterol and blood pressure during checkups."
        ]
    },
    "hypertension": {
        "high": [
            "Reduce salt and sodium‑rich food intake.",
            "Eat potassium‑rich foods like bananas and spinach.",
            "Maintain a healthy weight.",
            "Manage stress through relaxation techniques.",
            "Monitor blood pressure daily."
        ],
        "low": [
            "Maintain a balanced, low‑salt diet.",
            "Stay physically active.",
            "Monitor your blood pressure during checkups."
        ]
    }
}

# =========================
# Helper function for random tips
# =========================
def get_random_recommendations(tips, count=3):
    return random.sample(tips, min(len(tips), count))

# =========================
# Casual input prompts
# =========================
casual_prompts = {
    "Glucose": "What's your blood sugar level? ",
    "BloodPressure": "What's your blood pressure? ",
    "Insulin": "What's your insulin level? ",
    "Age": "How old are you? ",
    "Pregnancies": "How many times have you been pregnant? (0 if none) ",
    "SkinThickness": "What's your skin fold thickness? ",
    "BMI": "What's your Body Mass Index (BMI)? ",
    "DiabetesPedigreeFunction": "Enter your diabetes pedigree function value: ",

    "age": "How old are you? ",
    "sex": "What's your gender? (1 for male, 0 for female) ",
    "cp": "What's your chest pain type? (0-3) ",
    "chol": "What's your cholesterol level? ",
    "fbs": "Is your fasting blood sugar > 120 mg/dL? (1 for yes, 0 for no) ",
    "trestbps": "What's your resting blood pressure? ",
    "restecg": "What's your resting ECG result? ",
    "thalach": "What's your maximum heart rate achieved? ",
    "exang": "Did you have exercise-induced angina? (1 for yes, 0 for no) ",
    "oldpeak": "What's your ST depression value? ",
    "slope": "What's the slope of the peak exercise ST segment? ",
    "ca": "Number of major vessels colored by fluoroscopy? ",
    "thal": "Thalassemia type? (0-3) ",

    "education": "What's your education level (1-4)? ",
    "is_smoking": "Do you smoke? (1 for yes, 0 for no) ",
    "cigsPerDay": "How many cigarettes do you smoke per day? ",
    "BPMeds": "Are you taking BP medication? (1 for yes, 0 for no) ",
    "prevalentStroke": "Have you had a stroke before? (1 for yes, 0 for no) ",
    "diabetes": "Do you have diabetes? (1 for yes, 0 for no) ",
    "totChol": "What's your total cholesterol level? ",
    "sysBP": "What's your systolic BP? ",
    "diaBP": "What's your diastolic BP? ",
    "heartRate": "What's your resting heart rate? ",
    "glucose": "What's your fasting glucose level? "
}

# =========================
# Get input (ask only when needed)
# =========================
def get_input(feature_name, default):
    val = input(casual_prompts.get(feature_name, f"{feature_name} (default {default}): ")).strip()
    return float(val) if val else default

# =========================
# Prediction function
# =========================
def predict_disease(model, always_ask, feature_order, defaults, recs, disease_name):
    print(f"\nEnter patient details for {disease_name.capitalize()} Prediction:")

    # Ask for always_ask features, use defaults for the rest
    values = []
    for f in feature_order:
        if f in always_ask:
            values.append(get_input(f, defaults[f]))
        else:
            values.append(defaults[f])

    values_array = np.array(values).reshape(1, -1)
    pred = model.predict(values_array)[0]
    prob = model.predict_proba(values_array)[0][int(pred)]

    if pred == 1:
        print(f"\n⚠️ High Risk of {disease_name.capitalize()} — {prob*100:.2f}% probability")
        selected_tips = get_random_recommendations(recs["high"])
    else:
        print(f"\n✅ Low Risk of {disease_name.capitalize()} — {prob*100:.2f}% probability")
        selected_tips = get_random_recommendations(recs["low"])

    print("Recommendations:")
    for tip in selected_tips:
        print(f"- {tip}")

# =========================
# Main AI Agent Logic
# =========================
while True:
    print("\nChoose model to use:")
    print("1 - Diabetes Prediction")
    print("2 - Heart Disease Prediction")
    print("3 - Hypertension Prediction")
    print("4 - All Diseases")
    print("0 - Exit")

    choice = input("Enter choice (0/1/2/3/4): ").strip()

    if choice == "0":
        print("Goodbye! ❤️")
        break
    elif choice == "1":
        predict_disease(
            diabetes_model,
            always_ask=["Glucose", "BloodPressure", "Insulin", "Age"],
            feature_order=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
            defaults=default_values["diabetes"],
            recs=recommendations["diabetes"],
            disease_name="diabetes"
        )
    elif choice == "2":
        predict_disease(
            heart_model,
            always_ask=["age", "sex", "fbs", "chol", "cp"],
            feature_order=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            defaults=default_values["heart"],
            recs=recommendations["heart"],
            disease_name="heart disease"
        )
    elif choice == "3":
        predict_disease(
            hypertension_model,
            always_ask=["age", "education", "sex", "is_smoking", "diabetes", "heartRate", "glucose"],
            feature_order=["age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds", "prevalentStroke", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"],
            defaults=default_values["hypertension"],
            recs=recommendations["hypertension"],
            disease_name="hypertension"
        )
    elif choice == "4":
        print("\n--- Predicting All Diseases ---")
        predict_disease(
            diabetes_model,
            always_ask=["Glucose", "BloodPressure", "Insulin", "Age"],
            feature_order=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
            defaults=default_values["diabetes"],
            recs=recommendations["diabetes"],
            disease_name="diabetes"
        )
        predict_disease(
            heart_model,
            always_ask=["age", "sex", "fbs", "chol", "cp"],
            feature_order=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
            defaults=default_values["heart"],
            recs=recommendations["heart"],
            disease_name="heart disease"
        )
        predict_disease(
            hypertension_model,
            always_ask=["age", "education", "sex", "is_smoking", "diabetes", "heartRate", "glucose"],
            feature_order=["age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds", "prevalentStroke", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"],
            defaults=default_values["hypertension"],
            recs=recommendations["hypertension"],
            disease_name="hypertension"
        )
    else:
        print("❌ Invalid choice. Please try again.")
