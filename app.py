from flask import Flask, render_template, request
import numpy as np
import pickle
import re

app = Flask(__name__)

# ============================
# Load Model & Encoder
# ============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Feature names from model (safe)
df_cols = list(model.feature_names_in_)

print("Loaded feature count:", len(df_cols))

# ============================
# Smart Symptom Aliases
# (Keys must match dataset-style names)
# ============================
SYMPTOM_ALIASES = {
    "fever": ["fever", "high temperature", "hot body"],
    "shortness of breath": ["shortness of breath", "breathing problem", "difficulty breathing"],
    "sharp chest pain": ["chest pain", "chest tightness", "heart pain"],
    "headache": ["headache", "head pain", "migraine"],
    "abdominal pain": ["stomach pain", "abdominal pain", "stomach ache"],
    "cough": ["cough", "dry cough", "wet cough"],
    "fatigue": ["fatigue", "tired", "weakness"],
    "dizziness": ["dizziness", "lightheaded", "giddiness"],
    "nausea": ["nausea", "vomiting", "throwing up"],
}

# ============================
# Emergency Symptoms
# ============================
EMERGENCY_SYMPTOMS = [
    "chest pain", "shortness of breath",
    "loss of consciousness", "severe bleeding",
    "stroke", "heart attack"
]

# ============================
# Specialist Recommendation
# ============================
SPECIALIST_MAP = {
    "heart": "Cardiologist",
    "diabetes": "Endocrinologist",
    "asthma": "Pulmonologist",
    "depression": "Psychiatrist",
    "migraine": "Neurologist",
    "gastr": "Gastroenterologist",
    "covid": "Pulmonologist",
    "skin": "Dermatologist",
    "infection": "General Physician"
}

def recommend_specialist(disease):
    for key in SPECIALIST_MAP:
        if key in disease.lower():
            return SPECIALIST_MAP[key]
    return "General Physician"

# ============================
# Routes
# ============================
@app.route("/")
def home():
    return render_template("index.html")

# Allow GET to avoid 404 on refresh
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html")

    # SAFE form read
    user_input = request.form.get("symptoms", "").lower().strip()

    if not user_input:
        return render_template("index.html", error="Please enter your symptoms.")

    # Clean input
    user_input = re.sub(r"[^a-zA-Z\s]", " ", user_input)

    input_vector = np.zeros(len(df_cols))

    # Smart symptom matching
    for i, col in enumerate(df_cols):
        clean_col = col.replace("_", " ").lower()

        # Direct match
        if clean_col in user_input:
            input_vector[i] = 1

        # Alias match
        if clean_col in SYMPTOM_ALIASES:
            for alias in SYMPTOM_ALIASES[clean_col]:
                if alias in user_input:
                    input_vector[i] = 1

    # Predict probabilities
    probs = model.predict_proba([input_vector])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    top3 = []
    for idx in top3_idx:
        disease_name = encoder.inverse_transform([idx])[0]
        confidence = round(probs[idx] * 100, 2)
        top3.append((disease_name, confidence))

    # Emergency detection
    is_emergency = any(sym in user_input for sym in EMERGENCY_SYMPTOMS)

    # Specialist
    specialist = recommend_specialist(top3[0][0])

    return render_template(
        "result.html",
        top3=top3,
        specialist=specialist,
        emergency=is_emergency
    )

if __name__ == "__main__":
    app.run(debug=True)
