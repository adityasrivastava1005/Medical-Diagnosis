import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading dataset...")

DATASET_PATH = "dataset/symptoms.csv"
df = pd.read_csv(DATASET_PATH)

print("Columns:", df.columns)
print("Dataset shape:", df.shape)

# ===============================
# Target & Features
# ===============================
target_col = "diseases"
X = df.drop(columns=[target_col])
y_raw = df[target_col]

# ===============================
# Encode labels
# ===============================
print("Encoding labels...")
encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

# ===============================
# Handle missing values
# ===============================
X = X.fillna(0)

# ===============================
# Train-test split (FIXED)
# ===============================
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Memory-Optimized Model
# ===============================
print("Training model...")

model = RandomForestClassifier(
    n_estimators=40,        # 🔽 further reduced for low RAM
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# ===============================
# Save model
# ===============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\n✅ SUCCESS!")
print("model.pkl created")
print("encoder.pkl created")
print("Training complete — memory safe version")
