# train.py

"""
IRIS ML Pipeline Training Script
- Loads the IRIS dataset
- Splits into train/test
- Trains a RandomForest classifier
- Evaluates accuracy
- Saves the trained model and test data for DVC tracking
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# ------------------------
# Load dataset
# ------------------------
DATA_PATH = "data/iris.csv"  # your raw iris dataset
df = pd.read_csv(DATA_PATH)

# Features and target
X = df.drop(columns=["species"])
y = df["species"]

# ------------------------
# Split into train/test
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# Train model
# ------------------------
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# ------------------------
# Evaluate
# ------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Test accuracy: {acc:.2f}")

# ------------------------
# Save model and test data
# ------------------------
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

os.makedirs("data", exist_ok=True)
test_df = X_test.copy()
test_df["species"] = y_test
test_df.to_csv("data/test.csv", index=False)

print("✅ Model and test data saved successfully.")
