import numpy as np
import pickle
import joblib
import json
import os

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==============================
# 1. Load Dataset
# ==============================

data = fetch_california_housing()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 2. Train Regression Model
# ==============================

model = LinearRegression()
model.fit(X_train, y_train)

# ==============================
# 3. Save Model using Pickle
# ==============================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ==============================
# 4. Save Model using Joblib
# ==============================

joblib.dump(model, "model.joblib")

# ==============================
# 5. Save Weights as JSON
# ==============================

model_weights = {
    "coefficients": model.coef_.tolist(),
    "intercept": model.intercept_
}

with open("model_weights.json", "w") as f:
    json.dump(model_weights, f, indent=4)

# ==============================
# 6. File Sizes
# ==============================

print("Saved Model Files & Sizes:")
print(f"model.pkl: {os.path.getsize('model.pkl')} bytes")
print(f"model.joblib: {os.path.getsize('model.joblib')} bytes")
print(f"model_weights.json: {os.path.getsize('model_weights.json')} bytes")
