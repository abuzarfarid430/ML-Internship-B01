import numpy as np
import pickle
import joblib
import json
import os
import time

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# ==============================
# 1. Load Sample Data
# ==============================

data = fetch_california_housing()
X_sample = data.data[:5]  # new/unseen data

# ==============================
# 2. Load Pickle Model
# ==============================

start = time.time()
with open("model.pkl", "rb") as f:
    pickle_model = pickle.load(f)
pickle_time = time.time() - start

pickle_pred = pickle_model.predict(X_sample)

# ==============================
# 3. Load Joblib Model
# ==============================

start = time.time()
joblib_model = joblib.load("model.joblib")
joblib_time = time.time() - start

joblib_pred = joblib_model.predict(X_sample)

# ==============================
# 4. Load JSON Weights Manually
# ==============================

start = time.time()
with open("model_weights.json", "r") as f:
    weights = json.load(f)
json_time = time.time() - start

json_model = LinearRegression()
json_model.coef_ = np.array(weights["coefficients"])
json_model.intercept_ = weights["intercept"]

json_pred = json_model.predict(X_sample)

# ==============================
# 5. File Sizes
# ==============================

pickle_size = os.path.getsize("model.pkl")
joblib_size = os.path.getsize("model.joblib")
json_size = os.path.getsize("model_weights.json")

# ==============================
# 6. Results
# ==============================

print("\nPredictions Comparison:")
print("Pickle Predictions:", pickle_pred)
print("Joblib Predictions:", joblib_pred)
print("JSON Predictions  :", json_pred)

print("\nFile Size & Load Time Comparison:")
print("Format   | Size (bytes) | Load Time (sec)")
print("------------------------------------------")
print(f"Pickle   | {pickle_size:^12} | {pickle_time:.6f}")
print(f"Joblib   | {joblib_size:^12} | {joblib_time:.6f}")
print(f"JSON     | {json_size:^12} | {json_time:.6f}")
