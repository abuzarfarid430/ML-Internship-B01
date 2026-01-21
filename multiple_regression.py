"""
Multiple Linear Regression using scikit-learn
---------------------------------------------
Dataset: California Housing Dataset

"""

import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load California Housing Dataset

housing = fetch_california_housing()
X = housing.data
y = housing.target
feature_names = housing.feature_names

print("Feature Names:", feature_names)


# 2. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Initialize & Train Model

model = LinearRegression()
model.fit(X_train, y_train)


# 4. Predictions

y_pred = model.predict(X_test)

# 5. Evaluation Metrics

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# 6. Feature Coefficients

print("\nFeature Coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")

print(f"\nIntercept: {model.intercept_:.4f}")


# 7. Actual vs Predicted Plot

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_predicted.png")
plt.show()


# 8. Residuals Plot

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.savefig("residuals_plot.png")
plt.show()


# 9. Save Model Weights

joblib.dump(model, "multiple_linear_regression_model.pkl")
print("\nModel saved as multiple_linear_regression_model.pkl")
