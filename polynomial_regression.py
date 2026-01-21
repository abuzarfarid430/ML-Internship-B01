import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error

# ==============================
# 1. Create Synthetic Non-Linear Dataset
# ==============================

np.random.seed(42)

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 - X**2 + X + np.random.normal(0, 3, size=X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 2. Polynomial Regression Setup
# ==============================

degrees = [1, 2, 3, 5, 10]
train_errors = []
test_errors = []

X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.figure()

# ==============================
# 3. Loop Over Polynomial Degrees
# ==============================

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_plot_poly = poly.transform(X_plot)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    y_plot_pred = model.predict(X_plot_poly)
    
    # Errors (MSE)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    # Plot regression curve
    plt.plot(X_plot, y_plot_pred, label=f"Degree {degree}")

# ==============================
# 4. Plot Dataset & Models
# ==============================

plt.scatter(X_train, y_train, color="black", label="Training Data")
plt.title("Polynomial Regression with Different Degrees")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.savefig("polynomial_models.png")
plt.show()

# ==============================
# 5. Error Table
# ==============================

print("\nDegree | Train MSE | Test MSE")
print("-----------------------------")
for d, tr, te in zip(degrees, train_errors, test_errors):
    print(f"{d:^6} | {tr:^9.3f} | {te:^9.3f}")

# ==============================
# 6. Plot Train vs Test Error
# ==============================

plt.figure()
plt.plot(degrees, train_errors, marker='o', label="Train Error")
plt.plot(degrees, test_errors, marker='o', label="Test Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Train vs Test Error")
plt.legend()
plt.savefig("train_test_error.png")
plt.show()

# ==============================
# 7. Learning Curves
# ==============================

plt.figure()

for degree in [1, 3, 10]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_poly,
        y,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5
    )
    
    train_mse = -np.mean(train_scores, axis=1)
    test_mse = -np.mean(test_scores, axis=1)
    
    plt.plot(train_sizes, test_mse, label=f"Degree {degree}")

plt.xlabel("Training Set Size")
plt.ylabel("MSE")
plt.title("Learning Curves")
plt.legend()
plt.savefig("learning_curves.png")
plt.show()

# ==============================
# 8. Overfitting Explanation
# ==============================

print("""
Overfitting Analysis:
- Degree 1: Underfitting (high bias, poor fit)
- Degree 2 & 3: Good balance (low train & test error)
- Degree 5: Slight overfitting
- Degree 10: Clear overfitting (very low train error, high test error)
""")
