"""
Model:
    y = w*x + b

Dataset:
    y = 2x + 1 + noise

Author: Abu Zar Farid
"""

import numpy as np
import matplotlib.pyplot as plt


# 1. Create Synthetic Dataset

np.random.seed(42)  # for reproducibility

# Generate input feature (x)
X = np.linspace(0, 10, 100)  # 100 points between 0 and 10

# Generate noise
noise = np.random.normal(0, 1, size=len(X))

# True relationship
y = 2 * X + 1 + noise

# Reshape X for consistency
X = X.reshape(-1, 1)


# 2. Linear Regression Class

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize learning rate and number of iterations
        """
        self.lr = learning_rate
        self.iterations = iterations
        self.w = 0  # weight
        self.b = 0  # bias
        self.cost_history = []

   
    # 3. Cost Function (MSE)
    def compute_cost(self, y_true, y_pred):
        """
        Mean Squared Error:
        J(w, b) = (1/n) * Σ(y_pred - y_true)^2
        """
        n = len(y_true)
        cost = (1 / n) * np.sum((y_pred - y_true) ** 2)
        return cost

   
    # 4. Gradient Descent (fit)
    def fit(self, X, y):
        """
        Train the model using gradient descent
        """
        n = len(y)

        for i in range(self.iterations):
            # Prediction using current w and b
            y_pred = self.w * X.flatten() + self.b

            # Compute gradients
            dw = (2 / n) * np.sum((y_pred - y) * X.flatten())
            db = (2 / n) * np.sum(y_pred - y)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Store cost for plotting
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)

            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    
    # 5. Prediction Function
    def predict(self, X):
        """
        Predict output for given X
        """
        return self.w * X.flatten() + self.b



# 6. Train the Model

model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)


# 7. R² Score (Manual)

def r2_score(y_true, y_pred):
    """
    R² = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

r2 = r2_score(y, y_pred)
print(f"\nR² Score: {r2:.4f}")


# 8. Plot Data & Regression Line

plt.figure()
plt.scatter(X, y, label="Data Points")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Simple Linear Regression from Scratch")
plt.legend()
plt.savefig("regression_line.png")
plt.show()


# 9. Plot Cost vs Iterations

plt.figure()
plt.plot(model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.savefig("cost_convergence.png")
plt.show()
