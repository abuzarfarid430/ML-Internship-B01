import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

input_size = 2
hidden_size = 4   # You can experiment with this
output_size = 1

# Weight matrices
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# 3. Sigmoid Activation Function
def sigmoid(z):
    """
    Sigmoid activation function.
    
    Formula:
        σ(z) = 1 / (1 + e^(-z))
    
    It squashes values between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """
    Derivative of sigmoid.
    
    If:
        a = sigmoid(z)
    Then:
        dσ/dz = a * (1 - a)
    """
    return a * (1 - a)

# 4. Forward Propagation

def forward_propagation(X):
    """
    Forward pass through the network.
    
    Layer 1:
        Z1 = XW1 + b1
        A1 = sigmoid(Z1)
    
    Layer 2:
        Z2 = A1W2 + b2
        A2 = sigmoid(Z2)
    
    A2 is final prediction.
    """
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2

# 5. Cross-Entropy Loss
def compute_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss:
    
    L = -1/m * Σ [ y log(y_hat) + (1-y) log(1-y_hat) ]
    
    This measures how far predictions are from true labels.
    """
    m = y_true.shape[0]
    epsilon = 1e-8  # to avoid log(0)
    loss = - (1/m) * np.sum(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )
    return loss

# 6. Backpropagation
def backward_propagation(X, y, Z1, A1, Z2, A2):
    """
    Backpropagation computes gradients using chain rule.
    
    For output layer:
        dZ2 = A2 - y
        dW2 = A1^T * dZ2
        db2 = sum(dZ2)
    
    For hidden layer:
        dA1 = dZ2 * W2^T
        dZ1 = dA1 * sigmoid_derivative(A1)
        dW1 = X^T * dZ1
        db1 = sum(dZ1)
    """
    m = X.shape[0]

    # Output layer gradients
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# 7. Gradient Descent Update
def update_parameters(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

epochs = 1000
learning_rate = 0.1
losses = []

for epoch in range(epochs):
    # Forward pass
    Z1, A1, Z2, A2 = forward_propagation(X)

    # Compute loss
    loss = compute_loss(y, A2)
    losses.append(loss)

    # Backward pass
    dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, A2)

    # Update weights
    update_parameters(dW1, db1, dW2, db2, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 9. Plot Loss Curve

plt.figure()
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# 10. Visualize Decision Boundary

def plot_decision_boundary():
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    _, _, _, probs = forward_propagation(grid)
    Z = probs.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
    plt.title("Decision Boundary (XOR)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


plot_decision_boundary()


# 11. Final Predictions
_, _, _, predictions = forward_propagation(X)
print("\nFinal Predictions:")
print(predictions.round())