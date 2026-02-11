import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Create 2D dataset for visualization
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=42
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# 2. Train SVM with different kernels
kernels = ['linear', 'rbf', 'poly']
best_models = {}
best_scores = {}

for kernel in kernels:
    svc = SVC(kernel=kernel, random_state=42)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    best_models[kernel] = svc
    best_scores[kernel] = acc
    print(f"{kernel.upper()} Kernel Accuracy: {acc:.4f}")
    plot_decision_boundary(svc, X, y, title=f"SVM Decision Boundary ({kernel} kernel)")

# 3. Hyperparameter Tuning for RBF kernel
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_svc = grid_search.best_estimator_

print("\nBest RBF SVM Parameters:")
print(grid_search.best_params_)

# Evaluate best model
y_pred_best = best_svc.predict(X_test)
print("\nBest RBF SVM Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification Report:\n", classification_report(y_test, y_pred_best))

# Decision Boundary for Best Model
plot_decision_boundary(best_svc, X, y, title="Best SVM (RBF) Decision Boundary")

# 4. Accuracy vs C and gamma plots
scores = grid_search.cv_results_['mean_test_score']

# Reshape scores to 4x4 (C x gamma)
scores_matrix = scores.reshape(len(param_grid['C']), len(param_grid['gamma']))

plt.figure(figsize=(8, 6))
for i, C in enumerate(param_grid['C']):
    plt.plot(param_grid['gamma'], scores_matrix[i], marker='o', label=f"C={C}")
plt.xlabel("Gamma")
plt.ylabel("Mean CV Accuracy")
plt.title("SVM Accuracy vs Gamma for Different C Values")
plt.legend()
plt.show()

# 5. Comparison Table
comparison = pd.DataFrame({
    'Kernel': kernels + ['Best RBF (Tuned)'],
    'Accuracy': [best_scores[k] for k in kernels] + [accuracy_score(y_test, y_pred_best)]
})
print("\nKernel Comparison Table:\n", comparison)

# 6. Save best model
joblib.dump(best_svc, "best_svm_model.pkl")
print("\nBest SVM model saved as 'best_svm_model.pkl'")
