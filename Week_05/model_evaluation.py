import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold,
    learning_curve, validation_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# 2. Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42)
}

# 3. K-Fold Cross-Validation (k=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n--- K-Fold Cross-Validation ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    conf_interval = 1.96 * std_score / np.sqrt(len(scores))
    print(f"{name}: Mean Accuracy={mean_score:.4f}, Std={std_score:.4f}, 95% CI=±{conf_interval:.4f}")

# 4. Stratified K-Fold (for imbalanced data)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n--- Stratified K-Fold Cross-Validation ---")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skfold)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    conf_interval = 1.96 * std_score / np.sqrt(len(scores))
    print(f"{name}: Mean Accuracy={mean_score:.4f}, Std={std_score:.4f}, 95% CI=±{conf_interval:.4f}")

# 5. Learning Curves
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

print("\n--- Learning Curves ---")
for name, model in models.items():
    plot_learning_curve(model, X, y, title=f"Learning Curve ({name})")

# 6. Validation Curves
def plot_validation_curve(model, X, y, param_name, param_range, title):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring="accuracy", n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.plot(param_range, train_mean, marker='o', label='Training score')
    plt.plot(param_range, test_mean, marker='o', label='Cross-validation score')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Validation curve for Logistic Regression (C)
print("\n--- Validation Curve: Logistic Regression (C) ---")
param_range = [0.01, 0.1, 1, 10, 100]
plot_validation_curve(LogisticRegression(max_iter=1000), X, y, 'C', param_range, "Logistic Regression: C vs Accuracy")

# Validation curve for Random Forest (n_estimators)
print("\n--- Validation Curve: Random Forest (n_estimators) ---")
param_range = [10, 50, 100, 200]
plot_validation_curve(RandomForestClassifier(random_state=42), X, y, 'n_estimators', param_range, "Random Forest: n_estimators vs Accuracy")

# Validation curve for SVM (gamma)
print("\n--- Validation Curve: SVM (gamma) ---")
param_range = [0.001, 0.01, 0.1, 1]
plot_validation_curve(SVC(kernel='rbf'), X, y, 'gamma', param_range, "SVM RBF: gamma vs Accuracy")
