# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ===============================
# 2. Load Digits Dataset
# ===============================
digits = load_digits()

X = digits.data
y = digits.target
class_names = [str(i) for i in range(10)]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. Feature Scaling
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 5. Train Models
# ===============================

# Logistic Regression (One-vs-Rest)
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)  # Tree does not require scaling

# ===============================
# 6. Predictions
# ===============================
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_dt = dt.predict(X_test)

# ===============================
# 7. Accuracy Scores
# ===============================
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_dt = accuracy_score(y_test, y_pred_dt)

print("Accuracy Scores:")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"KNN               : {acc_knn:.4f}")
print(f"Decision Tree     : {acc_dt:.4f}")

# ===============================
# 8. Classification Reports
# ===============================
print("\nClassification Report - Logistic Regression")
print(classification_report(y_test, y_pred_lr))

print("\nClassification Report - KNN")
print(classification_report(y_test, y_pred_knn))

print("\nClassification Report - Decision Tree")
print(classification_report(y_test, y_pred_dt))

# ===============================
# 9. Confusion Matrices
# ===============================
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm_lr, annot=False, cmap="Blues", ax=axes[0])
axes[0].set_title("Logistic Regression")

sns.heatmap(cm_knn, annot=False, cmap="Greens", ax=axes[1])
axes[1].set_title("KNN")

sns.heatmap(cm_dt, annot=False, cmap="Oranges", ax=axes[2])
axes[2].set_title("Decision Tree")

for ax in axes:
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ===============================
# 10. Accuracy Comparison Bar Chart
# ===============================
models = ["Logistic Regression", "KNN", "Decision Tree"]
accuracies = [acc_lr, acc_knn, acc_dt]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Digits Dataset)")
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.show()
