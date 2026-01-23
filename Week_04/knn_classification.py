# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# ===============================
# 2. Load Iris Dataset
# ===============================
iris = load_iris()

# Select only 2 features for visualization
# Feature 0: sepal length, Feature 1: sepal width
X = iris.data[:, :2]
y = iris.target

feature_names = iris.feature_names[:2]
class_names = iris.target_names

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ===============================
# 4. K Values
# ===============================
k_values = [1, 3, 5, 7, 11, 15]

# Lists to store accuracy
euclidean_accuracies = []
manhattan_accuracies = []

# ===============================
# 5. Train Models for Different K
# ===============================
for k in k_values:
    # Euclidean Distance
    knn_eu = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn_eu.fit(X_train, y_train)
    y_pred_eu = knn_eu.predict(X_test)
    euclidean_accuracies.append(accuracy_score(y_test, y_pred_eu))

    # Manhattan Distance
    knn_man = KNeighborsClassifier(n_neighbors=k, metric="manhattan")
    knn_man.fit(X_train, y_train)
    y_pred_man = knn_man.predict(X_test)
    manhattan_accuracies.append(accuracy_score(y_test, y_pred_man))

# ===============================
# 6. Plot Accuracy vs K
# ===============================
plt.figure(figsize=(8, 5))
plt.plot(k_values, euclidean_accuracies, marker="o", label="Euclidean")
plt.plot(k_values, manhattan_accuracies, marker="s", label="Manhattan")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K for KNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 7. Decision Boundary (K = 5)
# ===============================
def plot_decision_boundary(X, y, model, title):
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ["red", "green", "blue"]

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        edgecolor="k"
    )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Train KNN for K=5 using Euclidean distance
knn_k5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_k5.fit(X_train, y_train)

plot_decision_boundary(
    X_train,
    y_train,
    knn_k5,
    title="Decision Boundary (K=5, Euclidean)"
)

# ===============================
# 8. Comparison Table
# ===============================
comparison_df = pd.DataFrame({
    "K Value": k_values,
    "Euclidean Accuracy": euclidean_accuracies,
    "Manhattan Accuracy": manhattan_accuracies
})

print("\nComparison Table:")
print(comparison_df)

# ===============================
# 9. Identify Optimal K
# ===============================
best_eu_k = comparison_df.loc[
    comparison_df["Euclidean Accuracy"].idxmax(), "K Value"
]

best_man_k = comparison_df.loc[
    comparison_df["Manhattan Accuracy"].idxmax(), "K Value"
]

print("\nOptimal K Values:")
print(f"Best K (Euclidean): {best_eu_k}")
print(f"Best K (Manhattan): {best_man_k}")
