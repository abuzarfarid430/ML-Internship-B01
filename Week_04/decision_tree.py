# ===============================
# 1. Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# ===============================
# 2. Load Dataset
# ===============================
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 4. Max Depth Values for Pruning
# ===============================
depth_values = [3, 5, 10, None]
accuracies = {}
models = {}

# ===============================
# 5. Train Decision Trees
# ===============================
for depth in depth_values:
    clf = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies[depth] = acc
    models[depth] = clf

    print(f"Max Depth = {depth}, Accuracy = {acc:.4f}")

# ===============================
# 6. Plot Accuracy vs Depth
# ===============================
depth_labels = ["3", "5", "10", "None"]
accuracy_values = list(accuracies.values())

plt.figure(figsize=(8, 5))
plt.plot(depth_labels, accuracy_values, marker="o")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs Max Depth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================
# 7. Identify Best Model
# ===============================
best_depth = max(accuracies, key=accuracies.get)
best_model = models[best_depth]

print(f"\nBest max_depth: {best_depth}")
print(f"Best Accuracy: {accuracies[best_depth]:.4f}")

# ===============================
# 8. Visualize Decision Tree
# ===============================
plt.figure(figsize=(20, 10))
plot_tree(
    best_model,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True
)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.tight_layout()
plt.savefig("decision_tree_visualization.png", dpi=300)
plt.show()

# ===============================
# 9. Feature Importance
# ===============================
importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": wine.feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)
plt.title("Feature Importance (Decision Tree)")
plt.tight_layout()
plt.show()

# ===============================
# 10. Save Best Model
# ===============================
with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model saved as decision_tree_model.pkl")

# ===============================
# 11. Save Feature Importance Table
# ===============================
importance_df.to_csv("feature_importance.csv", index=False)
print("Feature importance saved as feature_importance.csv")
