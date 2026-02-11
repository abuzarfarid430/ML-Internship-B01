# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# 2. Load dataset (Heart disease alternative: breast cancer dataset)
data = load_breast_cancer()
X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Create base Random Forest model
rf_base = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True  # Enable OOB score
)

rf_base.fit(X_train, y_train)

print("\nBase Random Forest Accuracy:",
      accuracy_score(y_test, rf_base.predict(X_test)))
print("Base OOB Score:", rf_base.oob_score_)


# 5. GridSearchCV for Hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 10, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearch:")
print(grid_search.best_params_)


# 6. Train model with best parameters
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

print("\nBest Random Forest Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))


# 7. Compare with single Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, dt_pred))


# 8. Plot Feature Importances
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]),
           feature_names[indices],
           rotation=90)
plt.tight_layout()
plt.show()


# 9. Plot OOB Error vs n_estimators
oob_errors = []

n_estimators_range = [10, 50, 100, 200]

for n in n_estimators_range:
    rf = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    oob_error = 1 - rf.oob_score_
    oob_errors.append(oob_error)

plt.figure()
plt.plot(n_estimators_range, oob_errors, marker='o')
plt.xlabel("Number of Estimators")
plt.ylabel("OOB Error")
plt.title("OOB Error vs Number of Trees")
plt.show()


# 10. Save best model
joblib.dump(best_rf, "best_random_forest_model.pkl")
print("\nBest model saved as 'best_random_forest_model.pkl'")
