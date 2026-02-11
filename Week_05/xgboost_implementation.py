# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# 2. Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Dataset shape:", X.shape)


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Train Base XGBoost Model
xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# eval_set allows tracking training history
xgb_base.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# Predictions
y_pred_base = xgb_base.predict(X_test)

print("\nBase XGBoost Accuracy:",
      accuracy_score(y_test, y_pred_base))


# 5. Hyperparameter Tuning using GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(
    XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearch:")
print(grid_search.best_params_)

best_xgb = grid_search.best_estimator_

# Evaluate best model
y_pred = best_xgb.predict(X_test)

print("\nBest XGBoost Accuracy:",
      accuracy_score(y_test, y_pred))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred))


# 6. Plot Training History (Log Loss)
results = xgb_base.evals_result()

plt.figure()
plt.plot(results['validation_0']['logloss'], label='Train')
plt.plot(results['validation_1']['logloss'], label='Test')
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("XGBoost Training History")
plt.legend()
plt.show()


# 7. Feature Importance Plot
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (XGBoost)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]),
           feature_names[indices],
           rotation=90)
plt.tight_layout()
plt.show()


# 8. SHAP Explanation
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)


# 9. Compare with Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_pred))


# 10. Save Model
joblib.dump(best_xgb, "best_xgboost_model.pkl")
best_xgb.save_model("best_xgboost_model.json")

print("\nModel saved as:")
print(" - best_xgboost_model.pkl")
print(" - best_xgboost_model.json")
