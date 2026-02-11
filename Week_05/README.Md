# Week 5: Ensemble Methods & Advanced Algorithms

This repository contains Python implementations for **Week 5 tasks** on Ensemble Methods, Gradient Boosting, Support Vector Machines (SVM), and Model Evaluation & Cross-Validation.

---

## 📂 Tasks Overview

### **Task 5.1: Random Forest Classifier**

- **File:** `random_forest.py`
- **Description:**
  - Implements Random Forest on a classification dataset (Breast Cancer).
  - Experiments with different numbers of estimators (`10, 50, 100, 200`) and `max_depth`.
  - Uses `GridSearchCV` for hyperparameter tuning.
  - Compares performance with a single Decision Tree.
  - Plots:
    - Feature importances
    - Out-of-Bag (OOB) error vs number of estimators
  - Saves the best model as `.pkl`.

---

### **Task 5.2: Gradient Boosting (XGBoost)**

- **File:** `xgboost_implementation.py`
- **Description:**
  - Implements XGBoost classifier on a dataset.
  - Performs hyperparameter tuning for `learning_rate`, `max_depth`, and `n_estimators` using `GridSearchCV`.
  - Plots:
    - Training history (log loss)
    - Feature importance
    - SHAP summary plot for model interpretability
  - Compares performance with Random Forest.
  - Saves the best model in `.pkl` and `.json` formats.

---

### **Task 5.3: Support Vector Machines (SVM)**

- **File:** `svm_classification.py`
- **Description:**
  - Implements SVM with different kernels (`linear`, `RBF`, `polynomial`) on a 2D synthetic dataset.
  - Visualizes decision boundaries for each kernel.
  - Experiments with hyperparameters `C` and `gamma`.
  - Plots:
    - Accuracy vs `C` and `gamma`
    - Decision boundaries
  - Compares kernel performances and saves the best performing model (`.pkl`).

---

### **Task 5.4: Model Evaluation & Cross-Validation**

- **File:** `model_evaluation.py`
- **Description:**
  - Implements evaluation strategies on multiple models (Logistic Regression, Random Forest, SVM):
    - K-Fold Cross-Validation (k=5)
    - Stratified K-Fold (for imbalanced datasets)
    - Confidence intervals for model performance
  - Generates **Learning Curves** to analyze underfitting/overfitting.
  - Generates **Validation Curves** to study impact of key hyperparameters.
  - Provides visualizations for training and validation performance.
  - Recommends evaluation strategies based on dataset characteristics.

---

## 📈 Visualizations

- Feature Importance (Random Forest & XGBoost)
- OOB Error vs Trees (Random Forest)
- Training History (XGBoost)
- SHAP Summary Plot (XGBoost)
- Decision Boundaries (SVM)
- Learning Curves (All Models)
- Validation Curves (Key Hyperparameters)

---

## 🛠 Installation & Requirements

- Python >= 3.8
- Required Libraries:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost shap joblib
```
