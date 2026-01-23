# Week 4 – Classification Algorithms (Machine Learning)

## 📌 Overview

This week focuses on **classification algorithms** and **evaluation metrics** using popular machine learning models. The tasks cover binary classification, multi-class classification, model comparison, hyperparameter tuning, visualization, and performance analysis using real-world datasets.

All implementations are done using **Python** and **scikit-learn**, with proper preprocessing, evaluation, and visualization.

## 🧪 Task 4.1: Logistic Regression (Binary Classification)

### 📊 Dataset

- **Breast Cancer Wisconsin Dataset**
- Binary classes: Malignant (0) and Benign (1)

### ⚙️ Steps Performed

- Data loading and exploration
- Train-test split
- Feature scaling using StandardScaler
- Model training using Logistic Regression
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Visualization:
  - Confusion Matrix
  - ROC Curve
- Model persistence using Pickle

### 📈 Key Outcome

Logistic Regression achieved high accuracy and strong ROC-AUC, making it suitable for medical binary classification tasks.

---

## 🧪 Task 4.2: K-Nearest Neighbors (KNN) Classification

### 📊 Dataset

- **Iris Dataset**

### ⚙️ Steps Performed

- Selected only **two features** for visualization
- Tested multiple values of K: **[1, 3, 5, 7, 11, 15]**
- Implemented two distance metrics:
  - Euclidean
  - Manhattan
- Evaluated accuracy for each K
- Visualizations:
  - Accuracy vs K plot
  - Decision boundary visualization (K = 5)
- Created a comparison table
- Identified optimal K value

### 📈 Key Outcome

Moderate K values (5 or 7) provided the best balance between bias and variance. Euclidean and Manhattan distances performed similarly on the Iris dataset.

---

## 🧪 Task 4.3: Decision Tree Classifier

### 📊 Dataset

- **Wine Dataset**

### ⚙️ Steps Performed

- Trained Decision Tree with different `max_depth` values:
  - 3, 5, 10, None
- Applied pruning by limiting tree depth
- Evaluated accuracy for each depth
- Visualizations:
  - Accuracy vs Depth
  - Tree structure using `plot_tree`
  - Feature importance bar chart
- Exported tree visualization as an image
- Saved the best-performing model

### 📈 Key Outcome

A moderate tree depth provided the best performance, preventing overfitting while maintaining interpretability. Feature importance analysis highlighted the most influential chemical properties.

---

## 🧪 Task 4.4: Multi-class Classification & Evaluation Metrics

### 📊 Dataset

- **Digits Dataset (0–9)**

### ⚙️ Models Implemented

- Logistic Regression (One-vs-Rest)
- K-Nearest Neighbors
- Decision Tree Classifier

### ⚙️ Steps Performed

- Multi-class classification
- Feature scaling where required
- Generated predictions for all models
- Evaluated models using:
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix
- Visualized:
  - Confusion matrices side-by-side
  - Accuracy comparison bar chart

### 📈 Key Outcome

KNN achieved the highest accuracy due to effective neighborhood-based learning. Logistic Regression performed competitively, while Decision Trees offered interpretability with slightly lower generalization.

---

## 📊 Comparative Summary

| Model               | Strengths                     | Weaknesses                   |
| ------------------- | ----------------------------- | ---------------------------- |
| Logistic Regression | Simple, fast, interpretable   | Limited to linear boundaries |
| KNN                 | High accuracy, non-parametric | Computationally expensive    |
| Decision Tree       | Highly interpretable          | Prone to overfitting         |

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## ▶️ How to Run

```bash
python logistic_regression.py
python knn_classification.py
python decision_tree.py
python multiclass_classification.py
```
