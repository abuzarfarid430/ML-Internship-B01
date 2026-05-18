"""
Run this script once to train a simple Iris classifier and save it as model.pkl
Usage: python train_model.py
"""

import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save():
    # load the built-in iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train a simple random forest - works well out of the box
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained! Test accuracy: {accuracy:.2%}")

    # save everything we need to make predictions later
    model_data = {
        "model": model,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names.tolist(),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("Model saved to model.pkl")
    print("\nFeatures expected by the model:")
    for i, name in enumerate(iris.feature_names):
        print(f"  {i+1}. {name}")

    print("\nPossible predictions:")
    for name in iris.target_names:
        print(f"  - {name}")


if __name__ == "__main__":
    train_and_save()
