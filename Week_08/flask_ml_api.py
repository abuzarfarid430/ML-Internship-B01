import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# -------------------------
# Load the model at startup
# -------------------------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found! Please run 'python train_model.py' first."
    )

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]

print(f"Model loaded successfully. Features: {feature_names}")


# -------------------------
# Routes
# -------------------------

@app.route("/", methods=["GET"])
def index():
    """Render a simple HTML form for testing the API manually."""
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON input with 4 iris features and return the predicted flower species.

    Expected input:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Please send JSON data in the request body"}), 400

        # map our friendly key names to the sklearn feature names
        key_map = {
            "sepal_length": "sepal length (cm)",
            "sepal_width": "sepal width (cm)",
            "petal_length": "petal length (cm)",
            "petal_width": "petal width (cm)",
        }

        # collect features in the right order
        features = []
        missing = []
        for short_key, full_name in key_map.items():
            if short_key not in data:
                missing.append(short_key)
            else:
                features.append(float(data[short_key]))

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing,
                "required": list(key_map.keys())
            }), 400

        # make the prediction
        prediction_index = model.predict([features])[0]
        prediction_proba = model.predict_proba([features])[0]

        # build a nice response
        probabilities = {
            name: round(float(prob), 4)
            for name, prob in zip(target_names, prediction_proba)
        }

        return jsonify({
            "prediction": target_names[prediction_index],
            "confidence": round(float(max(prediction_proba)), 4),
            "all_probabilities": probabilities,
            "input_received": data
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}. All features must be numbers."}), 400
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return info about the loaded model - useful for debugging."""
    return jsonify({
        "model_type": type(model).__name__,
        "features": feature_names,
        "classes": target_names,
        "feature_count": len(feature_names)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
