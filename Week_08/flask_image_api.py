"""
Flask Image Classification API using TFLite

NOTE: This uses MobileNetV2 from TensorFlow as the classification model.
The model is downloaded automatically on first run (~14MB).

Install dependencies:
    pip install flask tensorflow pillow numpy requests
"""

import io
import os
import json
import urllib.request

import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

# TFLite can be used via full tensorflow or the lighter tflite-runtime package
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)

# -----------------------------------------------
# Model setup - download MobileNetV2 if needed
# -----------------------------------------------
MODEL_PATH = "mobilenet_v2.tflite"
LABELS_PATH = "imagenet_labels.txt"

MOBILENET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "models/tflite/mobilenet_v2_1.0_224.tflite"
)
LABELS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "models/tflite/task_library/image_classification/android/mobilenet_v2_1.0_224_1_metadata_1.tflite"
)

# We'll use a local bundled labels file instead - easier
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/tensorflow/master/"
    "tensorflow/lite/java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt"
)


def download_model_files():
    """Download model and labels if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading MobileNetV2 TFLite model (~14MB)...")
        urllib.request.urlretrieve(MOBILENET_URL, MODEL_PATH)
        print("Model downloaded.")

    if not os.path.exists(LABELS_PATH):
        print("Downloading ImageNet labels...")
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, LABELS_PATH)
        print("Labels downloaded.")


def load_labels(path):
    """Load class labels from a text file (one label per line)."""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


# Download and load everything at startup
download_model_files()

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Expected input shape: (1, 224, 224, 3)
input_shape = input_details[0]["shape"]
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]

labels = load_labels(LABELS_PATH)

print(f"Model ready. Input shape: {input_shape}")


# -----------------------------------------------
# Helper functions
# -----------------------------------------------

def preprocess_image(image_bytes):
    """
    Convert raw image bytes to a tensor the model can consume.
    - Opens the image (handles jpg, png, webp, etc.)
    - Resizes to 224x224
    - Normalizes pixel values to [0, 1]
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)  # add batch dimension


def run_inference(input_tensor):
    """Run the TFLite interpreter on a preprocessed image tensor."""
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]  # remove batch dimension


def get_top_predictions(scores, top_k=5):
    """Return the top-k predictions as a list of dicts."""
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for i in top_indices:
        label = labels[i] if i < len(labels) else f"class_{i}"
        results.append({
            "rank": len(results) + 1,
            "label": label,
            "confidence": round(float(scores[i]), 4)
        })
    return results


# -----------------------------------------------
# Routes
# -----------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Image Classification API (MobileNetV2)",
        "usage": {
            "single_image": "POST /classify with an image file (field name: 'image')",
            "batch": "POST /classify_batch with multiple images (field name: 'images')"
        }
    })


@app.route("/classify", methods=["POST"])
def classify_single():
    """
    Classify a single uploaded image.
    Returns top 5 predicted classes with confidence scores.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Send a file with field name 'image'"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Empty filename - please select an image"}), 400

    try:
        image_bytes = image_file.read()
        input_tensor = preprocess_image(image_bytes)
        scores = run_inference(input_tensor)
        predictions = get_top_predictions(scores, top_k=5)

        return jsonify({
            "filename": image_file.filename,
            "top_predictions": predictions,
            "model": "MobileNetV2"
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


@app.route("/classify_batch", methods=["POST"])
def classify_batch():
    """
    Classify multiple images at once.
    Send multiple files under the field name 'images'.
    """
    if "images" not in request.files:
        return jsonify({"error": "No images found. Send files with field name 'images'"}), 400

    files = request.files.getlist("images")

    if len(files) == 0:
        return jsonify({"error": "No files received"}), 400

    results = []
    errors = []

    for image_file in files:
        try:
            image_bytes = image_file.read()
            input_tensor = preprocess_image(image_bytes)
            scores = run_inference(input_tensor)
            predictions = get_top_predictions(scores, top_k=5)

            results.append({
                "filename": image_file.filename,
                "top_predictions": predictions
            })

        except Exception as e:
            errors.append({
                "filename": image_file.filename,
                "error": str(e)
            })

    return jsonify({
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    })


if __name__ == "__main__":
    app.run(debug=True, port=5002)
