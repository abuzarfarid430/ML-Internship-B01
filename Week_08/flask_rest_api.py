"""
RESTful API with SQLite database, pagination, filtering, API key auth, and Swagger docs.

Install:
    pip install flask flask-sqlalchemy flask-swagger-ui

Usage:
    python flask_rest_api.py

API Key: use 'dev-key-123' for testing (see VALID_API_KEYS below)
Swagger UI: http://localhost:5003/docs
"""

import os
import logging
import functools
from datetime import datetime

from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_swagger_ui import get_swaggerui_blueprint

# -----------------------------------------------
# App & DB setup
# -----------------------------------------------

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -----------------------------------------------
# Logging setup
# -----------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@app.before_request
def log_request():
    """Log every incoming request."""
    logger.info(f"{request.method} {request.path} | IP: {request.remote_addr}")


# -----------------------------------------------
# Database model
# -----------------------------------------------

class Prediction(db.Model):
    """Stores a single model prediction result."""
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    input_data = db.Column(db.Text, nullable=False)   # store as JSON string
    result = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "input_data": self.input_data,
            "result": self.result,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }


# Create tables if they don't exist
with app.app_context():
    db.create_all()
    logger.info("Database initialized.")


# -----------------------------------------------
# API Key authentication
# -----------------------------------------------

# In a real app, store these in a database or environment variables
VALID_API_KEYS = {
    "dev-key-123": "developer",
    "admin-key-456": "admin",
}


def require_api_key(f):
    """Decorator that checks for a valid API key in the request headers."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return jsonify({"error": "Missing API key. Add 'X-API-Key' header."}), 401

        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempt: {api_key}")
            return jsonify({"error": "Invalid API key"}), 403

        g.api_user = VALID_API_KEYS[api_key]
        return f(*args, **kwargs)

    return decorated


# -----------------------------------------------
# Routes
# -----------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Predictions REST API",
        "docs": "/docs",
        "note": "All /predictions endpoints require X-API-Key header"
    })


@app.route("/predictions", methods=["POST"])
@require_api_key
def create_prediction():
    """
    Save a new prediction to the database.

    Body:
    {
        "model_name": "iris_classifier",
        "input_data": "{\"sepal_length\": 5.1}",
        "result": "setosa",
        "confidence": 0.97
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required_fields = ["model_name", "input_data", "result"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400

    prediction = Prediction(
        model_name=data["model_name"],
        input_data=str(data["input_data"]),
        result=data["result"],
        confidence=data.get("confidence")
    )

    db.session.add(prediction)
    db.session.commit()

    logger.info(f"New prediction saved: id={prediction.id}, model={prediction.model_name}")

    return jsonify({
        "message": "Prediction saved",
        "prediction": prediction.to_dict()
    }), 201


@app.route("/predictions", methods=["GET"])
@require_api_key
def get_predictions():
    """
    Retrieve predictions with optional filtering and pagination.

    Query params:
    - page (int): Page number, default 1
    - per_page (int): Results per page, default 10, max 100
    - model (str): Filter by model name
    - date (str): Filter by date (YYYY-MM-DD format)
    """
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 10, type=int), 100)
    model_filter = request.args.get("model")
    date_filter = request.args.get("date")

    query = Prediction.query

    # apply filters if provided
    if model_filter:
        query = query.filter(Prediction.model_name.ilike(f"%{model_filter}%"))

    if date_filter:
        try:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d")
            next_day = datetime(filter_date.year, filter_date.month, filter_date.day + 1) \
                if filter_date.day < 28 else datetime(filter_date.year, filter_date.month + 1, 1)
            query = query.filter(
                Prediction.created_at >= filter_date,
                Prediction.created_at < next_day
            )
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    # order newest first
    query = query.order_by(Prediction.created_at.desc())

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        "predictions": [p.to_dict() for p in paginated.items],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": paginated.total,
            "pages": paginated.pages,
            "has_next": paginated.has_next,
            "has_prev": paginated.has_prev
        }
    })


@app.route("/predictions/<int:pred_id>", methods=["GET"])
@require_api_key
def get_prediction(pred_id):
    """Get a single prediction by ID."""
    prediction = Prediction.query.get(pred_id)

    if not prediction:
        return jsonify({"error": f"Prediction {pred_id} not found"}), 404

    return jsonify(prediction.to_dict())


@app.route("/predictions/<int:pred_id>", methods=["DELETE"])
@require_api_key
def delete_prediction(pred_id):
    """Delete a prediction by ID."""
    prediction = Prediction.query.get(pred_id)

    if not prediction:
        return jsonify({"error": f"Prediction {pred_id} not found"}), 404

    db.session.delete(prediction)
    db.session.commit()

    logger.info(f"Prediction {pred_id} deleted by {g.api_user}")
    return jsonify({"message": f"Prediction {pred_id} deleted"})


# -----------------------------------------------
# Swagger UI setup
# -----------------------------------------------

SWAGGER_URL = "/docs"
API_URL = "/swagger.json"

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={"app_name": "Predictions API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route("/swagger.json", methods=["GET"])
def swagger_spec():
    """Serve the OpenAPI specification."""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Predictions REST API",
            "version": "1.0.0",
            "description": "Store and retrieve ML model predictions"
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        },
        "security": [{"ApiKeyAuth": []}],
        "paths": {
            "/predictions": {
                "get": {
                    "summary": "List predictions",
                    "parameters": [
                        {"name": "page", "in": "query", "schema": {"type": "integer"}},
                        {"name": "per_page", "in": "query", "schema": {"type": "integer"}},
                        {"name": "model", "in": "query", "schema": {"type": "string"}},
                        {"name": "date", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {"200": {"description": "List of predictions"}}
                },
                "post": {
                    "summary": "Save a prediction",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_name": {"type": "string"},
                                        "input_data": {"type": "string"},
                                        "result": {"type": "string"},
                                        "confidence": {"type": "number"}
                                    },
                                    "required": ["model_name", "input_data", "result"]
                                }
                            }
                        }
                    },
                    "responses": {"201": {"description": "Prediction saved"}}
                }
            },
            "/predictions/{id}": {
                "get": {
                    "summary": "Get a prediction by ID",
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "Prediction details"}}
                },
                "delete": {
                    "summary": "Delete a prediction",
                    "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                    "responses": {"200": {"description": "Prediction deleted"}}
                }
            }
        }
    }
    return jsonify(spec)


if __name__ == "__main__":
    app.run(debug=True, port=5003)
