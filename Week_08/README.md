# Week 8: Flask Backend & API Development

A collection of Flask APIs built as part of Week 8 coursework. Covers basic REST APIs, ML model deployment, image classification, and database-backed APIs with authentication.

---

## Project Structure

```
week8_flask/
├── flask_basics.py       # Task 8.1 - Basic REST API with in-memory storage
├── flask_ml_api.py       # Task 8.2 - ML model prediction API
├── train_model.py        # Task 8.2 - Script to train & save the Iris model
├── flask_image_api.py    # Task 8.3 - TFLite image classification API
├── flask_rest_api.py     # Task 8.4 - Full REST API with SQLite + auth + Swagger
├── templates/
│   └── index.html        # HTML form for testing the ML API
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo and enter the folder
git clone <your-repo-url>
cd week8_flask

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Task 8.1 – Flask Basics & First API

**File:** `flask_basics.py`  
**Port:** 5000

A simple CRUD API using an in-memory Python list. No database needed — data resets when the server restarts.

### Run

```bash
python flask_basics.py
```

### Endpoints

#### `GET /`
Returns a welcome message and lists all available endpoints.

```bash
curl http://localhost:5000/
```

**Response:**
```json
{
  "message": "Welcome to the Flask API!",
  "endpoints": { ... }
}
```

---

#### `GET /data`
Returns all stored items.

```bash
curl http://localhost:5000/data
```

**Response:**
```json
{
  "items": [
    { "id": 1, "name": "item one", "description": "a test item" }
  ],
  "count": 1
}
```

---

#### `POST /data`
Add a new item. Requires a `name` field; `description` is optional.

```bash
curl -X POST http://localhost:5000/data \
  -H "Content-Type: application/json" \
  -d '{"name": "item one", "description": "a test item"}'
```

**Response (201):**
```json
{
  "message": "Item added successfully",
  "item": { "id": 1, "name": "item one", "description": "a test item" }
}
```

---

#### `PUT /data/<id>`
Update an existing item by ID. Only include the fields you want to change.

```bash
curl -X PUT http://localhost:5000/data/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "updated name"}'
```

**Response:**
```json
{
  "message": "Item updated",
  "item": { "id": 1, "name": "updated name", "description": "a test item" }
}
```

---

#### `DELETE /data/<id>`
Delete an item by ID.

```bash
curl -X DELETE http://localhost:5000/data/1
```

**Response:**
```json
{ "message": "Item 1 deleted successfully" }
```

---

## Task 8.2 – ML Model Deployment with Flask

**File:** `flask_ml_api.py`  
**Port:** 5001

Serves predictions from a trained Iris flower classifier. The model is a Random Forest trained on the classic Iris dataset.

### Run

```bash
# Step 1: train and save the model (only needed once)
python train_model.py

# Step 2: start the API
python flask_ml_api.py
```

Open http://localhost:5001 in your browser to use the HTML test form.

### Endpoints

#### `GET /`
Renders the HTML form for manual testing.

---

#### `POST /predict`
Accepts flower measurements and returns the predicted species.

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Response:**
```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "all_probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

#### `GET /model-info`
Returns info about the loaded model.

```bash
curl http://localhost:5001/model-info
```

---

## Task 8.3 – Image Classification API with TFLite

**File:** `flask_image_api.py`  
**Port:** 5002

Classifies images using MobileNetV2 (downloaded automatically on first run). Returns the top 5 predicted ImageNet classes.

### Run

```bash
python flask_image_api.py
```

> The model (~14MB) and labels file will download automatically on first run.

### Endpoints

#### `POST /classify`
Upload a single image to classify. Supports jpg, png, webp, and other PIL-compatible formats.

```bash
curl -X POST http://localhost:5002/classify \
  -F "image=@/path/to/your/photo.jpg"
```

**Response:**
```json
{
  "filename": "photo.jpg",
  "model": "MobileNetV2",
  "top_predictions": [
    { "rank": 1, "label": "golden retriever", "confidence": 0.8743 },
    { "rank": 2, "label": "Labrador retriever", "confidence": 0.0512 },
    ...
  ]
}
```

---

#### `POST /classify_batch`
Upload multiple images at once.

```bash
curl -X POST http://localhost:5002/classify_batch \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.png"
```

**Response:**
```json
{
  "processed": 2,
  "failed": 0,
  "results": [ ... ],
  "errors": []
}
```

---

## Task 8.4 – RESTful API with Database Integration

**File:** `flask_rest_api.py`  
**Port:** 5003

A production-style API backed by SQLite. Features pagination, date/model filtering, API key authentication, request logging, and Swagger UI docs.

### Run

```bash
python flask_rest_api.py
```

Swagger UI: http://localhost:5003/docs  
Log file: `api.log`

### Authentication

All `/predictions` endpoints require an `X-API-Key` header.

| Key | Role |
|-----|------|
| `dev-key-123` | developer |
| `admin-key-456` | admin |

### Endpoints

#### `POST /predictions`
Save a new prediction record.

```bash
curl -X POST http://localhost:5003/predictions \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "iris_classifier",
    "input_data": "{\"sepal_length\": 5.1}",
    "result": "setosa",
    "confidence": 0.97
  }'
```

**Response (201):**
```json
{
  "message": "Prediction saved",
  "prediction": {
    "id": 1,
    "model_name": "iris_classifier",
    "result": "setosa",
    "confidence": 0.97,
    "created_at": "2024-01-15T10:30:00"
  }
}
```

---

#### `GET /predictions`
List predictions with optional filters and pagination.

```bash
# basic usage
curl http://localhost:5003/predictions \
  -H "X-API-Key: dev-key-123"

# with filters
curl "http://localhost:5003/predictions?model=iris&page=1&per_page=5" \
  -H "X-API-Key: dev-key-123"

# filter by date
curl "http://localhost:5003/predictions?date=2024-01-15" \
  -H "X-API-Key: dev-key-123"
```

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `page` | int | Page number (default: 1) |
| `per_page` | int | Results per page (default: 10, max: 100) |
| `model` | string | Filter by model name (partial match) |
| `date` | string | Filter by date in YYYY-MM-DD format |

---

#### `GET /predictions/<id>`
Get a single prediction by ID.

```bash
curl http://localhost:5003/predictions/1 \
  -H "X-API-Key: dev-key-123"
```

---

#### `DELETE /predictions/<id>`
Delete a prediction by ID.

```bash
curl -X DELETE http://localhost:5003/predictions/1 \
  -H "X-API-Key: dev-key-123"
```

---

## Notes

- All servers run in debug mode — not suitable for production as-is
- The SQLite database file (`predictions.db`) is created automatically in the same directory
- For production, replace the hardcoded API keys with a proper secrets manager
