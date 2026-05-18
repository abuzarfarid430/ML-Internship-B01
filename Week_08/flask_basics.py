from flask import Flask, request, jsonify

app = Flask(__name__)

# Simple in-memory storage - just a list of items
# Each item is a dict with an id and some data
data_store = []
next_id = 1  # we'll use this to auto-increment IDs


@app.route("/", methods=["GET"])
def home():
    """Welcome endpoint - just to confirm the API is running."""
    return jsonify({
        "message": "Welcome to the Flask API!",
        "endpoints": {
            "GET /": "This page",
            "GET /data": "List all items",
            "POST /data": "Add a new item",
            "PUT /data/<id>": "Update an item by ID",
            "DELETE /data/<id>": "Delete an item by ID"
        }
    })


@app.route("/data", methods=["GET"])
def get_all_data():
    """Return everything in the data store."""
    return jsonify({
        "items": data_store,
        "count": len(data_store)
    })


@app.route("/data", methods=["POST"])
def add_item():
    """Add a new item to the store. Expects JSON body with at least a 'name' field."""
    global next_id

    body = request.get_json()

    # basic check - make sure they sent something
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    if "name" not in body:
        return jsonify({"error": "Field 'name' is required"}), 400

    new_item = {
        "id": next_id,
        "name": body["name"],
        "description": body.get("description", ""),  # optional field
    }

    data_store.append(new_item)
    next_id += 1

    return jsonify({
        "message": "Item added successfully",
        "item": new_item
    }), 201


@app.route("/data/<int:item_id>", methods=["PUT"])
def update_item(item_id):
    """Update an existing item by its ID."""
    body = request.get_json()

    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    # find the item with the matching id
    for item in data_store:
        if item["id"] == item_id:
            # only update fields that were provided
            if "name" in body:
                item["name"] = body["name"]
            if "description" in body:
                item["description"] = body["description"]

            return jsonify({
                "message": "Item updated",
                "item": item
            })

    return jsonify({"error": f"Item with id {item_id} not found"}), 404


@app.route("/data/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    """Delete an item by its ID."""
    global data_store

    original_length = len(data_store)
    data_store = [item for item in data_store if item["id"] != item_id]

    if len(data_store) == original_length:
        return jsonify({"error": f"Item with id {item_id} not found"}), 404

    return jsonify({"message": f"Item {item_id} deleted successfully"})


if __name__ == "__main__":
    # debug=True so it auto-reloads when we make changes
    app.run(debug=True, port=5000)
