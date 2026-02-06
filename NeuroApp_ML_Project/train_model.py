from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)

# Save model
model.save("best_plate_model.pt")
