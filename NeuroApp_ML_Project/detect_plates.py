import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO

# Load OCR and YOLO model
reader = easyocr.Reader(['en'])
model = YOLO("best_plate_model.pt")

def detect_and_read_plates(video_path):
    cap = cv2.VideoCapture(video_path)
    results_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame)

        for box in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop plate
            plate_img = frame[y1:y2, x1:x2]

            if plate_img.size == 0:
                continue

            # OCR
            ocr_result = reader.readtext(plate_img)
            plate_text = " ".join([res[1] for res in ocr_result])

            results_list.append({
                "frame": frame_count,
                "plate_number": plate_text,
                "confidence": round(conf, 2),
                "bbox": f"{x1},{y1},{x2},{y2}"
            })

    cap.release()
    return results_list


def save_to_csv(results, output_path="plates_detected.csv"):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print("CSV saved:", output_path)
