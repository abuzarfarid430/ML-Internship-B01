# 🚗 License Plate Detection & Recognition System

**YOLOv8 + EasyOCR + Flask**

## 📌 Project Overview

This project is a **Deep Learning--based License Plate Detection and
Recognition system**.\
It detects vehicle license plates from videos, reads the plate number
using OCR, and exposes the functionality through a **Flask REST API**.\
The detected results are saved in **CSV format** for further analysis.

---

## 🧠 Technologies Used

- YOLOv8 -- License plate detection\
- EasyOCR -- Optical Character Recognition\
- OpenCV -- Video processing\
- Flask -- REST API\
- Pandas -- Data handling

---

## 📂 Project Structure

    license_plate_project/
    │
    ├── dataset/
    │   ├── images/train
    │   ├── images/val
    │   ├── labels/train
    │   └── labels/val
    │
    ├── data.yaml
    ├── train_model.py
    ├── detect_plates.py
    ├── app.py
    ├── test_api.py
    ├── requirements.txt
    └── README.md

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Use any YOLO-format license plate dataset (e.g., Roboflow).\
Update `data.yaml` accordingly.

```yaml
train: dataset/images/train
val: dataset/images/val
nc: 1
names: ["license_plate"]
```

---

## 🏋️ Model Training

```bash
python train_model.py
```

The trained model will be saved as:

    best_plate_model.pt

---

## 🌐 Run Flask Server

```bash
python app.py
```

Server URL:

    http://127.0.0.1:5000

---

## 🚀 API Usage

### POST /detect

```json
{
  "video_path": "cars_video.mp4"
}
```

---

## 📄 Output

Results are stored in:

    plates_detected.csv

Format:

    frame,plate_number,confidence,bbox
    15,ABC123,0.89,100,200,300,400

---

## 🎯 Author

**Abu Zar Farid**\
Computer Science Student

---

## 📜 License

Educational use only.
