# Real-Time Object Detection API

Object detection system built with YOLOv8 that identifies and localizes common objects in images.
The system produces annotated images with bounding boxes and structured JSON outputs containing labels and confidence scores.

This project focuses on building a production-oriented computer vision pipeline that can later be exposed through a REST API and deployed using modern backend tooling.

---

## Features

- Image-based object detection using YOLOv8  
- Support for 80 object classes (people, vehicles, animals, everyday objects)  
- Bounding box visualization on detected objects  
- Confidence scores per detection  
- Structured JSON output for downstream processing  

---

## Tech Stack

- Python 3.10  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  
- FastAPI (planned)  
- PostgreSQL (planned)  
- Docker and Railway (planned)  

---

## Project Structure

```
object-detection-api/
├── src/
│   ├── detector.py
│   ├── test_detector.py
│   └── test_yolo.py
├── images/        # local test images (gitignored)
├── results/       # detection outputs (gitignored)
├── models/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local Setup

```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Object Detection

Place `.jpg` or `.png` images into the `images/` directory and run:

```
python -m src.test_detector
```

Detection results are saved to the `results/` directory:
- Annotated images with bounding boxes
- JSON files containing detection metadata

---

## Output Format

Each JSON result contains:
- image path  
- image shape  
- total number of detected objects  
- list of detections with:
  - class name  
  - confidence score  
  - bounding box coordinates `[x1, y1, x2, y2]`  

---

## Notes

- YOLO model weights are downloaded automatically on first run  
- Generated images and detection results are excluded from version control  
- API layer and deployment setup will be added incrementally  

---

## Next Steps

- Build a FastAPI service for image uploads  
- Add database logging for detection metadata  
- Containerize the application using Docker  
- Deploy the service using Railway  
