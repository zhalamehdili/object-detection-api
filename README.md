# Real-Time Object Detection API

This project implements an object detection system using YOLOv8 that identifies and localizes common objects in images. It produces annotated images with bounding boxes as well as structured JSON outputs containing class labels and confidence scores. The goal of this project is to build a clean and practical computer vision pipeline that is exposed through a REST API using FastAPI.

## Features

- Image-based object detection using YOLOv8  
- Support for 80 object classes including people, vehicles, animals and everyday objects  
- Bounding box visualization on detected objects  
- Confidence scores for each detection  
- Structured JSON output for detections  
- REST API with image upload support  
- Annotated image responses returned via HTTP  

## Tech Stack

- Python 3.10  
- Ultralytics YOLOv8  
- OpenCV  
- NumPy  
- FastAPI  
- Pydantic  
- Pytest  

## Project Structure

- object-detection-api  
  - src  
    - api.py – FastAPI application  
    - detector.py – YOLO detection logic  
    - schemas.py – Pydantic models  
    - test_detector.py  
    - test_yolo.py  
  - tests  
    - test_api.py – API tests  
  - images – local test images (gitignored)  
  - results – detection outputs (gitignored)  
  - temp – temporary upload files (gitignored)  
  - requirements.txt  
  - .gitignore  
  - README.md  

## Local Setup

- Create a virtual environment using Python 3.10  
- Activate the environment  
- Install dependencies from `requirements.txt`  

## Running the API

- Start the FastAPI server locally using uvicorn  
- Once running, interactive API documentation is available at:  
  - http://localhost:8000/docs  

## API Endpoints

- **GET /health**  
  - Returns the API status and whether the detection model is loaded  

- **GET /classes**  
  - Returns the list of all detectable object classes  

- **POST /detect**  
  - Uploads an image and returns JSON detection results including class names, confidence scores and bounding boxes  

- **POST /detect/annotated**  
  - Uploads an image and returns the annotated image with bounding boxes drawn on detected objects  

## Local Image Detection Without the API

- Images can be processed directly without running the API  
- Place jpg or png files in the `images` directory  
- Run the local detector script  
- Detection results are saved to the `results` directory and include:  
  - Annotated images with bounding boxes  
  - JSON files containing detection metadata  

## Output Format

Each JSON detection result contains:

- A unique detection ID  
- The original filename  
- Image width and height  
- Total number of detected objects  
- A list of detections, where each detection includes:  
  - Class name  
  - Confidence score  
  - Bounding box coordinates in the form x1, y1, x2, y2  

## Testing

- API tests can be executed using pytest  
- The tests cover:  
  - Health checks  
  - Class listing  
  - Invalid file handling  
  - JSON detection responses  
  - Annotated image outputs  

## Notes

- YOLO model weights are downloaded automatically on first run  
- Temporary upload files and generated results are excluded from version control  
- The API is structured to be extended without changing the core detection logic  