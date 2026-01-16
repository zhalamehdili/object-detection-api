# Real-Time Object Detection API

This project implements a real-time object detection system using YOLOv8. It identifies and localizes common objects in images and exposes the detection pipeline through a REST API built with FastAPI. The system returns structured JSON outputs containing class labels, confidence scores and bounding boxes and persists detection metadata in a PostgreSQL database.

The project focuses on building a clean, deployable computer vision service with proper API design, containerization and database integration.

## Features

- Object detection using YOLOv8  
- Support for 80 object classes  
- Bounding box localization with confidence scores  
- Image upload via REST API  
- JSON-based detection responses  
- Persistent detection history and statistics  
- Dockerized setup for local and cloud execution  
- PostgreSQL-backed data storage  

## Tech Stack

- Python 3.10  
- Ultralytics YOLOv8  
- FastAPI  
- SQLAlchemy  
- PostgreSQL  
- OpenCV  
- NumPy  
- Docker  
- Render  

## Project Structure

object-detection-api  
├── src  
│   ├── api.py – FastAPI application and endpoints  
│   ├── detector.py – YOLOv8 detection logic  
│   ├── database.py – SQLAlchemy models and session handling  
│   ├── init_db.py – Database initialization  
│   ├── schemas.py – Pydantic response models  
│   └── __init__.py  
├── screenshots – API and deployment screenshots  
├── Dockerfile  
├── docker-compose.yml  
├── railway.json  
├── requirements.txt  
├── .env.example  
├── .gitignore  
└── README.md  

## Local Setup

Create and activate a Python 3.10 virtual environment, install dependencies from requirements.txt and build and run the services using Docker Compose. After startup, the API is available locally at http://localhost:8000/docs.

## Deployment

The application is deployed as a Docker-based web service with a managed PostgreSQL database. Environment variables are used for database configuration and runtime settings. The service is hosted on Render and automatically rebuilds on updates to the main branch.

## API Endpoints

Health check endpoint returns service status and model availability at /health.

Object detection endpoint accepts an uploaded image and returns detection results at /detect.

Statistics endpoint returns aggregated detection metrics at /stats.

Detection history endpoint returns recent detection records at /history.

Interactive API documentation is available at /docs.

## Example Remote Endpoints

https://object-detection-api-rmtj.onrender.com/health  
https://object-detection-api-rmtj.onrender.com/detect  
https://object-detection-api-rmtj.onrender.com/stats  
https://object-detection-api-rmtj.onrender.com/history  
https://object-detection-api-rmtj.onrender.com/docs  

## Output Format

Each detection response includes a unique detection ID, the original filename, image width and height, total number of detected objects, processing time, timestamp and a list of detections. Each detection entry contains the class name, confidence score and bounding box coordinates in x1, y1, x2, y2 format.

## Notes

YOLO model weights are downloaded automatically on first startup. Detection data is persisted in PostgreSQL. Temporary files and generated outputs are excluded from version control. Free-tier cloud instances may spin down during inactivity, which can cause initial request delays.