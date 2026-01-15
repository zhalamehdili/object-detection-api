from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
import logging
import io
from src.init_db import init_database

from src.detector import ObjectDetector
from src.schemas import DetectionResponse, HealthResponse
from src.database import get_db, DetectionLog, ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Object Detection API",
    description="object detection using YOLOv8",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: ObjectDetector | None = None


@app.on_event("startup")
async def startup_event():
    global detector
    logger.info("initializing database...")
    init_database()
    logger.info("database initialized")

    logger.info("loading YOLO model...")
    detector = ObjectDetector()
    logger.info("model loaded")


@app.get("/")
async def root():
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classes": "/classes",
            "detect": "/detect",
            "detect_annotated": "/detect/annotated",
            "stats": "/stats",
            "history": "/history",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if detector else "unhealthy",
        model_loaded=detector is not None,
        model_classes=detector.get_class_count() if detector else 0,
        timestamp=datetime.now(),
    )


@app.get("/classes")
async def get_classes():
    if not detector:
        raise HTTPException(status_code=503, detail="model not loaded")

    classes = detector.get_class_names()
    return {"total_classes": len(classes), "classes": classes}


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="only JPEG and PNG supported")

    if not detector:
        raise HTTPException(status_code=503, detail="model not loaded")

    try:
        contents = await file.read()

        result = detector.detect_from_file(
            file_bytes=contents,
            filename=file.filename,
            conf_threshold=confidence,
        )

        try:
            detection_log = DetectionLog(
                detection_id=result["detection_id"],
                filename=result["filename"],
                total_objects=result["total_objects"],
                image_width=result["image_width"],
                image_height=result["image_height"],
                processing_time=result["processing_time"],
                confidence_threshold=confidence,
                detections=result["detections"],
                created_at=result["timestamp"],
            )
            db.add(detection_log)
            db.commit()

            metrics = db.query(ModelMetrics).filter_by(model_name="YOLOv8n").first()
            if metrics:
                metrics.total_detections += 1
                metrics.last_updated = datetime.now()
                db.commit()

            logger.info(f"logged detection {result['detection_id']} to database")

        except Exception as e:
            logger.error(f"database logging failed: {str(e)}")
            db.rollback()

        return DetectionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detection failed: {str(e)}")


@app.post("/detect/annotated")
async def detect_objects_annotated(
    file: UploadFile = File(...),
    confidence: float = Query(0.25, ge=0.0, le=1.0),
):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="only JPEG and PNG supported")

    if not detector:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await file.read()
    try:
        annotated_bytes, meta = detector.detect_and_annotate(
            file_bytes=contents,
            filename=file.filename,
            conf_threshold=confidence,
        )
        return StreamingResponse(
            io.BytesIO(annotated_bytes),
            media_type="image/jpeg",
            headers={
                "X-Total-Objects": str(meta["total_objects"]),
                "X-Processing-Time": str(meta["processing_time"]),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detection failed: {str(e)}")


@app.get("/stats")
async def get_statistics(db: Session = Depends(get_db)):
    total_detections = db.query(DetectionLog).count()

    if total_detections == 0:
        return {
            "total_detections": 0,
            "average_processing_time": 0,
            "total_objects_detected": 0,
            "most_common_objects": [],
        }

    times = db.query(DetectionLog).with_entities(DetectionLog.processing_time).all()
    avg_processing = sum([t[0] for t in times]) / len(times)

    objects = db.query(DetectionLog).with_entities(DetectionLog.total_objects).all()
    total_obj_count = sum([o[0] for o in objects])

    metrics = db.query(ModelMetrics).filter_by(model_name="YOLOv8n").first()

    return {
        "total_detections": total_detections,
        "average_processing_time": round(avg_processing, 3),
        "total_objects_detected": total_obj_count,
        "model_info": {
            "name": metrics.model_name if metrics else "YOLOv8n",
            "total_classes": metrics.total_classes if metrics else 80,
            "version": metrics.model_version if metrics else "8.0",
        },
    }


@app.get("/history")
async def get_detection_history(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    detections = (
        db.query(DetectionLog).order_by(DetectionLog.created_at.desc()).limit(limit).all()
    )

    return {
        "total_returned": len(detections),
        "detections": [
            {
                "detection_id": d.detection_id,
                "filename": d.filename,
                "total_objects": d.total_objects,
                "processing_time": d.processing_time,
                "created_at": d.created_at.isoformat(),
            }
            for d in detections
        ],
    }


@app.get("/detection/{detection_id}")
async def get_detection_details(
    detection_id: str,
    db: Session = Depends(get_db),
):
    detection = db.query(DetectionLog).filter_by(detection_id=detection_id).first()

    if not detection:
        raise HTTPException(status_code=404, detail="detection not found")

    return {
        "detection_id": detection.detection_id,
        "filename": detection.filename,
        "total_objects": detection.total_objects,
        "image_width": detection.image_width,
        "image_height": detection.image_height,
        "processing_time": detection.processing_time,
        "confidence_threshold": detection.confidence_threshold,
        "detections": detection.detections,
        "created_at": detection.created_at.isoformat(),
    }