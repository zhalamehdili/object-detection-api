from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import io

from src.detector import ObjectDetector
from src.schemas import DetectionResponse, HealthResponse

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
):
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="only JPEG and PNG supported")

    if not detector:
        raise HTTPException(status_code=503, detail="model not loaded")

    contents = await file.read()
    try:
        result = detector.detect_from_file(contents, file.filename, conf_threshold=confidence)
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
        annotated_bytes, meta = detector.detect_and_annotate(contents, file.filename, conf_threshold=confidence)
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