from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class BoundingBox(BaseModel):
    x1: float = Field(..., description="top-left x")
    y1: float = Field(..., description="top-left y")
    x2: float = Field(..., description="bottom-right x")
    y2: float = Field(..., description="bottom-right y")


class Detection(BaseModel):
    class_id: int = Field(..., description="class id")
    class_name: str = Field(..., description="class name")
    confidence: float = Field(..., ge=0, le=1, description="confidence score")
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")


class DetectionResponse(BaseModel):
    detection_id: str
    filename: str
    total_objects: int
    detections: List[Detection]
    image_width: int
    image_height: int
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_classes: int
    timestamp: datetime