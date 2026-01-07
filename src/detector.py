from ultralytics import YOLO
import cv2
from typing import Dict, List
import uuid
import time
from datetime import datetime
from pathlib import Path


class ObjectDetector:
    """runs YOLOv8 object detection on images and uploaded files"""

    def __init__(self, model_path: str = "yolov8n.pt"):
        print(f"loading model: {model_path}")
        self.model = YOLO(model_path)
        print("model ready")

    def detect_objects(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        results = self.model(image_path, conf=conf_threshold)
        result = results[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),
                }
            )

        return {
            "image_path": image_path,
            "total_objects": len(detections),
            "image_shape": result.orig_shape,
            "detections": detections,
        }

    def draw_detections(self, image_path: str, output_path: str, conf_threshold: float = 0.25) -> str:
        results = self.model(image_path, conf=conf_threshold)
        annotated = results[0].plot()
        cv2.imwrite(output_path, annotated)
        return output_path

    def detect_from_file(self, file_bytes: bytes, filename: str, conf_threshold: float = 0.25) -> Dict:
        detection_id = f"det_{uuid.uuid4().hex[:8]}"

        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{detection_id}_{filename}"

        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            start_time = time.time()
            results = self.model(str(temp_path), conf=conf_threshold)
            result = results[0]
            processing_time = time.time() - start_time

            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [float(x) for x in box.xyxy[0].tolist()],
                    }
                )

            img_height, img_width = result.orig_shape

            return {
                "detection_id": detection_id,
                "filename": filename,
                "total_objects": len(detections),
                "detections": detections,
                "image_width": int(img_width),
                "image_height": int(img_height),
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now(),
            }

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def detect_and_annotate(self, file_bytes: bytes, filename: str, conf_threshold: float = 0.25) -> tuple:
        detection_id = f"det_{uuid.uuid4().hex[:8]}"

        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{detection_id}_{filename}"

        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            start_time = time.time()
            results = self.model(str(temp_path), conf=conf_threshold)
            result = results[0]
            processing_time = time.time() - start_time

            annotated_img = result.plot()
            success, buffer = cv2.imencode(".jpg", annotated_img)
            if not success:
                raise RuntimeError("failed to encode annotated image")

            annotated_bytes = buffer.tobytes()

            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0]),
                        "bbox": [float(x) for x in box.xyxy[0].tolist()],
                    }
                )

            img_height, img_width = result.orig_shape

            detection_data = {
                "detection_id": detection_id,
                "filename": filename,
                "total_objects": len(detections),
                "detections": detections,
                "image_width": int(img_width),
                "image_height": int(img_height),
                "processing_time": round(processing_time, 3),
                "timestamp": datetime.now(),
            }

            return annotated_bytes, detection_data

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def get_class_names(self) -> List[str]:
        return list(self.model.names.values())

    def get_class_count(self) -> int:
        return len(self.model.names)