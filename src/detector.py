from ultralytics import YOLO
import cv2
from typing import Dict, List


class ObjectDetector:
    """runs YOLOv8 object detection on images and returns structured results"""

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

    def get_class_names(self) -> List[str]:
        return list(self.model.names.values())

    def get_class_count(self) -> int:
        return len(self.model.names)