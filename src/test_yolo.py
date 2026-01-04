from ultralytics import YOLO

print("loading YOLOv8 model...")
model = YOLO("yolov8n.pt")

print("model loaded successfully")
print(f"model classes: {len(model.names)}")
print(f"sample classes: {list(model.names.values())[:10]}")
