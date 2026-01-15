from src.database import create_tables, SessionLocal, ModelMetrics
from datetime import datetime


def init_database():
    """Initialize database with tables and default data"""

    print("Creating database tables...")
    create_tables()

    db = SessionLocal()

    existing = db.query(ModelMetrics).filter_by(model_name="YOLOv8n").first()

    if not existing:
        initial_metrics = ModelMetrics(
            model_name="YOLOv8n",
            model_version="8.0",
            total_classes=80,
            average_inference_time=0.095,
            total_detections=0,
            last_updated=datetime.utcnow(),
            notes="YOLOv8 Nano model for real-time object detection",
        )
        db.add(initial_metrics)
        db.commit()
        print("Initial model metrics added")
    else:
        print("Model metrics already exist")

    db.close()
    print("Database initialization complete")


if __name__ == "__main__":
    init_database()