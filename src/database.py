from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/objectdetection"
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()


class DetectionLog(Base):
    """Store detection results"""
    
    __tablename__ = "detection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String(50), unique=True, index=True)
    filename = Column(String(255))
    total_objects = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    processing_time = Column(Float)
    confidence_threshold = Column(Float)
    detections = Column(JSON)  # Store full detections as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<DetectionLog(id={self.id}, objects={self.total_objects})>"


class ModelMetrics(Base):
    """Store model information and metrics"""
    
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(50))
    model_version = Column(String(20))
    total_classes = Column(Integer)
    average_inference_time = Column(Float)
    total_detections = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name})>"


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()