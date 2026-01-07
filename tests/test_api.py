from fastapi.testclient import TestClient
from pathlib import Path
from src.api import app
import src.api as api_module

client = TestClient(app)


def setup_module():
    # load the model once for tests
    if api_module.detector is None:
        api_module.detector = api_module.ObjectDetector()


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["model_classes"] == 80


def test_get_classes():
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 80
    assert "person" in data["classes"]
    assert "car" in data["classes"]


def test_invalid_file_type():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_detect_objects_if_image_exists():
    # uses any local image if present (keeps tests flexible)
    images_dir = Path("images")
    candidates = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not candidates:
        return

    img_path = candidates[0]
    with open(img_path, "rb") as f:
        response = client.post(
            "/detect",
            files={"file": (img_path.name, f, "image/jpeg")},
            params={"confidence": 0.25},
        )

    assert response.status_code == 200
    data = response.json()
    assert "detection_id" in data
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_detect_annotated_if_image_exists():
    images_dir = Path("images")
    candidates = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not candidates:
        return

    img_path = candidates[0]
    with open(img_path, "rb") as f:
        response = client.post(
            "/detect/annotated",
            files={"file": (img_path.name, f, "image/jpeg")},
            params={"confidence": 0.25},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")