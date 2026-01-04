from pathlib import Path
import json

from src.detector import ObjectDetector


def main():
    detector = ObjectDetector()

    images_dir = Path("images")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not image_files:
        print("no .jpg or .png images found in images/")
        return

    for img_path in image_files:
        print(f"processing {img_path.name}")

        detection_result = detector.detect_objects(str(img_path))

        for det in detection_result["detections"]:
            print(f"  {det['class_name']}: {det['confidence']:.2%}")

        out_img = results_dir / f"detected_{img_path.name}"
        detector.draw_detections(str(img_path), str(out_img))
        print(f"saved image: {out_img}")

        out_json = results_dir / f"detected_{img_path.stem}.json"
        with open(out_json, "w") as f:
            json.dump(detection_result, f, indent=2)
        print(f"saved json: {out_json}\n")


if __name__ == "__main__":
    main()