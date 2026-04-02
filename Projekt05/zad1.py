# zad. 1
# W pierwszym zadaniu wykorzystałem model YOLO w wersji 8 Nano.
# Wewnątrz opiera się on na architekturze CSPDarknet, która służy do wyciągania cech z obrazu.
# Został on wytrenowany na popularnym zbiorze COCO, co oznacza, że potrafi rozpoznać 80 różnych klas obiektów codziennego użytku.

# Program analizuje zdjęcie i wideo, zapisując dane do JSON-a. Eksperymentowałem z progami confidence.
# Z moich obserwacji wynika, że przy minimalnym progu, np. 0.1, model staje się nadwrażliwy – wykrywa wszystko, ale generuje mnóstwo błędów typu False Positive,
# Z kolei wysoki próg, np. 0.7, sprawia, że model omija obiekty zamazane lub mniejsze.
# Optymalnym balansem okazały się wartości w przedziale 0.3 do 0.5.

import json
from collections import Counter
from ultralytics import YOLO

# Initialize model and define parameters
model = YOLO("yolov8n.pt")
THRESHOLDS = [0.1, 0.3, 0.5, 0.7]


def parse_boxes(boxes):
    """
    Quickly extracts data from YOLO boxes into a dictionary format.
    """
    return [
        {
            "class_id": int(b.cls[0]),
            "label": model.names[int(b.cls[0])],
            "confidence": round(float(b.conf[0]), 4),
            "bbox": b.xyxy[0].tolist(), # współrzędne prostokąta wykrytego obiektu
        }
        for b in boxes
    ]


def run_yolo_tasks():
    for conf in THRESHOLDS:
        # TASK 1A & 1B & 1C: IMAGE PROCESSING
        # Predict on image and save JSON
        res_img = model.predict(
            "office_yolo.png", conf=conf, save=True, name=f"img_conf_{conf}"
        )

        with open(f"image_processing/detections_img_{conf}.json", "w") as f:
            json.dump(parse_boxes(res_img[0].boxes), f, indent=4)

        # TASK 1D & 1E: VIDEO PROCESSING
        # Predict on video using stream=True
        res_vid = model.predict(
            "office_yolo.mp4",
            conf=conf,
            save=True,
            stream=True,
            name=f"vid_conf_{conf}",
        )

        frames_data = []
        # Using Counter for automatic stat tracking
        stats = Counter()

        for i, frame_result in enumerate(res_vid):
            dets = parse_boxes(frame_result.boxes)
            frames_data.append({"frame": i, "detections": dets})

            # Generator expression updates stats instantly
            stats.update(d["label"] for d in dets)

        # Save video JSON and print stats
        with open(f"video_processing/detections_vid_{conf}.json", "w") as f:
            json.dump({"summary": dict(stats), "frames": frames_data}, f, indent=4)

        print(f"\n[Stats for confidence {conf}]: {dict(stats)}")


if __name__ == "__main__":
    run_yolo_tasks()
