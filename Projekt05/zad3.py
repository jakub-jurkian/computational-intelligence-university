# zad. 3
# Sprawdzam czy YOLO umie liczyć te latające obiekty - radzi sobie z tym bardzo słabo, ale wnioski są takie:

# 1: miniaturki były stanowczo za małe dla YOLO. Powiększenie obrazu (preprocessing) było konieczne, ponieważ tak małe,
# kilkupikselowe kropki po prostu zniknęłyby w warstwach zmniejszających (tzw. pooling layers) sieci konwolucyjnej.
# Dlatego w kodzie powiększam obraz trzykrotnie.

# 2: zgodnie z sugestią z instrukcji, ustawiłem zliczanie trzech klas: ptaka, samolotu i latawca. YOLO opiera się na teksturach,
# a te kropki nie mają żadnych cech szczególnych. Model po prostu nie potrafi ich odróżnić, więc zliczanie wszystkich trzech klas
# pomaga uchwycić więcej plam.

# 3: próg confidence dla YOLO musiałem ustawić bardzo nisko, na poziomie 0.1. Ponieważ obiektom brakuje jakichkolwiek detali,
# takich jak dziób czy skrzydła, sieć nigdy nie będzie pewna swojej predykcji.

# Podsumowując: eksperyment udowadnia, że do detekcji jednolitych plam pozbawionych tekstur,
# klasyczna matematyka na pikselach (OpenCV) jest skuteczniejsza niż skomplikowane sieci neuronowe.



import cv2
from pathlib import Path
from ultralytics import YOLO

# Model setup and target classes
model = YOLO("yolov8n.pt")

# Using a Python 'set' {}.
# 4: airplane, 14: bird, 33: kite
TARGET_CLASSES = {4, 14, 33}


def count_birds_yolo(path, scale=3, conf=0.1):
    # Load image
    img = cv2.imread(str(path))

    # Preprocessing (Upscaling)
    # CNN pooling layers destroy 2-pixel objects.
    h, w = img.shape[:2]
    resized = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # YOLO Inference with extremely low confidence.
    # Birds lack texture here, so the model will never be 50% sure.
    # We take the first element [0] since we only predict one image at a time.
    res = model.predict(resized, conf=conf, verbose=False)[0]

    # We sum '1' for every bounding box that belongs to our target classes.
    return sum(1 for box in res.boxes if int(box.cls[0]) in TARGET_CLASSES)


def process_yolo_folder(folder_name):
    # File management using pathlib.
    folder = Path(folder_name)

    print(f"{'Filename':<35} | {'YOLO Count'}")
    print("-" * 50)

    # Iterating only valid images
    images = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"),
        key=lambda p: p.name.lower(),
    )

    for path in images:
        print(f"{path.name:<35} | {count_birds_yolo(path)}")


if __name__ == "__main__":
    process_yolo_folder("bird_miniatures")
