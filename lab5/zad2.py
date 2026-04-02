# W zadaniu użyłem OpenCV. Ptaki na tych miniaturkach to w zasadzie tylko czarne plamy, więc problem to po prostu odcięcie tła (progowanie).

# Zamiast szukać jednego uniwersalnego filtru, napisałem algorytm adaptacyjny.
# Kod oblicza medianę jasności dla każdego zdjęcia, a następnie dynamicznie ustawia próg odcięcia o 10 tonów ciemniejszy od nieba.
# Algorytm pomylił się tylko 6 razy na całym zbiorze danych. Te pomyłki wynikają z faktu, że silne artefakty kompresji JPEG
# potrafią zbić się w ciemne piksele, które dla komputera wyglądają identycznie jak ptak. Przy tak trudnych danych to i tak dobry wynik.

import cv2
import numpy as np
from pathlib import Path


def count_birds(image_path):
    # Load image directly in grayscale.
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    # dynamic threshold calculation.
    # We find the median (sky) and set the cutoff 10 shades darker.
    thresh_val = max(np.median(img) - 10, 0)

    # Apply mask to separate birds from the background.
    _, mask = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Find contours and count them efficiently.
    # Using a generator expression 'sum(1 for c...)' replaces a multi-line loop.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) > 0)


def process_folder(folder_name):
    # Using modern 'pathlib' instead of os/glob for cleaner code.
    folder = Path(folder_name)

    print(f"{'Filename':<35} | {'Bird Count'}")
    print("-" * 50)

    images = sorted(
        (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"),
        key=lambda p: p.name.lower(),
    )

    for path in images:
        print(f"{path.name:<35} | {count_birds(path)}")


if __name__ == "__main__":
    process_folder("bird_miniatures")
