# Zadanie 2: CNN dla psow i kotow

Implementacja jest osadzona w strukturze ml-project-template.

## Co zostalo dodane

- Modul eksperymentu obrazowego:
  - `src/cats_dogs/config.py`
  - `src/cats_dogs/data.py`
  - `src/cats_dogs/models.py`
  - `src/cats_dogs/trainer.py`
- CLI:
  - `src/cli/run_cats_dogs.py`
  - `src/cli/compare_cats_dogs.py`
- Konfiguracje eksperymentow:
  - `configs/cats_dogs_cnn_sgd.yaml`
  - `configs/cats_dogs_cnn_adam.yaml`
  - `configs/cats_dogs_cnn_deeper.yaml`
  - `configs/cats_dogs_transfer_resnet18.yaml`
- Targety `make` do uruchamiania i porownania.

## Jak uruchomic

1. Instalacja zaleznosci:

```bash
uv sync --group dev
```

2. Pojedynczy eksperyment (CNN od zera):

```bash
make run-cats-cnn-sgd
```

lub:

```bash
make run-cats-cnn-adam
```

3. Transfer learning (ResNet18):

```bash
make run-cats-transfer
```

4. Wszystkie eksperymenty i porownanie:

```bash
make run-cats-all
make compare-cats
```

## Artefakty wynikowe

Dla kazdego eksperymentu zapisuja sie:

- `reports/<experiment>/training_history.csv`
- `reports/<experiment>/learning_curves.png`
- `reports/<experiment>/confusion_matrix.png`
- `reports/<experiment>/misclassified_examples.png`
- `reports/<experiment>/test_predictions.csv`
- `reports/<experiment>/summary.json`
- `models/<experiment>/best.pt`

## Pokrycie podpunktow zadania

- b) Wczytanie i podzial danych + transformacje: `src/cats_dogs/data.py`
- c) CNN od zera: `src/cats_dogs/models.py` (`SimpleCNN`)
- d) Walidacja, krzywe, bledne klasyfikacje: `src/cats_dogs/trainer.py`
- e) Kilka konfiguracji: pliki `configs/cats_dogs_*.yaml`
- f) Transfer learning: `configs/cats_dogs_transfer_resnet18.yaml` + `TransferResNet18`
