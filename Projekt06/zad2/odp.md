a) Zapoznanie z materialami PyTorch - odpowiedzi
Zapoznalem sie z podejsciem transfer learning w klasyfikacji obrazow na podstawie:
 - oficjalnego tutoriala PyTorch (transfer learning),
 - przykladow implementacji klasyfikacji cats vs dogs.

Najwazniejsze wnioski praktyczne:
1. Dla malego/umiarkowanego zbioru danych transfer learning zwykle daje lepsza jakosc niz trenowanie od zera.
2. Kluczowe sa poprawne transformacje i normalizacja obrazow.
3. Najwygodniej porownywac konfiguracje przez osobne pliki YAML i automatyczny zapis artefaktow (wykresy, metryki, checkpointy).

b) Zaladowanie i preprocessing danych - odpowiedzi
Zbior danych:
 - dogs-cats-mini.zip (12 500 kotow + 12 500 psow, lacznie 25 000 obrazow).

Wykonane kroki:
1. Wczytanie obrazow z katalogu danych.
2. Wyznaczenie etykiety klasy na podstawie nazwy pliku:
 - cat.* -> etykieta 0,
 - dog.* -> etykieta 1.
3. Podzial stratified na zbiory:
 - treningowy,
 - walidacyjny,
 - testowy.
4. Transformacje PyTorch:
 - Resize,
 - ToTensor,
 - Normalize (statystyki ImageNet),
 - opcjonalnie augmentacja (flip/rotation/color jitter) zaleznie od konfiguracji.

Implementacja:
 - src/cats_dogs/data.py

c) Budowa i trening CNN - odpowiedzi
Zaimplementowane modele:
1. CNN od zera (SimpleCNN): Conv2d + BatchNorm + ReLU/LeakyReLU + MaxPool + FC + Dropout.
2. Model transfer learning (ResNet18): wariant pre-trained i fine-tuning.

Uruchamiane konfiguracje:
 - configs/cats_dogs_cnn_sgd.yaml
 - configs/cats_dogs_cnn_adam.yaml
 - configs/cats_dogs_cnn_deeper.yaml
 - configs/cats_dogs_transfer_resnet18.yaml

Implementacja:
 - src/cats_dogs/models.py
 - src/cats_dogs/trainer.py
 - src/cli/run_cats_dogs.py

d) Walidacja modelu i analiza bledow - odpowiedzi
Dla kazdego eksperymentu zapisane zostaly:
1. Krzywe uczenia (train/val loss i train/val accuracy): learning_curves.png
2. Macierz pomylek: confusion_matrix.png
3. Predykcje testowe: test_predictions.csv
4. Przyklady blednych klasyfikacji: misclassified_examples.png

Wyniki (test_acc):
1. cnn_sgd_baseline: 0.6744
2. cnn_adam_augmented: 0.6776
3. cnn_deeper_leakyrelu: 0.7304
4. transfer_resnet18: 0.9308

Pytania z zadania:
1. Czy wystepuja koty podobne do psow i psy podobne do kotow?
 - Tak, co widac po niezerowej liczbie pomylek i konkretnych bledach w test_predictions.csv.
2. Ile takich przypadkow mozna wskazac?
 - Dla najlepszego modelu transfer_resnet18: 173 bledne klasyfikacje na 2500 probek testowych.
3. Czy mozna pokazac konkretne obrazy z bledami?
 - Tak, sa zapisane w reports/transfer_resnet18/misclassified_examples.png.
 - Konkretne przyklady z predykcji: cat.2021.jpg -> dog, cat.7486.jpg -> dog, dog.11910.jpg -> cat.

e) Kilka konfiguracji i wybor najlepszej - odpowiedzi
Porownalem kilka wariantow zmieniajac:
1. optymalizator (SGD vs Adam),
2. architekture (glebsza siec, liczba filtrow),
3. funkcje aktywacji (ReLU vs LeakyReLU),
4. dropout,
5. parametry augmentacji,
6. podejscie transfer learning.

Najlepsza konfiguracja:
1. transfer_resnet18
2. test_acc: 0.9308
3. confusion_matrix: [[1150, 100], [73, 1177]]

Artefakty porownawcze:
 - reports/comparisons/cats_dogs/comparison.csv
 - reports/comparisons/cats_dogs/comparison_test_acc.png
 - reports/comparisons/cats_dogs/comparison_summary.json

Wniosek:
Transfer learning dal wyraznie lepszy wynik niz CNN trenowane od zera.

f) Transfer learning vs trening od zera - odpowiedzi
Tak, model pre-trained mozna skutecznie dotrenowac i uzyskac lepsze rozpoznawanie psow i kotow.

Porownanie:
1. Najlepszy model od zera (cnn_deeper_leakyrelu): 0.7304
2. Transfer learning (transfer_resnet18): 0.9308
3. Rznica: +0.2004 na korzysc transfer learningu

Dodatkowo:
1. Trening zapisuje checkpoint best.pt dla kazdej konfiguracji.
2. Mozna wznowic/odtworzyc najlepszy model z pliku:
 - models/transfer_resnet18/best.pt

Podsumowanie koncowe:
Zadanie 2 (a-f) zostalo wykonane dla projektu cats_dogs. Przygotowano pipeline danych, modele CNN i transfer learning, walidacje, analize bledow, porownanie eksperymentow i artefakty raportowe.
