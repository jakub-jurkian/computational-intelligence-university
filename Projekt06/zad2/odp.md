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
a) Narysuj schemat blokowy architektury i przeplywu fetch -> preprocess -> train -> evaluate -> visualize -> compare - odpowiedzi
docs/architecture_flow.mmd

Przeplyw:
configs/ -> src/cli/ -> src/experiments/ -> fetch -> preprocess -> train -> evaluate -> visualize -> compare

Role katalogow (wejscie/wyjscie):
- wejscie: data/raw (dane surowe), configs (konfiguracje), src/cli (punkt wejscia uruchomien)
- przetwarzanie: src/experiments, src/data, src/training, src/models, src/visualization
- wyjscie: data/processed (zapis danych po preprocessingu), models (checkpointy i artefakty), reports (metryki, wykresy, porownania)
- testy i walidacja jakosci: tests

b) Uruchom projekt z terminala: make setup, make pipeline, make test, make pyright, make ruff - odpowiedzi
W tym srodowisku komenda make nie byla dostepna, dlatego uruchomilem rownowazne polecenia z Makefile przez uv:

1. setup (odpowiednik make setup)
- uv sync --group dev
- uv run pre-commit install
- uv run pre-commit install --hook-type pre-push
- uv run pre-commit install --hook-type commit-msg
Wynik: poprawnie, zaleznosci sprawdzone i hooki zainstalowane.

2. pipeline (odpowiednik make pipeline)
- uv run -m src.cli.fetch_data --config configs/iris_mlp_zscore.yaml
- uv run -m src.cli.run_experiment --config configs/iris_mlp_zscore.yaml
- uv run -m src.cli.run_experiment --config configs/iris_mlp_minmax.yaml
- uv run -m src.cli.run_experiment --config configs/iris_linear_zscore.yaml
- uv run -m src.cli.run_experiment --config configs/iris_linear_minmax.yaml
- uv run -m src.cli.visualize --config configs/iris_mlp_zscore.yaml --kind raw
- uv run -m src.cli.compare_experiments --config configs/compare_experiments.yaml
Wynik: poprawnie, dane pobrane, wszystkie 4 eksperymenty wykonane, raport porownawczy zapisany.
Accuracy z logow:
- iris_mlp_zscore: 0.9667
- iris_mlp_minmax: 0.9333
- iris_linear_zscore: 0.9333
- iris_linear_minmax: 0.9333

3. test (odpowiednik make test)
- uv run pytest
Wynik: 5 passed.
Po co: sprawdza poprawne dzialanie kluczowych elementow projektu i chroni przed regresjami.

4. pyright (odpowiednik make pyright)
- uv run pyright
Wynik: 0 errors, 0 warnings.
Po co: type-checking, czyli wykrywanie niezgodnosci typow jeszcze przed uruchomieniem kodu.

5. ruff (odpowiednik make ruff)
- uv run ruff check .
Wynik: All checks passed.
Po co: linting, czyli wykrywanie problemow stylu i potencjalnych bledow jakosciowych.

c) Przeczytaj konfiguracje eksperymentow w configs/ i porownaj je - odpowiedzi
Porownane pliki:
- configs/iris_mlp_zscore.yaml
- configs/iris_mlp_minmax.yaml
- configs/iris_linear_zscore.yaml
- configs/iris_linear_minmax.yaml
- configs/compare_experiments.yaml

Glowne roznice miedzy gotowymi eksperymentami:
1. model.name
- mlp_classifier (warianty MLP)
- linear_classifier (warianty liniowe)

2. preprocessing.strategy
- zscore
- minmax

3. training
- MLP: hidden_dims [16, 16], learning_rate 0.01, epochs 150
- Linear: hidden_dims [], learning_rate 0.05, epochs 120

4. paths
- kazdy eksperyment ma osobne katalogi artefaktow, np.:
  models/iris_mlp_zscore, reports/iris_mlp_zscore, data/processed/iris_mlp_zscore
  models/iris_linear_minmax, reports/iris_linear_minmax, data/processed/iris_linear_minmax

Pola konfiguracji, ktore wplywaja na:
- model: model.name, training.hidden_dims, training.learning_rate, training.epochs, training.batch_size, training.weight_decay
- preprocessing: preprocessing.strategy oraz data.feature_columns
- sciezki zapisu artefaktow: caly blok paths (processed_dir, preprocessor_artifact, model_checkpoint, training_history_csv, training_summary_json, evaluation_json, predictions_csv, confusion_matrix_png, training_curves_png)

d) Przesledz wykonanie jednego eksperymentu od src/cli/run_experiment.py do reports/ i models/ - odpowiedzi
Przesledzony eksperyment: iris_mlp_zscore

Kolejnosc wywolan funkcji:
1. src/cli/run_experiment.py: main()
2. parse_args()
3. ProjectConfig.from_yaml(args.config)
4. read_dataset(config.paths.raw_csv)
5. ExperimentFactory.build(config.model.name, config)
6. experiment.run(frame)

W klasie bazowej (src/experiments/base.py):
7. run(frame)
8. preprocess(frame)
	- split_dataset(...)
	- fit_preprocessor(...)
	- apply_preprocessor(...) dla train/validation/test
	- zapis do data/processed/<experiment_name>/train.csv, validation.csv, test.csv
	- write_json(...) do models/<experiment_name>/preprocessor.json
9. train()
	- set_global_seed(...)
	- build_model() (implementacja konkretna w klasie eksperymentu)
	- train_model(..., checkpoint_path=config.paths.model_checkpoint)
	- zapis historii do reports/<experiment_name>/training_history.csv
	- zapis podsumowania do reports/<experiment_name>/training_summary.json
10. evaluate()
	- build_model()
	- load_checkpoint(model)
	- predict_with_model(...)
	- save_confusion_matrix(...)
	- zapis predykcji do reports/<experiment_name>/test_predictions.csv
	- zapis metryk do reports/<experiment_name>/evaluation.json
11. visualize_training()
	- save_training_curves(...)
	- zapis wykresu do reports/<experiment_name>/training_curves.png

Finalnie artefakty trafiaja do models/ i reports/ zgodnie z sekcja paths w YAML.

e) Wyjasnij klase bazowa eksperymentu src/experiments/base.py - odpowiedzi
Elementy wspolne dla wszystkich eksperymentow:
1. Init i wspolny kontekst
- __init__(config), logger, dostep do calej konfiguracji

2. Wspolny pipeline
- run(frame) uruchamia zawsze te same etapy: preprocess -> train -> evaluate -> visualize_training

3. Wspolna logika preprocessingu
- podzial danych, fit/apply preprocessora, zapis danych przetworzonych i artefaktu preprocessora

4. Wspolna logika treningu i ewaluacji
- trening przez train_model
- wczytanie checkpointu przez load_checkpoint
- predykcja i metryki (accuracy + classification_report)
- zapis raportow i wykresow

Elementy, ktore moga sie zmieniac (rozszerzanie projektu):
1. name() - unikalna nazwa typu eksperymentu
2. build_model() - konkretna architektura modelu

W praktyce oznacza to, ze aby dodac nowy typ eksperymentu, implementuje sie nowa klase dziedziczaca po BaseExperiment i definiuje glownie build_model() (oraz nazwe), a reszta pipeline pozostaje wspolna.

f) Sprawdz jak dziala walidacja konfiguracji i przygotuj 2 bledne YAML - odpowiedzi
Przeanalizowany plik: src/utils/config.py
Walidacja opiera sie o Pydantic i Literal, czyli tylko dozwolone wartosci przechodza dalej.

Dodane bledne pliki konfiguracyjne:
1. configs/invalid_bad_strategy.yaml
- blad: preprocessing.strategy = robust_plus
- dlaczego odrzucany: dozwolone sa tylko zscore, minmax, robust
- potwierdzony blad runtime: Input should be 'zscore', 'minmax' or 'robust'

2. configs/invalid_missing_model_name.yaml
- blad: model: {} (brak pola model.name)
- dlaczego odrzucany: pole model.name jest wymagane przez ModelConfig
- potwierdzony blad runtime: model.name Field required

Dodatkowo dopisalem testy walidacji:
- tests/test_config.py::test_project_config_rejects_unknown_preprocessing_strategy
- tests/test_config.py::test_project_config_rejects_missing_model_name

g) Dodaj nowy wariant eksperymentu bez zmiany kodu i porownaj wyniki - odpowiedzi
Dodany nowy wariant konfiguracyjny:
- configs/iris_mlp_zscore_tuned.yaml

Zmiany wzgledem bazowego iris_mlp_zscore:
- hidden_dims: [32, 16] zamiast [16, 16]
- learning_rate: 0.005 zamiast 0.01
- epochs: 220 zamiast 150

Uruchomienie:
- uv run -m src.cli.run_experiment --config configs/iris_mlp_zscore_tuned.yaml

Wynik (z logu):
- iris_mlp_zscore_tuned accuracy: 0.9333
- bazowy iris_mlp_zscore accuracy: 0.9667

Wniosek: ten wariant konfiguracyjny okazal sie slabszy od bazowego, mimo wiekszej liczby epok i wiekszej sieci.

h) Dodaj nowy preprocessing robust + walidacja + testy - odpowiedzi
Wprowadzone zmiany:
1. src/data/preprocessing.py
- dodane fit_robust_scaler(...)
- dodane apply_robust_scaler(...)
- rozszerzone fit_preprocessor(...) i apply_preprocessor(...) o strategy=robust

2. src/utils/config.py
- PreprocessingConfig.strategy rozszerzone o robust

3. src/experiments/base.py
- predict_one(...) rozszerzone o obsluge strategii robust (median/iqr)

4. testy
- tests/test_preprocessing.py::test_robust_scaler_centers_median_to_zero

Wynik walidacji jakosci po zmianach:
- pytest: 9 passed
- pyright: 0 errors
- ruff: All checks passed

i) Dodaj nowy model albo nowy typ eksperymentu i uruchamianie przez konfiguracje - odpowiedzi
Zrealizowane jako nowy typ eksperymentu i nowy model:

1. nowy model
- src/models/mlp_dropout.py
- klasa IrisMLPDropout (MLP z warstwa Dropout)

2. nowy eksperyment
- src/experiments/mlp_dropout_experiment.py
- klasa MLPDropoutClassifierExperiment z rejestracja @register_experiment("mlp_dropout_classifier")

3. rejestr i walidacja
- src/experiments/__init__.py: import i ekspozycja nowego eksperymentu
- src/utils/config.py: model.name dopuszcza mlp_dropout_classifier

4. nowa konfiguracja eksperymentu
- configs/iris_mlp_dropout_robust.yaml
- model.name: mlp_dropout_classifier
- preprocessing.strategy: robust

Uruchomienie:
- uv run -m src.cli.run_experiment --config configs/iris_mlp_dropout_robust.yaml

Wynik:
- iris_mlp_dropout_robust accuracy: 0.9333

j) Rozszerz raportowanie wynikow o nowa metryke albo wykres - odpowiedzi
Rozszerzylem raportowanie o nowa metryke porownawcza macro_f1.

Zmiany:
1. src/cli/compare_experiments.py
- odczyt macro_f1 z evaluation[classification_report][macro avg][f1-score]
- dodanie kolumny macro_f1 do raportu CSV/JSON
- wykres porownawczy rozszerzony do dwoch slupkow na eksperyment: test_accuracy i macro_f1

2. configs/compare_experiments.yaml
- dodane nowe eksperymenty do porownania:
	configs/iris_mlp_zscore_tuned.yaml
	configs/iris_mlp_dropout_robust.yaml

Wynik zapisow:
- reports/comparisons/experiment_comparison.csv (ma teraz kolumny test_accuracy i macro_f1)
- reports/comparisons/experiment_comparison.json
- reports/comparisons/experiment_comparison.png (wykres grupowany)

k) Dopisz test bledu dla niepoprawnej nazwy eksperymentu albo blednej konfiguracji - odpowiedzi
Dodane testy bledow:
1. tests/test_experiments.py::test_experiment_factory_rejects_unknown_experiment_name
- sprawdza, ze ExperimentFactory.build("unknown_experiment", ...) rzuca ValueError

2. tests/test_config.py
- test_project_config_rejects_unknown_preprocessing_strategy
- test_project_config_rejects_missing_model_name

Wynik:
- testy przechodza (pytest: 9 passed)

l) Dodaj wlasne polecenie CLI - odpowiedzi
Dodane nowe polecenie CLI:
- src/cli/list_configs.py

Co robi:
- wypisuje dostepne konfiguracje eksperymentow z katalogu configs
- pomija compare_experiments.yaml oraz celowo bledne pliki invalid_*.yaml

Uruchomienie:
- uv run -m src.cli.list_configs --configs-dir configs

Wynik:
- wypisane 6 poprawnych konfiguracji eksperymentow

Dodatkowo dodalem skrot w Makefile:
- make list-configs

m) Podmien dataset na diagnosis.csv i opisz co trzeba zmienic, a co zostalo bez zmian - odpowiedzi

Co zmienilem:
1. dane i konfiguracja
- skopiowalem plik z io/lab3 do projektu: data/raw/diagnosis.csv
- dodalem metadane: data/raw/diagnosis_metadata.json
- dodalem nowa konfiguracje: configs/diagnosis_linear_robust.yaml
	- target_column: diagnosis
	- feature_columns: [param1, param2, param3]
	- class_names: ["0", "1"]
	- preprocessing: robust
	- model: linear_classifier

2. uogolnienie kodu pod dowolna kolumne target
- src/training/trainer.py
	- dataframe_to_loader(...) przyjmuje teraz target_column
	- etykiety mapowane sa z frame[target_column].astype(str)
- src/training/trainer.py + src/experiments/base.py
	- TrainingBundle ma pole target_column
	- BaseExperiment.train przekazuje config.data.target_column

3. wygodne uruchomienie
- dodany target Makefile: run-diagnosis-linear-robust

Co pozostalo bez zmian (architektura dziala dalej tak samo):
1. warstwy projektu
- src/cli -> src/experiments -> src/training/src/data -> reports/models

2. pipeline
- ten sam przebieg: preprocess -> train -> evaluate -> visualize_training

3. mechanizmy jakosci
- testy, type-checking i linting bez zmian uruchamiania (pytest, pyright, ruff)

Uruchomienie i wynik:
- uv run -m src.cli.run_experiment --config configs/diagnosis_linear_robust.yaml
- accuracy: 0.7857
- macro_f1: 0.7355
- raport: reports/diagnosis_linear_robust/evaluation.json

Weryfikacja po zmianach:
- pytest: 9 passed
- pyright: 0 errors
- ruff: All checks passed
