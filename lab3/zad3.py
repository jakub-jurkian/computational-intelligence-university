# uczenie maszynowe
# sprawdzamy, który algorytm najlepiej poradzi sobie z klasyfikacją irysów.
# porównanie metody drzewa, przez szukanie sąsiadów (k-NN)
# użycie statystyki (Naive Bayes) i sieci neuronowych (MLP).
# skalowanie danych (StandardScaler) i użycie potoków do automatyzacji procesu
# przygotowania danych dla modeli wrażliwych na rozpiętość liczb.
# bez normalizacji algorytmy oparte na dystansie (k-NN, sieci neuronowe) mogłyby działać błędnie.

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # Statystyczny klasyfikator Bayesa
from sklearn.neighbors import KNeighborsClassifier # Algorytm k-Najbliższych Sąsiadów
from sklearn.neural_network import MLPClassifier # Sztuczna sieć neuronowa
from sklearn.pipeline import Pipeline # łączy skaler i model
from sklearn.preprocessing import StandardScaler # Normalizacja (skalowanie) danych
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris_big.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

train_inputs = train_set[:, 0:4].astype(float)
train_classes = train_set[:, 4]

test_inputs = test_set[:, 0:4].astype(float)
test_classes = test_set[:, 4]

labels = ["setosa", "versicolor", "virginica"]

classifiers = {
    "DecTree": DecisionTreeClassifier(random_state=300806),
    # Zanim podasz dane do algorytmu k-NN, najpierw przepuść je przez StandardScaler
    "3NN": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]
    ),
    "5NN": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]
    ),
    "11NN": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=11)),
        ]
    ),
    "Naive Bayes": GaussianNB(),
    "MLP": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(10,),
                    max_iter=2000,
                    random_state=300806,
                ),
            ),
        ]
    ),
}

results = []

# trenuj -> przewiduj -> oceń
for name, classifier in classifiers.items():
    classifier.fit(train_inputs, train_classes)
    predictions = classifier.predict(test_inputs)
    accuracy = classifier.score(test_inputs, test_classes) * 100
    cm = confusion_matrix(test_classes, predictions, labels=labels)

    results.append((name, accuracy))

    print(f"Klasyfikator: {name}")
    print(f"Dokładność: {accuracy:.2f}%")
    print("Macierz błędów:")
    print(cm)
    print()

# tworzenie tabeli z wynikami modeli
results_df = pd.DataFrame(results, columns=["Klasyfikator", "Dokładność [%]"])
results_df = results_df.sort_values(by="Dokładność [%]", ascending=False).reset_index(drop=True)

print("Porównanie klasyfikatorów:")
print(results_df.to_string(index=False))
print()

best_name = results_df.loc[0, "Klasyfikator"]
best_accuracy = results_df.loc[0, "Dokładność [%]"]

print(f"Najlepszy klasyfikator: {best_name} ({best_accuracy:.2f}%)")

# Najlepszy jest klasyfikator, który uzyskał najwyższą dokładność na zbiorze testowym, czyli 11NN (98%)

#k-NN (3, 5, 11): Uczy, że liczba sąsiadów ma znaczenie. Zbyt mała (np. 1) może prowadzić do overfittingu (wyłapywania szumu), a większa (np. 11) zazwyczaj lepiej uśrednia wynik i jest odporniejsza na błędy.

#Naive Bayes Pokazuje, że prosta statystyka czasem wystarczy.

#MLP Wprowadza w świat sieci neuronowych – najbardziej potężnych, ale i najbardziej skomplikowanych modeli.

# skalowanie - Większość modeli (poza drzewami) "boi się" dużych różnic w skali. Bez skalowania model myślałby, że cecha z większymi liczbami jest ważniejsza, mimo że w rzeczywistości może być zupełnie odwrotnie.
# potoki - łączenie skalowania danych i podanie ich do modelu od razu.

# Naive Bayes: Zakłada, że cechy są od siebie całkowicie niezależne
