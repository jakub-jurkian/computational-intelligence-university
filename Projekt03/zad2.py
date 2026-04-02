# zad. 2
# maszyna sama się uczy klasyfikować gatunki irysów za pomocą struktury drzewiastej.
# Działanie -> Algo zadaje pytania o wymiary płatków i na podstawie odpowiedzi (tak/nie)
# schodzi niżej, aż dotrze do konkretnego gatunku.
# Wizualizacja schematu oraz sprawdzenie skuteczności za pomocą macierzy błędów.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
# DecisionTreeClassifier to sam algorytm, plot_tree do narysowania

df = pd.read_csv("iris_big.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=300806)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]

test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

tree = DecisionTreeClassifier(random_state=300806) # stworzenie modelu
tree.fit(train_inputs, train_classes)
# PROCES NAUKI (FITOWANIE) - algo analizuje dane treningowe i buduje reguły if/else.

# Rysowanie graficznego drzewa
plt.figure(figsize=(16, 10))
plot_tree(
    tree,
    feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    class_names=["setosa", "versicolor", "virginica"],
    filled=True,
    rounded=True,
)

plt.title("Drzewo decyzyjne dla iris_big.csv")
plt.savefig("drzewo_iris.png", dpi=300, bbox_inches="tight")

accuracy = tree.score(test_inputs, test_classes)
# sprawdzanie dokładności
print(accuracy * 100, "%")
print()

# macierz błędów - informuje jak model się mylił, z czym konkretnie
predictions = tree.predict(test_inputs)
cm = confusion_matrix(test_classes, predictions, labels=["setosa", "versicolor", "virginica"])

print("Macierz błędów:")
print(cm)
