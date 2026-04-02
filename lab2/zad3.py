import pandas as pd  # do pracy z danymi w formie tabelarycznej
import matplotlib.pyplot as plt  # do tworzenia wykresow
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
)  # do normalizacji i standaryzacji danych

# Wczytanie oczyszczonego zbioru danych
df = pd.read_csv("iris_big.csv")

kolumna_x = "sepal length (cm)"
kolumna_y = "sepal width (cm)"
gatunki = df["target_name"].unique()
kolory = ["blue", "green", "red"]

X = df[[kolumna_x, kolumna_y]].values

# Przygotowanie 3 wersji danych
X_oryginalne = X.copy()
X_minmax = MinMaxScaler().fit_transform(X)  # przeskalowane dane do zakresu [0, 1]
X_zscore = StandardScaler().fit_transform(
    X
)  # przekształcone dane aby mialy srednia 0 i odchyl. st. 1

zbiory_danych = [
    ("Dane oryginalne", X_oryginalne),
    ("Normalizacja Min-Max", X_minmax),
    ("Standaryzacja Z-score", X_zscore),
]

# Wykresy dla 3 wersji
fig, osie = plt.subplots(1, 3, figsize=(18, 5))

for os, (tytul, dane) in zip(osie, zbiory_danych):
    for gatunek, kolor in zip(gatunki, kolory):
        maska = df["target_name"] == gatunek
        os.scatter(
            dane[maska, 0], dane[maska, 1], label=gatunek, alpha=0.5, s=15, c=kolor
        )
    os.set_xlabel(kolumna_x)
    os.set_ylabel(kolumna_y)
    os.set_title(tytul)
    os.legend()

plt.tight_layout()
plt.savefig("zad3_normalization.png", dpi=150)
plt.show()

# Statystyki dla każdej wersji
for tytul, dane in zbiory_danych:
    tmp = pd.DataFrame(dane, columns=[kolumna_x, kolumna_y])
    print(f"=== {tytul} ===")
    print(f"  Min:    {tmp.min().values}")
    print(f"  Max:    {tmp.max().values}")
    print(f"  Średnia:   {tmp.mean().values}")
    print(f"  Odchylenie standardowe:    {tmp.std().values}")
    print()

# zadanie pokazuje, że normalizacja i standaryzacja nie zmieniają klas,
# ale zmieniają skalę cech, co jest kluczowe dla dalszych dzialan.
