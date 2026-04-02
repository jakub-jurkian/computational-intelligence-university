import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Wczytanie oczyszczonego zbioru danych
df = pd.read_csv("iris_big.csv")

kolumny_cech = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
X = df[kolumny_cech].values
y = df["target_name"]

# PCA z 4 komponentami, aby sprawdzić wariancję
pca_full = PCA(n_components=4).fit(X)

print("PCA – wyjaśnione proporcje wariancji")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"  Komponent {i+1}: {ratio:.4f}  ({ratio*100:.2f}%)")

wariancja_suma = pca_full.explained_variance_ratio_.cumsum()
print("\nSkumulowana wariancja:")
for i, c in enumerate(wariancja_suma):
    print(f"  Po {i+1} komponentach: {c:.4f}  ({c*100:.2f}%)")


n = len(pca_full.explained_variance_ratio_)
print("\n=== Utrata informacji przy usuwaniu ostatnich i kolumn ===")
for i in range(1, n):
    utrata = pca_full.explained_variance_ratio_[n - i:].sum()
    print(f"  Usuń ostatnie {i} kolumny: utrata = {utrata:.4f}  ({utrata*100:.2f}%)")
    if utrata <= 0.05:
        print(f"    -> Można usunąć {i} kolumny (utrata <= 5%)")
    else:
        print(f"    -> Nie można usunąć {i} kolumny (utrata > 5%)")

# PCA zredukowane do 2 komponentów – wykres 2D (punktowy)
pca2 = PCA(n_components=2)
X2 = pca2.fit_transform(X)

plt.figure(figsize=(8, 6))
gatunki = y.unique()
kolory = ["blue", "green", "red"]
for gatunek, kolor in zip(gatunki, kolory):
    maska = y == gatunek
    plt.scatter(X2[maska, 0], X2[maska, 1], label=gatunek, alpha=0.5, s=15, c=kolor)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA – Zbiór Iris (2 komponenty)")
plt.legend()
plt.tight_layout()
plt.savefig("zad2_pca_2d.png", dpi=150)
plt.show()

# PCA zredukowane do 3 komponentów – wykres 3D
pca3 = PCA(n_components=3)
X3 = pca3.fit_transform(X)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
for gatunek, kolor in zip(gatunki, kolory):
    maska = y == gatunek
    ax.scatter(X3[maska, 0], X3[maska, 1], X3[maska, 2], label=gatunek, alpha=0.5, s=15, c=kolor)
ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("PCA – Zbiór Iris (3 komponenty)")
ax.legend()
plt.tight_layout()
plt.savefig("zad2_pca_3d.png", dpi=150)
plt.show()

# Zastosowałem PCA, żeby zredukować liczbę kolumn ale żeby wartość informacji pozostała na wysokim poziomie.
# potem pokazałem to na wykresach 2D i 3D dla trzech klas irysów.