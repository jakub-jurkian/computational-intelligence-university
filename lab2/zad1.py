import pandas as pd


# Ładowanie danych (pomijanie błędnych wierszy)
df = pd.read_csv("iris_big_with_errors.csv", on_bad_lines="skip")

print("a) sprawdzenie struktury")
print(f"Struktura po załadowaniu: (liczba wierszy, liczba kolumn) -> {df.shape} (oczekiwane 1500 wierszy, jest {len(df)} - {1500-len(df)} błędne wiersze pominięte)")
print(f"Kolumny: {list(df.columns)}")
print()


print("b) brakujące wartości w kolumnach")
print(df.isnull().sum())
print()
print("Opis statystyczny (przed czyszczeniem):")
print(df.describe()) # ile niepustych wartosci, ile roznych wartosci, najczestsza wartosc, ile razy wystepuje top
print()

kolumny_liczbowe = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# Zamiana przecinka dziesiętnego na kropkę (np. '4,75' -> '4.75') - konwersja wartosci na tekst.
for col in kolumny_liczbowe:
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)

# Konwersja do typu liczbowego - wartości nieliczbowe stają się NaN
for col in kolumny_liczbowe:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Zamiana wartości spoza zakresu (<=0 lub >=15) na średnią kolumny
for col in kolumny_liczbowe:
    srednia_kolumny = df[col].mean()
    df.loc[(df[col] <= 0) | (df[col] >= 15), col] = srednia_kolumny

# Uzupełnienie pozostałych NaN średnią kolumny
for col in kolumny_liczbowe:
    srednia_kolumny = df[col].mean()
    df[col] = df[col].fillna(srednia_kolumny)

# Naprawa błędów w kolumnie target_name
print("d) Błędy w nazwach klas")
print("Przed czyszczeniem:")
print(df["target_name"].value_counts())
print()

# Usunięcie białych znaków, zamiana na małe litery i usunięcie znaków specjalnych
df["target_name"] = df["target_name"].str.strip().str.lower()
df["target_name"] = df["target_name"].str.replace(".", "", regex=False)
df["target_name"] = df["target_name"].str.replace("?", "", regex=False)
df["target_name"] = df["target_name"].str.replace("-", "", regex=False)
df["target_name"] = df["target_name"].str.replace("_", "", regex=False)
df["target_name"] = df["target_name"].str.replace(" ", "", regex=False)

# Mapowanie znanych literówek i prefiksów do poprawnych nazw
mapa_nazw = {
    "setosa": "setosa",
    "irissetosa": "setosa",
    "versicolor": "versicolor",
    "irisversicolor": "versicolor",
    "versicolr": "versicolor",
    "virginica": "virginica",
    "irisvirginica": "virginica",
}

df["target_name"] = df["target_name"].map(mapa_nazw)

# Wiersze, których nie udało się zmapować, potem uzupełnienie dominantą (np . "unknown")
niezmapowane = df["target_name"].isna()
if niezmapowane.any():
    print(f"Niezmapowane wiersze target_name: {niezmapowane.sum()} - uzupełnianie dominantą")
    df["target_name"] = df["target_name"].fillna(df["target_name"].mode()[0])

print("\nPo czyszczeniu:")
print(df["target_name"].value_counts())
print()

# Podsumowanie
print("=== Dane po czyszczeniu ===")
print(f"Rozmiar: {df.shape}")
print(df.astype("string").describe())
print()
print("Brakujące wartości po czyszczeniu:")
print(df.isnull().sum())


# Wyczyściłem błędny zbiór danych: naprawiłem złe wartości
# liczbowe i nazwy klas, uzupełniłem braki, dzięki czemu dostałem 
# spójne dane gotowe do analizy.