# manualna klasyfikacja gatunków irysów

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris_big.csv")

# dobieramy dane treningowe oraz testowe
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=300806)

# tworzymy funkcję służącą do klasyfikacji irysów (parametry na podstawie obserwacji)
def classify_iris(sl, sw, pl, pw):
    if pl < 2.5:
        return "setosa"
    elif pw > 1.7:
        return "virginica"
    else:
        return "versicolor"


good_predictions = 0
# ile wierszy (irysów) znaduje się w zbiorze testowym
len = test_set.shape[0]
# train_set -> w głowie, sam dostrzegłem z obserwacji jakie wartości ustawić w algo.

for i in range(len):
    # porównujemy wynik funkcji z prawdziwym gatunkiem
    if (
        classify_iris(
            float(test_set[i][0]),
            float(test_set[i][1]),
            float(test_set[i][2]),
            float(test_set[i][3]),
        )
        == test_set[i][4]
    ):
        good_predictions = good_predictions + 1

print(good_predictions)
# określamy dokładność klasyfikacji w %
print(f"{(good_predictions / len * 100):.2f}%")

# train_df = pd.DataFrame(train_set, columns=["sl", "sw", "pl", "pw", "species"])
# train_df = train_df.sort_values("species")
# print(train_df.to_string(index=False))