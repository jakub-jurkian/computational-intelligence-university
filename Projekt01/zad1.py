# zad. 1
import datetime
from math import sin, pi

# pobieranie danych
name = input("Podaj swoje imię: ")
year_of_birth = int(input("Podaj rok urodzenia: "))
month_of_birth = int(input("Podaj miesiąc urodzenia: "))
day_of_birth = int(input("Podaj dzień urodzenia: "))

# obliczanie przeżytych dni usera
birth_date = datetime.date(year_of_birth, month_of_birth, day_of_birth)
today = datetime.date.today()
amount_of_days = (today - birth_date).days

# zastosowanie wzorów na fale
physical_wave = sin((2 * pi / 23) * amount_of_days)
emotional_wave = sin((2 * pi / 28) * amount_of_days)
intellectual_wave = sin((2 * pi / 33) * amount_of_days)


print(f"Cześć {name}! Dzisiaj jest Twój {amount_of_days} dzień na Ziemi.")
print(f"Fizyczny biorytm: {physical_wave:.2f}")
print(f"Emocjonalny biorytm: {emotional_wave:.2f}")
print(f"Intelektualny biorytm: {intellectual_wave:.2f}")
print("Wartości mieszczą się w zakresie [-1, 1].")


def analyze_wave(wave, name, typ, days, period):
    if wave > 0.5:
        print(f"Gratulacje! Twój {typ} biorytm jest dziś wysoki.")
    elif wave < -0.5:
        print(f"{name}, Twój {typ} biorytm jest dziś niski. Głowa do góry!")
        next_wave = sin((2 * pi / period) * (days + 1))
        if next_wave > wave:
            print("Nie martw się. Jutro będzie lepiej!")
        else:
            print("Jutro wynik będzie podobny lub niższy.")


analyze_wave(physical_wave, name, "fizyczny", amount_of_days, 23)
analyze_wave(emotional_wave, name, "emocjonalny", amount_of_days, 28)
analyze_wave(intellectual_wave, name, "intelektualny", amount_of_days, 33)

# 40min