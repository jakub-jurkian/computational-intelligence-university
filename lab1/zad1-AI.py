import datetime
from math import sin, pi


def get_user_data():
    """Pobiera dane użytkownika i zwraca je jako słownik."""
    name = input("Podaj swoje imię: ")
    try:
        year = int(input("Podaj rok urodzenia: "))
        month = int(input("Podaj miesiąc urodzenia (1-12): "))
        day = int(input("Podaj dzień urodzenia (1-31): "))
        birth_date = datetime.date(year, month, day)
    except ValueError:
        print("Podano nieprawidłowe dane. Spróbuj ponownie.")
        return get_user_data()
    return {"name": name, "birth_date": birth_date}


def calculate_biorhythms(days_alive):
    """Zwraca słownik z wartościami biorytmów."""
    return {
        "fizyczny": sin((2 * pi / 23) * days_alive),
        "emocjonalny": sin((2 * pi / 28) * days_alive),
        "intelektualny": sin((2 * pi / 33) * days_alive),
    }


def print_biorhythms(name, days_alive, biorhythms):
    print(f"\nCześć {name}! Dzisiaj jest Twój {days_alive} dzień na Ziemi.")
    for typ, value in biorhythms.items():
        print(f"{typ.capitalize()} biorytm: {value:.2f}")
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


def main():
    user = get_user_data()
    today = datetime.date.today()
    days_alive = (today - user["birth_date"]).days
    biorhythms = calculate_biorhythms(days_alive)
    print_biorhythms(user["name"], days_alive, biorhythms)
    periods = {"fizyczny": 23, "emocjonalny": 28, "intelektualny": 33}
    for typ, value in biorhythms.items():
        analyze_wave(value, user["name"], typ, days_alive, periods[typ])


if __name__ == "__main__":
    main()


# 20 seconds