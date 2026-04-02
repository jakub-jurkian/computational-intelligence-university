# zad. 2
import math 
import random
import matplotlib.pyplot as plt

v = 50  # prędkość początkową pocisku [m/s]
h = 100  # wysokość startu pocisku [m]
g = 9.81  # przyspieszenie ziemskie [m/s^2]

random_distance = random.randint(50, 340)
print(f"Distance to the object: {random_distance} m")

number_of_shots = 0

while True:
    number_of_shots += 1 
    angle = float(input("Enter the angle (in degrees): "))

    a = math.radians(angle)  # ze stopni na radiany
    vy = v * math.sin(a)  # składowa pionowa prędkości
    vx = v * math.cos(a)  # składowa pozioma prędkości
    t_total = (vy + math.sqrt(vy**2 + 2 * g * h)) / g  # całkowity czas lotu do ziemi

    impact_distance = vx * t_total  # odległość miejsca upadku pocisku
    print(f"Impact distance: {impact_distance:.2f} m")

    if abs(impact_distance - random_distance) <= 5:  # (tolerancja 5 m)
        print(f"Target destroyed! Number of shots: {number_of_shots}")

        # Rysowanie trajektorii dla trafionego strzału
        points = 300  # ustawia liczbę punktów do narysowania trajektorii
        dt = t_total / (points - 1)  # krok czasu między punktami
        x_values = []  # lista współrzędnych x
        y_values = []  # lista współrzędnych y

        for i in range(points):  # iteruje po kolejnych punktach trajektorii
            t = i * dt  # oblicza czas dla bieżącego punktu
            x = vx * t  # pozycja pozioma dla czasu t
            y = h + vy * t - 0.5 * g * t**2  # pozycja pionowa dla czasu t
            x_values.append(x)  # dodaje x do listy punktów
            y_values.append(y)  # dodaje y do listy punktów

        plt.figure(figsize=(10, 5))
        plt.plot(x_values, y_values, color="blue")
        plt.grid(True)
        plt.xlabel("Distance [m]")
        plt.ylabel("Height [m]")
        plt.title("Warwolf projectile trajectory") 
        plt.savefig("trajektoria.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("Plot saved as trajektoria.png")
        break
    elif impact_distance < random_distance:  # sprawdza, czy pocisk spadł przed celem
        print("Too short. Try again.") 
    else:
        print("Too far. Try again.")
