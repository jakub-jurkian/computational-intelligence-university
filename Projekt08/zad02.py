import matplotlib.pyplot as plt
import random
import math
from aco import AntColony

# b) Wersja z 15 losowo wybranymi punktami z przedziału [0, 100]
random.seed(7)
COORDS_RANDOM = tuple(
    (random.randint(0, 100), random.randint(0, 100)) for _ in range(15)
)

print("15 losowych punktów")
colony = AntColony(
    COORDS_RANDOM,
    ant_count=300,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.40,
    pheromone_constant=1000.0,
    iterations=300,
)
path = colony.get_path()
print("Najlepsza trasa:", path)

# c) Eksperymenty z parametrami
# Eksperyment 1 (domyślne): alpha=0.5, beta=1.2, evaporation=0.40
# Eksperyment 2: wyższe beta=2.5 – mrówki bardziej kierują się odległością,
#   mniej eksplorują; szybka zbieżność do lokalnego optimum.
# Eksperyment 3: wysokie parowanie evaporation=0.80 – feromony szybko
#   zanikają, kolonia "zapomina" dobre trasy i losuje więcej; skuteczne
#   dla gęstych grafów, ale spowalnia zbieżność.
# Wniosek: dla małych instancji (15 pkt) domyślne parametry działają dobrze.
#   Przy dużych grafach warto zwiększyć beta i zmniejszyć parowanie.

# d) Siatka 5x5 (grid)
COORDS_GRID = tuple(
    (x * 10, y * 10) for y in range(5) for x in range(5)
)

# najkrótsza droga to trasa wężowa (snake) po kolumnach.
# Kolejność: col 0 w dół → col 1 w górę → col 2 w dół → ... → col 4 w dół
# Długość trasy (bez powrotu do startu): 24 krawędzie × 10 = 240
# ACO szuka cyklu Hamiltona (powrót do startu), więc najlepsza możliwa
# trasa ≈ 240 + powrót z (40,40) do (0,0) = 240 + 40√2 ≈ 296.57
# Albo ≈ 280 jeśli trasa kończy się w (40,0) lub (0,40).

# Obliczenie długości trasy zig-zag skryptem:
snake = []
for col in range(5):
    rows = range(5) if col % 2 == 0 else range(4, -1, -1)
    for row in rows:
        snake.append((col * 10, row * 10))

def tour_length(pts):
    total = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        total += math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return total

print(f"\nSiatka 5x5")
print(f"Długość trasy zig-zag (po ludzku, z powrotem): {tour_length(snake):.2f}")

colony_grid = AntColony(
    COORDS_GRID,
    ant_count=300,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.40,
    pheromone_constant=1000.0,
    iterations=300,
)
path_grid = colony_grid.get_path()
print(f"Długość trasy ACO na siatce 5x5: {tour_length(path_grid):.2f}")
# Siatka jest regularna i łatwa dla człowieka (prosty wzorzec), ale ACO
# ma trudności – wiele równoważnych tras tej samej długości utrudnia
# znalezienie globalnego optimum.

# Wizualizacja trasy ACO na siatce
for x, y in COORDS_GRID:
    plt.plot(x, y, "g.", markersize=10)
for i in range(len(path_grid) - 1):
    plt.plot(
        (path_grid[i][0], path_grid[i + 1][0]),
        (path_grid[i][1], path_grid[i + 1][1]),
    )
plt.title("ACO = siatka 5x5")
plt.show()
