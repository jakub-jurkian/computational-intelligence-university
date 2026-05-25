import numpy as np
import pyswarms as ps
import time

# Definicja macierzy labiryntu 12x12 (1 = ściana, 0 = wolna ścieżka)
maze = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

num_steps = 30  # Liczba wymiarów odpowiadająca maksymalnej liczbie kroków agenta


def maze_cost_function(x):
    n_particles = x.shape[0]
    costs = np.zeros(n_particles)

    for i in range(n_particles):
        curr_x, curr_y = 1, 1
        exit_x, exit_y = 10, 10

        # Mapowanie pozycji ciągłej na dyskretne kierunki: 0, 1, 2, 3 za pomocą funkcji floor
        discrete_moves = np.floor(x[i]).astype(int)

        for move in discrete_moves:
            next_x, next_y = curr_x, curr_y

            if move == 0:  # Góra
                next_x -= 1
            elif move == 1:  # Dół
                next_x += 1
            elif move == 2:  # Lewo
                next_y -= 1
            elif move == 3:  # Prawo
                next_y += 1

            if maze[next_x, next_y] == 0:
                curr_x, curr_y = next_x, next_y

            if curr_x == exit_x and curr_y == exit_y:
                break

        if curr_x == exit_x and curr_y == exit_y:
            costs[i] = -100.0  # Sukces: Osiągnięto cel (najniższy możliwy koszt)
        else:
            # Odległość Manhattan, którą algorytm stara się zminimalizować do zera
            costs[i] = float(abs(curr_x - exit_x) + abs(curr_y - exit_y))

    return costs


# Dziedzina ciągła zmapowana od 0 do 3.999, aby zaokrąglenie w dół dawało poprawne indeksy ruchów
min_bounds = np.zeros(num_steps)
max_bounds = np.ones(num_steps) * 3.999
bounds = (min_bounds, max_bounds)

options = {"c1": 0.6, "c2": 0.4, "w": 0.8}

success_count = 0
execution_times = []

# Pętla wykonująca 10 niezależnych uruchomień statystycznych
for run in range(10):
    optimizer = ps.single.GlobalBestPSO(
        n_particles=300, dimensions=num_steps, options=options, bounds=bounds
    )

    start_time = time.time()
    best_cost, best_pos = optimizer.optimize(
        maze_cost_function, iters=200, verbose=False
    )
    end_time = time.time()

    if best_cost == -100.0:
        success_count += 1
        execution_times.append(end_time - start_time)

success_rate_percentage = (success_count / 10) * 100
print(f"Skuteczność wyjścia z labiryntu: {success_rate_percentage}%")

if execution_times:
    average_time = np.mean(execution_times)
    print(f"Średni czas wykonania dla udanych prób: {average_time:.4f} sekund")
