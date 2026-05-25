import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

# Problem inżynieryjny: Stop Metali
# Szukamy maksimum funkcji endurance za pomocą PSO (pyswarms minimalizuje,
# więc negujemy funkcję endurance).

dimensions = 6

# Ograniczenia dziedziny: wszystkie zmienne z przedziału [0, 1)
min_bounds = np.zeros(dimensions)
max_bounds = np.ones(dimensions)
bounds = (min_bounds, max_bounds)

# Hiperparametry roju
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}


# Funkcja kosztu dla całego roju (pyswarms wymaga tablicy wyników)
def cost_function(x):
    n_particles = x.shape[0]
    costs = np.zeros(n_particles)
    for i in range(n_particles):
        xi, y, z, u, v, w = x[i]
        endurance = (
            np.exp(-2 * (y - np.sin(xi)) ** 2) + np.sin(z * u) + np.cos(v * w)
        )
        # Negujemy, bo PSO minimalizuje, a my chcemy maksymalizować wytrzymałość
        costs[i] = -endurance
    return costs


optimizer = ps.single.GlobalBestPSO(
    n_particles=50, dimensions=dimensions, options=options, bounds=bounds
)

best_cost, best_pos = optimizer.optimize(cost_function, iters=1000)

maximum_endurance = -best_cost
print(f"Optymalne proporcje metali (x, y, z, u, v, w): {best_pos}")
print(f"Maksymalna wytrzymałość stopu: {maximum_endurance:.6f}")

# Wykres zbieżności roju – wytrzymałość (fitness) rośnie z iteracją
fitness_history = [-c for c in optimizer.cost_history]
plt.plot(fitness_history)
plt.title("PSO – Iteracja vs. Wytrzymałość")
plt.xlabel("Iteracja")
plt.ylabel("Wytrzymałość")
plt.show()
