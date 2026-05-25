import pygad
import numpy
import math

# Zadanie stopu metali - jaka jest najlepsza wytrzymałość stopu metali
# oraz jakie ilości metali trzeba zmieszać, by uzyskać najbardziej wytrzymały stop.

# Chromosom ma 6 genów, bo wzór endurance(x, y, z, u, v, w) ma 6 argumentów.
num_genes = 6

# b) Dla genów ciągłych używamy przestrzeni [0.0, 1.0).
gene_space = {"low": 0.0, "high": 1.0}


# c) Funkcja fitness jest tu prosta: zwraca bezpośrednio wytrzymałość stopu.
def fitness_func(model, solution, solution_idx):
    x, y, z, u, v, w = solution
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


fitness_function = fitness_func

# d) Sensowne parametry dla krótkiego chromosomu.
sol_per_pop = 30
num_parents_mating = 15
num_generations = 60
keep_parents = 2

# sss = steady, rws = roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"
crossover_type = "single_point"

# e) Dla 6 genów mutację zwiększamy do kilkunastu procent, żeby uniknąć warninga.
mutation_type = "random"
mutation_percent_genes = 20

best_overall_fitness = -1e9
best_overall_solution = None
fitness_history = []

for run in range(5):
    # Inicjacja algorytmu z parametrami wpisanymi w atrybuty.
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,  # type: ignore[arg-type]
    )

    # Uruchomienie algorytmu.
    ga_instance.run()

    # Podsumowanie: najlepsze znalezione rozwiązanie (chromosom + ocena).
    solution, solution_fitness, _ = ga_instance.best_solution()
    fitness_history.append(float(solution_fitness))

    if solution_fitness > best_overall_fitness:
        best_overall_fitness = solution_fitness
        best_overall_solution = solution.copy()

    print(f"Uruchomienie {run + 1}:")
    print(f"Parametry najlepszego rozwiązania: {solution}")
    print(f"Wartość fitness najlepszego rozwiązania = {solution_fitness}")

    # Wyświetlenie wykresu zmian funkcji fitness w kolejnych pokoleniach.
    if run == 4:
        ga_instance.plot_fitness(title=f"Uruchomienie {run + 1} - fitness")

print("\nNajlepszy wynik ogólny:")
print(f"Parametry najlepszego rozwiązania: {best_overall_solution}")
print(f"Wartość fitness najlepszego rozwiązania = {best_overall_fitness}")
print(f"Fitness z poszczególnych uruchomień: {fitness_history}")
print(f"Średni fitness z uruchomień: {numpy.mean(fitness_history):.6f}")

