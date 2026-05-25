import pygad
import numpy
import time

# Macierz labiryntu 12x12 z obramowaniem (1 = ściana, 0 = wolna ścieżka)
maze = numpy.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],  # Start jest w punkcie (1, 1)
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Wyjście jest w punkcie (10, 10)
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

# 0: góra, 1: dół, 2: lewo, 3: prawo
gene_space = [0, 1, 2, 3]
num_genes = 30  # Maksymalna liczba kroków


def fitness_func(_, solution, __):
    curr_x, curr_y = 1, 1  # Pozycja startowa
    exit_x, exit_y = 10, 10  # Pozycja docelowa

    for move in solution:
        next_x, next_y = curr_x, curr_y

        if move == 0:  # Góra
            next_x -= 1
        elif move == 1:  # Dół
            next_x += 1
        elif move == 2:  # Lewo
            next_y -= 1
        elif move == 3:  # Prawo
            next_y += 1

        # Ruch wykonujemy wtedy, gdy docelowe pole nie jest ścianą
        if maze[next_x, next_y] == 0:
            curr_x, curr_y = next_x, next_y

        # sprawdzenie, czy osiągnięto wyjście
        if curr_x == exit_x and curr_y == exit_y:
            return 100.0  # Duża nagroda za przejście labiryntu

    # Jeśli nie osiągnięto wyjścia, liczymy ujemną odległość Manhattan do celu
    distance = abs(curr_x - exit_x) + abs(curr_y - exit_y)
    return -float(distance)
    # oznacza ile „kroków w bok i w górę/dół” dzieli aktualną pozycję od celu
    # ujemnie bo algo gene. maksymalizuje fitness (im blizej celu, tym mniej ujemny wynik)

# Zwiększone parametry populacji ze względu na złożoność problemu
sol_per_pop = 300
num_parents_mating = 150
num_generations = 500
keep_parents = 10

parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10  # Wystarczająco dużo, by uniknąć ostrzeżeń dla długości 30

success_count = 0
execution_times = []
best_overall_fitness = -1e9
best_overall_solution = None

# Pętla testowa z 10 uruchomieniami
for run in range(10):
    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes, # type: ignore[arg-type]
        stop_criteria=["reach_100"],
        suppress_warnings=True,
    )

    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    solution, solution_fitness, _ = ga_instance.best_solution()

    if solution_fitness > best_overall_fitness:
        best_overall_fitness = solution_fitness
        best_overall_solution = solution.copy()

    if solution_fitness == 100.0:
        success_count += 1
        execution_times.append(end_time - start_time)

    print(f"Run {run + 1}:")
    print(f"Parametry najlepszego rozwiązania: {solution}")
    print(f"Wartość fitness najlepszego rozwiązania = {solution_fitness}")

    if run == 9:
        try:
            ga_instance.plot_fitness(title=f"Uruchomienie {run + 1} - fitness")
        except Exception:
            pass

# Końcowe statystyki
success_rate_percentage = (success_count / 10) * 100
print(f"Skuteczność: {success_rate_percentage:.1f}% ({success_count}/10)")

if execution_times:
    average_time = float(numpy.mean(execution_times))
    print(f"Średni czas wykonania dla udanych uruchomień: {average_time:.4f} s")
else:
    print(
        "W żadnym uruchomieniu nie osiągnięto wyjścia z labiryntu. Rozważ zwiększenie liczby pokoleń lub populacji."
    )

if best_overall_solution is not None:
    print("\nNajlepszy wynik ogólny:")
    print(f"Parametry najlepszego rozwiązania: {best_overall_solution}")
    print(f"Wartość fitness najlepszego rozwiązania = {best_overall_fitness}")
