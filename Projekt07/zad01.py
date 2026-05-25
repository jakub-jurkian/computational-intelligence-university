# Zadanie plecakowe - ile najwięcej wartościowych rzeczy jest się w stanie zabrać

import pygad
import numpy
import time

# Ten sam indeks w obu tablicach oznacza ten sam przedmiot (np. indeks 0 to zegar)
# a) używam list równoległych (wagi, wartości, nazwy).
#    Chromosom to binarna tablica długości równej liczbie przedmiotów (1 = zabieramy przedmiot).
wagi = numpy.array([7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15])
wartosci = numpy.array([100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300])
przedmioty = [
    "zegar",
    "obraz-pejzaż",
    "obraz-portret",
    "radio",
    "laptop",
    "lampka nocna",
    "srebrne sztućce",
    "porcelana",
    "figura z brązu",
    "skórzana torebka",
    "odkurzacz",
]
max_waga = 25

# Definiujemy parametry chromosomu.
# Geny to liczby: 0 lub 1.
gene_space = [0, 1]


# Definiujemy funkcję fitness (przypisuje chromosomowi ocenę mówiącą, jak dobre jest to rozwiązanie
# według kryteriów problemu - ocena służy do porównywania i wyboru osobników do dalszego działania).
# Chromosom ('solution') to binarna tablica, np. [1, 0, 1, 0]
# ocenia chromosom i używa wektoryzacji (solution * wagi), aby policzyć sumy szybciej.
def fitness_func(model, solution, solution_idx):
    # b) Funkcja fitness: sumujemy wartość wybranych przedmiotów.
    # Jeśli przekroczymy limit wagi, nakładamy karę, dzięki temu algorytm 
    # widzi kierunek poprawy (mniejsze przekroczenie wagi).
    laczna_waga = float(numpy.dot(solution, wagi))
    laczna_wartosc = float(numpy.dot(solution, wartosci))

    if laczna_waga <= max_waga:
        return laczna_wartosc

    # Kara proporcjonalna do przekroczenia wagi (duża, aby preferować dopuszczalne rozwiązania).
    penalty = 100.0 * (laczna_waga - max_waga)
    fitness = laczna_wartosc - penalty
    return fitness


fitness_function = fitness_func


sol_per_pop = 30 # Rozmiar populacji — ile chromosomów w każdym pokoleniu.
num_genes = len(wagi) # Długość chromosomu.

num_parents_mating = 15 # Ile rodziców dobieramy do krzyżowania.
num_generations = 200 # Maksymalna liczba pokoleń.
keep_parents = 2 # Ile rodziców zachowujemy bez zmian - chroni najlepsze rozwiązania.

# Jaki typ selekcji rodziców?
# sss = steady, rws = roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

crossover_type = "single_point" # Dzieli i wymienia części.

mutation_type = "random" # Sposób mutacji.
mutation_percent_genes = 10 # Procent genów poddawanych mutacji.

success_count = 0
execution_times = []
best_overall_fitness = -1e9
best_overall_solution = None

runs = 10
for run in range(runs):
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
        mutation_percent_genes=mutation_percent_genes, # type: ignore[arg-type]
        stop_criteria=["reach_1630"],
        suppress_warnings=True,
    )

    # Uruchomienie algorytmu.
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    # Podsumowanie: najlepsze znalezione rozwiązanie (chromosom + ocena).
    solution, solution_fitness, _ = ga_instance.best_solution()

    # Aktualizacja najlepszego wyniku ogólnego.
    if solution_fitness > best_overall_fitness:
        best_overall_fitness = solution_fitness
        best_overall_solution = solution.copy()

    if solution_fitness >= 1630:
        success_count += 1
        execution_times.append(end_time - start_time)

    # Dla jednego z uruchomień rysujemy wykres przebiegu optymalizacji.
    if run == runs - 1:
        try:
            ga_instance.plot_fitness(title=f"Run {run + 1} - Fitness")
        except Exception:
            pass

# e) f) statystyki.
success_rate_percentage = (success_count / runs) * 100
print(f"Skuteczność: {success_rate_percentage:.1f}% ({success_count}/{runs})")

if execution_times:
    average_time = float(numpy.mean(execution_times))
    print(f"Średni czas wykonania dla udanych uruchomień: {average_time:.4f} s")
else:
    print("Nie znaleziono rozwiązania optymalnego (1630) w żadnym uruchomieniu. Spróbuj zmienić parametry.")

# d) wypisz najlepsze rozwiązanie znalezione w całym eksperymencie
if best_overall_solution is not None:
    chosen_idxs = numpy.where(numpy.array(best_overall_solution) == 1)[0]
    total_w = int(numpy.dot(best_overall_solution, wagi))
    total_v = int(numpy.dot(best_overall_solution, wartosci))
    print("\nNajlepsze rozwiązanie ogólne:")
    print(f"Fitness: {best_overall_fitness}")
    print(f"Łączna waga: {total_w} / {max_waga}")
    print(f"Łączna wartość: {total_v}")
    print("Przedmioty do zabrania:")
    for i in chosen_idxs:
        name = przedmioty[i] if i < len(przedmioty) else f"Item_{i}"
        print(f" - {name}: waga={wagi[i]}, wartość={wartosci[i]}")