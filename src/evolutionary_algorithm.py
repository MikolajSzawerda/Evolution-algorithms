import random
from typing import Callable, TypeAlias
from collections import namedtuple
from functools import partial

import numpy as np

Specimen = namedtuple("Specimen", "x score")

Population: TypeAlias = list[Specimen]


def tournament_reproduction(population: Population) -> Population:
    new_population = []

    for _ in range(len(population)):
        tournament = random.choices(population, k=2)
        new_population.append(min(tournament, key=lambda specimen: specimen.score))

    return new_population


def mutate(f: Callable, population: Population, mutation_strength: float) -> Population:
    mutants = []
    for specimen in population:
        new_x = np.array(specimen.x) + mutation_strength * np.random.normal(size=len(population[0].x))
        mutants.append(Specimen(list(new_x), f(new_x)))
    return mutants


def random_selection(old_population: Population, mutants: Population) -> Population:
    combined_population = old_population + mutants

    return random.choices(combined_population, k=len(old_population))


def fitness_proportionate_selection(old_population: Population, mutants: Population) -> Population:
    combined_population = old_population + mutants

    combined_population2 = [Specimen(s.x, s.score + 1000) for s in combined_population]
    sum_of_scores = sum(specimen.score for specimen in combined_population2)

    choices = random.choices(
        range(len(combined_population)),
        k=len(old_population),
        weights=[specimen.score / sum_of_scores for specimen in combined_population2]
    )

    return [combined_population[i] for i in choices]


def elite_selection(
        old_population: Population,
        mutants: Population,
        elite_size: int,
) -> Population:
    elite = sorted(old_population, key=lambda specimen: specimen.score)[:elite_size]
    combined_population = sorted(mutants + elite, key=lambda specimen: specimen.score)

    return combined_population[:len(old_population)]


def density_selection(
        old_population: Population,
        mutants: Population,
) -> Population:
    combined_population = old_population + mutants

    while len(combined_population) > len(mutants):
        specimen1_idx, specimen2_idx = _find_two_nearest_specimens(combined_population)
        combined_population.pop(random.choice([specimen1_idx, specimen2_idx]))

    return list(combined_population)


def _find_two_nearest_specimens(population: Population) -> tuple[Specimen, Specimen]:
    nearest_specimen = [0, 1]
    min_dist = float("inf")

    for i, specimen1 in enumerate(population):
        for j, specimen2 in enumerate(population):
            if i == j:
                continue

            dist = np.linalg.norm(np.array(specimen1.x) - np.array(specimen2.x))
            if dist < min_dist:
                min_dist = dist
                nearest_specimen = [i, j]

    return nearest_specimen


def classic_evolution(
        f: Callable[[list[float]], float],
        population: Population,
        t_max: int,
        mutation_strength: float,
        selection_strategy: Callable[[Population, Population], Population],
) -> Specimen:
    best_specimen = min(population, key=lambda specimen: specimen.score)

    for t in range(t_max):
        newborns = tournament_reproduction(population)
        mutants = mutate(f, newborns, mutation_strength)

        candidate = min(mutants, key=lambda specimen: specimen.score)

        if candidate.score <= best_specimen.score:
            best_specimen = candidate

        population = selection_strategy(population, mutants)

    return best_specimen


if __name__ == "__main__":
    ff = lambda x: 14 * x[0]**4 + 12*x[0]**3 - 41*x[0]**2- 9*x[0] + 20

    init_pop = (250 + 250) * np.random.sample((20, 1)) - 250
    init_pop = [Specimen(list(a), ff(a)) for a in init_pop]

    print("Random: ", classic_evolution(ff, init_pop, 100, 0.5, random_selection))
    print("Proportional: ", classic_evolution(ff, init_pop, 100, 0.5, fitness_proportionate_selection))
    print("Elite: ", classic_evolution(ff, init_pop, 100, 0.5, partial(elite_selection, elite_size=1)))
    print("Density: ", classic_evolution(ff, init_pop, 100, 0.5, density_selection))
