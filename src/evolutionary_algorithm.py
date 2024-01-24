import random
from typing import Callable, TypeAlias
from collections import namedtuple

import numpy as np

Specimen = namedtuple("Specimen", "x score")

Population: TypeAlias = list[Specimen]

Log: TypeAlias = list[Population]


def _tournament_reproduction(population: Population) -> Population:
    new_population = []

    for _ in range(len(population)):
        tournament = random.choices(population, k=2)
        new_population.append(min(tournament, key=lambda specimen: specimen.score))

    return new_population


def _mutate(
        func: Callable,
        population: Population,
        mutation_strength: float,
        bounds: tuple[float, float],
) -> Population:
    mutants = []
    for specimen in population:
        new_x = list(np.array(specimen.x) + mutation_strength * np.random.normal(size=len(population[0].x)))

        for i in range(len(new_x)):
            if new_x[i] < bounds[0]:
                new_x[i] = bounds[0]
            if new_x[i] > bounds[1]:
                new_x[i] = bounds[1]

        mutants.append(Specimen(new_x, func(new_x)))
    return mutants


def random_selection(old_population: Population, mutants: Population) -> Population:
    combined_population = old_population + mutants

    return random.choices(combined_population, k=len(old_population))


def fitness_proportionate_selection(old_population: Population, mutants: Population) -> Population:
    combined_population = old_population + mutants

    choices = random.choices(
        range(len(combined_population)),
        k=len(old_population),
        weights=[1 / abs(specimen.score) for specimen in combined_population]
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


def generation_selection(old_population: Population, mutants: Population) -> Population:
    return mutants


def density_selection(
        old_population: Population,
        mutants: Population,
) -> Population:
    combined_population = old_population + mutants

    distances = []
    for i in range(len(combined_population)):
        for j in range(i+1, len(combined_population)):
            specimen1 = combined_population[i]
            specimen2 = combined_population[j]

            dist = np.linalg.norm(np.array(specimen1.x) - np.array(specimen2.x))
            distances.append((dist, i, j))

    distances.sort(reverse=True)

    indicies_to_pop = set()

    while len(indicies_to_pop) < len(mutants):
        _, specimen1_idx, specimen2_idx = distances.pop()
        indicies_to_pop.add(random.choice([specimen1_idx, specimen2_idx]))

    new_population = [
        combined_population[i] for i in range(len(combined_population))
        if i not in indicies_to_pop
    ]

    return new_population


def _initialize_population(func: Callable, population_size: int, bounds: tuple[float, float], dim: int) -> Population:
    init_pop = [list(np.random.uniform(bounds[0], bounds[1], dim)) for _ in range(population_size)]
    return [Specimen(x, func(x)) for x in init_pop]


def classic_evolution(
        func: Callable[[list[float]], float],
        population_size: int,
        dimensions: int,
        bounds: tuple[float, float],
        t_max: int,
        mutation_strength: float,
        selection_strategy: Callable[[Population, Population], Population],
) -> tuple[Specimen, Log]:
    log = []
    population = _initialize_population(func, population_size, bounds, dimensions)
    best_specimen = min(population, key=lambda specimen: specimen.score)

    for t in range(t_max):
        log.append(population)
        newborns = _tournament_reproduction(population)
        mutants = _mutate(func, newborns, mutation_strength, bounds)

        candidate = min(mutants, key=lambda specimen: specimen.score)

        if candidate.score <= best_specimen.score:
            best_specimen = candidate

        population = selection_strategy(population, mutants)

    return best_specimen, log
