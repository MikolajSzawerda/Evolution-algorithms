from typing import Callable

import numpy as np
from collections import namedtuple

Entity = namedtuple("Entity", "x value")
DataPoint = namedtuple("DataPoint", "t population")


def stop_condition(t, **kwargs):
    return t >= kwargs['t_max']


def tournament_reproduction(population: np.array, scores: np.array) -> np.array:
    new_population = []

    for _ in range(len(population)):
        specimen1_idx, specimen2_idx = np.random.randint(0, len(population), size=2)
        winner_idx = specimen1_idx if scores[specimen1_idx] < scores[specimen2_idx] else specimen2_idx
        new_population.append(np.copy(population[winner_idx]))

    return np.array(new_population)


def elite_selection(
        mutants: np.array,
        mutant_scores: np.array,
        old_population: np.array,
        old_population_scores: np.array,
        elite_size: int,
) -> tuple[np.array, np.array]:
    population_size = len(old_population)

    elite = sorted([
        (old_population_scores[i], old_population[i]) for i in range(population_size)
    ])[:elite_size]

    combined_population = [(mutant_scores[i], mutants[i]) for i in range(population_size)] + elite
    combined_population = sorted(combined_population)[:population_size]

    population = [x[1] for x in combined_population]
    scores = [x[0] for x in combined_population]

    return np.array(population), np.array(scores)


def mutate(population: np.array, mutation_strength: float):
    mutants = []
    for x in population:
        x = x + mutation_strength * np.random.normal(len(population[0]))
        mutants.append(x)
    return mutants


def classic_evolution(
        f: Callable[[np.array], float],
        init_population: np.array,
        t_max: int,
        mutation_strength: float,
        elite_size: int
) -> tuple[np.array, float]:
    evaluate_population_scores = lambda p: [f(a) for a in p]
    population = init_population
    population_scores = evaluate_population_scores(population)

    best_specimen_idx = np.argmin(population_scores)
    best_specimen = np.copy(population[best_specimen_idx])
    best_specimen_score = population_scores[best_specimen_idx]

    for t in range(t_max):
        newborns = tournament_reproduction(population, population_scores)
        mutants = mutate(newborns, mutation_strength)

        mutants_scores = evaluate_population_scores(mutants)
        candidate_idx = np.argmin(mutants_scores)

        if mutants_scores[candidate_idx] <= best_specimen_score:
            best_specimen = mutants[candidate_idx]

        population, population_scores = elite_selection(mutants, mutants_scores, population, population_scores, elite_size)

    return best_specimen, best_specimen_score


if __name__ == "__main__":
    f = lambda x: 14 * x[0]**4 + 12*x[0]**3 - 41*x[0]**2- 9*x[0] + 20
    init_pop = (100 + 100) * np.random.sample(20) - 100
    init_pop = [[a] for a in init_pop]

    print(classic_evolution(f, init_pop, 1000, 0.5, 1))