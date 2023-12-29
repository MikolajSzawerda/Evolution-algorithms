import numpy as np
import numpy.random as npr
from collections import namedtuple
from random import choices, getrandbits

Entity = namedtuple("Entity", "x value")
DataPoint = namedtuple("DataPoint", "t population")


def stop_condition(t, **kwargs):
    return t >= kwargs['t_max']


# tournament
def reproduce(population, **kwargs):
    return [min(choices(population, k=2), key=lambda x: x.value)
            for _ in range(kwargs['pop_size'])]


# # roulette
# def reproduce(population, **kwargs):
#     s = sum(1/x.value for x in population)
#     p = [(1/x.value)/s for x in population]
#     return choices(population, p, k=kwargs['pop_size'])


# # no crossover
# def crossover(population, **kwargs):
#     for p in population:
#         yield p.x


#uniform crosssover
def crossover(population, **kwargs):
    for _ in range(kwargs['pop_size']):
        a, b = choices(population, k=2)
        yield np.array([y[getrandbits(1)] for y in zip(a.x, b.x)])


# # mean
# def crossover(population, **kwargs):
#     for _ in range(kwargs['pop_size']):
#         weights = npr.normal(size=kwargs['dim'])
#         a, b = choices(population, k=2)
#         yield weights * a.x + (1-weights) * b.x


def mutate(population, f, **kwargs):
    mutants = []
    for x in crossover(population, **kwargs):
        x = x + kwargs['strength'] * npr.normal(size=kwargs['dim'])
        mutants.append(Entity(x, f(x)))
    return mutants


# elite selection
def select(mutants, old_population, **kwargs):
    mutants.extend(sorted(old_population, key=lambda x: x.value)[:kwargs['elite']])
    return sorted(mutants, key=lambda x: x.value)[:kwargs['pop_size']]


# #non elite
# def select(mutants, old_population, **kwargs):
#     return sorted(mutants, key=lambda x: x.value)[:kwargs['pop_size']]


def classic_evolution(f, init_pop, **kwargs) -> "tuple(list[DataPoint], DataPoint)":
    logs = []
    t = 0
    population = init_pop
    leader = min(population, key=lambda x: x.value)
    while not stop_condition(t, **kwargs):
        newborns = reproduce(population, **kwargs)
        mutants = mutate(newborns, f, **kwargs)
        candidate = min(mutants, key=lambda x: x.value)
        if candidate.value <= leader.value:
            leader = candidate
        population = select(mutants, population, **kwargs)
        logs.append(DataPoint(t, population))
        t += 1
    return logs, leader
