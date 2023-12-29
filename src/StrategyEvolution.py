from math import sqrt
import numpy as np
import numpy.random as npr
from collections import namedtuple
from random import choices, getrandbits

Entity = namedtuple("Entity", "x value sigma")
DataPoint = namedtuple("DataPoint", "t population")


def stop_condition(t, **kwargs):
    return t >= kwargs['t_max']


#uniform crossover
def crossover(population, **kwargs):
    for _ in range(kwargs['lambda']):
        a, b = choices(population, k=2)
        x = [y[getrandbits(1)] for y in zip(a.x, b.x)]
        sigma = [y[getrandbits(1)] for y in zip(a.sigma, b.sigma)]
        yield x, sigma


# #mean crossover
# def crossover(population, **kwargs):
#     for _ in range(kwargs['lambda']):
#         a, b = choices(population, k=2)
#         weights = npr.normal(size=kwargs['dim'])
#         x = weights * a.x + (1-weights)*b.x
#         sigma = weights * a.sigma + (1-weights)*b.sigma
#         yield x, sigma


def mutate(population, func, **kwargs):
    a = npr.normal()
    b = npr.normal(size=kwargs['dim'])
    tau = 1/sqrt(2*kwargs['dim'])
    taup = 1/sqrt(2*sqrt(kwargs['dim']))
    mutants = []
    for x, sigma in crossover(population, **kwargs):
        sigma = sigma*np.exp(taup*a+tau*b)
        x = x+sigma*npr.normal(size=kwargs['dim'])
        mutants.append(Entity(x, func(x), sigma))
    return mutants


def strategy_evolution(func, init_pop, **kwargs):
    logs = []
    t = 0
    population = init_pop
    leader = min(population, key=lambda x: x.value)
    while not stop_condition(t, **kwargs):
        population = choices(population, k=kwargs['lambda'])
        mutants = mutate(population, func, **kwargs)
        candidate = min(mutants, key=lambda x: x.value)
        if candidate.value <= leader.value:
            leader = candidate
        # mutants.extend(population)
        population = sorted(mutants, key=lambda x: x.value)[:kwargs['mi']]
        logs.append(DataPoint(t, population))
        t += 1
    return logs, leader
