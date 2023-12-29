import concurrent.futures
from tqdm import tqdm
import os
from Evolution import classic_evolution, Entity
import numpy.random as npr
from cec2017.simple import *
from StrategyEvolution import strategy_evolution, Entity as SEntity
import pandas as pd
from collections import namedtuple
from random import randint
from src.GradientDescent import gradient_descent
import time


def save_best_result(filename, results, dp):
    splitted_results = [
        dp(x.t, *list(zip(*x[1]))) for x in results
    ]
    pd.DataFrame(splitted_results).to_json(filename)


def save_results(filename, results):
    df = pd.DataFrame(results, columns=['results'])
    df.to_csv(filename)
    print(df['results'].describe())


def cls_evolution(func):
    np.random.seed(randint(0, 123456789))
    return classic_evolution(func, [
        Entity(x, func(x)) for x in [npr.uniform(-100, 100, 10) for _ in range(100)]
    ], **{
        'strength': 2.5,
        'elite': 4,
        'pop_size': 25,
        't_max': 500,
        'dim': 10
    })


def s_evolution(func):
    np.random.seed(randint(0, 123456789))
    init_pop = [SEntity(x, func(x), npr.normal(size=10)) for x in [npr.uniform(-100, 100, 10) for _l in range(1000)]]
    # init_pop = [SEntity(x, func(x), np.full(10, 1.0)) for x in [np.full(10, 100)]]
    return strategy_evolution(func, init_pop, **{
        'mi': 100,
        'lambda': 700,
        'dim': 10,
        't_max': 200
    })


def gd_algorithm(func):
    np.random.seed(randint(0, 123456789))
    x = npr.uniform(-100, 100, 10)
    # x = np.full(10, 100)
    return gradient_descent(func, Entity(x, func(x)), **{
        't_max': 100,
        'grad_epsilon': 1e-35,
        'bt_start': 2.5,
        'bt_alpha': 0.15,
        'bt_beta': 0.5,
        'dim': 10,
    })


def perf_test(alg, func):
    logs = []
    for _ in tqdm(range(100)):
        t0 = time.perf_counter()
        alg(func)
        t1 = time.perf_counter()
        logs.append(t1-t0)
    return logs


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        func = f9
        filename_base = f"gd_9best"
        path = "../results"
        best_leader_val = float("inf")
        jobs = [executor.submit(cls_evolution, func) for _ in range(100)]
        results = []
        for job in tqdm(jobs):
            result, leader = job.result()
            results.append(leader.value)
            if leader.value <= best_leader_val:
                best_result = result
                best_leader_val = leader.value
        dp = namedtuple("dp", "t population_x population_y")
        # dp = namedtuple("dp", "t population_x population_y population_sigma")
        save_results(os.path.join(path, filename_base+"_r.csv"), results)
        save_best_result(os.path.join(path, filename_base+"_br.json"), best_result, dp)
        # print(pd.DataFrame(perf_test(s_evolution, f9)).describe().to_markdown())
        print(best_leader_val)
