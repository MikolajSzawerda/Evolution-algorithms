from math import isnan, isinf
from autograd import grad
import autograd.numpy as np
import autograd.numpy.random as npr
from cec2017.simple import *
from collections import namedtuple


Entity = namedtuple("Entity", "x value")
DataPoint = namedtuple("DataPoint", "t population")


def validate_number(number):
    return not (isnan(number) or isinf(number))


def stop_condition(t, value, grad_value, **kwargs):
    return validate_number(value)\
           and np.linalg.norm(grad_value) > kwargs['grad_epsilon']\
           and t <= kwargs['t_max']


def backtrack_search(func, entity: Entity, grad_val, **kwargs):
    step = kwargs['bt_start']
    arg = entity.x - np.dot(step, grad_val)
    grad_bound = kwargs['bt_alpha'] * np.linalg.norm(grad_val) ** 2
    change = step * grad_bound
    while func(arg) > entity.value - change:
        step *= kwargs['bt_beta']
        arg = entity.x - np.dot(step, grad_val)
        change = step * grad_bound
    return step


def gradient_descent(func, entity, **kwargs):
    logs = []
    t = 0
    gradient = grad(func)
    grad_value = gradient(entity.x)
    while stop_condition(t, entity.value, grad_value, **kwargs):
        step = backtrack_search(func, entity, grad_value, **kwargs)
        x = entity.x - np.dot(step, grad_value)
        grad_value = gradient(x)
        entity = Entity(x, func(x))
        logs.append(DataPoint(t, [entity]))
        t += 1
    return logs, entity


if __name__ == "__main__":
    func = f1
    results = gradient_descent(func, npr.uniform(-100, 100, 10), **{
        't_max': 200,
        'grad_epsilon': 1e-20,
        'bt_start': 1.0,
        'bt_alpha': 0.15,
        'bt_beta': 0.5,
        'dim': 10,
        'reruns': 10
    })
