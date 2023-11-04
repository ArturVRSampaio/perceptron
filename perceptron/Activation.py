import math


def step_activation(value: float) -> int:
    if value >= 0:
        return 1
    else:
        return 0


def sigmoid(value: float):
    return 1 / (1 + math.exp(-value))
