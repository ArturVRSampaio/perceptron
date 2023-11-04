import math


def step(value: float) -> int:
    if value >= 0:
        return 1
    else:
        return 0


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def linear(value: float) -> float:
    return value


def tanh(value: float) -> float:
    return (math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value))
