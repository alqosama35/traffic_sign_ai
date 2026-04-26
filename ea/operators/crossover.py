import numpy as np


def single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split at a uniform random point in (1, n-1) and swap tails."""
    n = len(parent1)
    point = np.random.randint(1, n)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    p: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Swap each bit independently with probability p."""
    swap = np.random.random(len(parent1)) < p
    child1 = np.where(swap, parent2, parent1)
    child2 = np.where(swap, parent1, parent2)
    return child1, child2


def apply_crossover(
    parents: np.ndarray,
    crossover_fn,
    crossover_rate: float = 0.8,
) -> np.ndarray:
    """Apply crossover_fn to consecutive parent pairs.

    Each pair crosses with probability crossover_rate; otherwise both parents
    are copied unchanged. Handles odd-length parent arrays by appending the
    last parent directly.
    """
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.random() < crossover_rate:
            c1, c2 = crossover_fn(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.extend([c1, c2])

    if len(parents) % 2 == 1:
        offspring.append(parents[-1].copy())

    return np.array(offspring)
