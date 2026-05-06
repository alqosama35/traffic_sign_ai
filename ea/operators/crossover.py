import numpy as np


def single_point_crossover(parent1, parent2):
    """
    Split two parents at a random point and swap their tails.
    parent1, parent2 : bool arrays of length n
    Returns          : (child1, child2)
    
    Example:
    parent1 = [1,1,1,1,1]
    parent2 = [0,0,0,0,0]
    point   = 3
    child1  = [1,1,1,0,0]
    child2  = [0,0,0,1,1]
    """
    n = len(parent1)
    point = np.random.randint(1, n)  # random split point between 1 and n-1

    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])

    return child1, child2


def uniform_crossover(parent1, parent2, p=0.5):
    """
    For each bit position, randomly decide which parent it comes from.
    parent1, parent2 : bool arrays of length n
    p                : probability of swapping each bit
    Returns          : (child1, child2)
    """
    n = len(parent1)
    child1 = parent1.copy()
    child2 = parent2.copy()

    # For each position, flip a coin — if heads, swap that bit
    mask = np.random.rand(n) < p
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    return child1, child2


def apply_crossover(parents, crossover_fn, crossover_rate=0.8):
    """
    Apply crossover to consecutive pairs of parents.
    80% of the time: apply crossover and produce new children
    20% of the time: copy parents unchanged
    Returns : offspring array same shape as parents
    """
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        p1 = parents[i]
        p2 = parents[i + 1]

        if np.random.rand() < crossover_rate:
            c1, c2 = crossover_fn(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        offspring.extend([c1, c2])

    # If odd number of parents, carry the last one unchanged
    if len(parents) % 2 == 1:
        offspring.append(parents[-1].copy())

    return np.array(offspring)