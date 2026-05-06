import numpy as np


def tournament_selection(population, fitnesses, k=3):
    """
    Pick k random individuals, return the one with highest fitness.
    population : ndarray (pop_size, n)
    fitnesses  : ndarray (pop_size,)
    k          : tournament size
    Returns    : one chromosome (ndarray of length n)
    """
    indices = np.random.choice(len(population), size=k, replace=False)
    best = indices[np.argmax(fitnesses[indices])]
    return population[best]


def select_parents_tournament(population, fitnesses, n_parents, k=3):
    """
    Run tournament selection n_parents times to get a full parent array.
    Returns : ndarray (n_parents, n)
    """
    parents = []
    for _ in range(n_parents):
        parent = tournament_selection(population, fitnesses, k)
        parents.append(parent)
    return np.array(parents)


def roulette_wheel_selection(population, fitnesses, n_parents):
    """
    Fitness-proportionate selection.
    Chromosomes with higher fitness get a higher chance of being picked.
    Returns : ndarray (n_parents, n)
    """
    # Stabilize — shift fitnesses so minimum is just above 0
    shifted = fitnesses - fitnesses.min() + 1e-8

    # Convert to probabilities that sum to 1
    probs = shifted / shifted.sum()

    # Pick n_parents indices according to those probabilities
    indices = np.random.choice(len(population), size=n_parents, p=probs, replace=True)
    return population[indices]  