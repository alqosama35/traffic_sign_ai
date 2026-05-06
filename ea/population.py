import numpy as np

def init_population(pop_size, n, seed=42):
    """
    Create initial population of binary chromosomes.
    pop_size : number of individuals (50)
    n        : chromosome length (1280 features)
    Returns  : ndarray shape (pop_size, n) dtype bool
    """
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(pop_size, n)).astype(bool)

    # Ensure no all-zero chromosomes (would crash the fitness function)
    for i in range(pop_size):
        if not population[i].any():
            population[i, rng.integers(0, n)] = True

    return population