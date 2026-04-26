import numpy as np


def init_population(pop_size: int, n: int, seed: int = 42) -> np.ndarray:
    """Return a boolean population array of shape (pop_size, n).

    Each bit is Bernoulli(0.5). Any all-zero chromosome gets one random bit
    flipped to True so no chromosome is the empty-feature set.
    """
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(pop_size, n)).astype(bool)

    zero_rows = np.where(~population.any(axis=1))[0]
    for i in zero_rows:
        population[i, rng.integers(n)] = True

    return population
