import numpy as np


def hamming_distances(population: np.ndarray) -> np.ndarray:
    """Return pairwise Hamming distance matrix of shape (pop_size, pop_size).

    Vectorised: avoids an explicit double loop by broadcasting along two new
    axes, then summing the boolean disagreements over the feature axis.
    """
    return np.sum(population[:, None, :] != population[None, :, :], axis=2)


def sharing_function(d: np.ndarray | float, sigma_share: float, alpha: float = 1.0):
    """Triangular sharing kernel: 1 - (d/sigma_share)^alpha, floored at 0.

    Works on scalars and arrays alike (uses np.maximum for element-wise clamp).
    sh(0) == 1.0 always, so self-contribution is correctly included.
    """
    return np.maximum(0.0, 1.0 - (d / sigma_share) ** alpha)


def apply_fitness_sharing(
    raw_fitnesses: np.ndarray,
    population: np.ndarray,
    sigma_share: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Divide each raw fitness by its niche count (FR-EA-11).

    Niche count for individual i = sum_j sh(hamming(i, j)), which includes
    the self term sh(0) = 1.  The result is used for selection only — raw
    fitness is still used for stagnation checks and JSON logging.

    Returns:
        shared_fitnesses: float array of shape (pop_size,), same order as
        raw_fitnesses.
    """
    raw_fitnesses = np.asarray(raw_fitnesses, dtype=float)
    dists = hamming_distances(population)
    sh = sharing_function(dists, sigma_share, alpha)
    niche_counts = sh.sum(axis=1)
    return raw_fitnesses / niche_counts
