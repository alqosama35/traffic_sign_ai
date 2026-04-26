import numpy as np


def tournament_selection(
    population: np.ndarray,
    fitnesses: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Select one chromosome via k-tournament: sample k, return the fittest."""
    indices = np.random.choice(len(population), size=k, replace=False)
    winner = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()


def select_parents_tournament(
    population: np.ndarray,
    fitnesses: np.ndarray,
    n_parents: int,
    k: int = 3,
) -> np.ndarray:
    """Return n_parents chromosomes selected by independent tournaments."""
    fitnesses = np.asarray(fitnesses)
    return np.array([tournament_selection(population, fitnesses, k) for _ in range(n_parents)])


def roulette_wheel_selection(
    population: np.ndarray,
    fitnesses: np.ndarray,
    n_parents: int,
) -> np.ndarray:
    """Fitness-proportionate selection with numerical stabilisation.

    Stabilisation: shift fitnesses up by (min + eps) so all weights are
    strictly positive even when some chromosomes score 0.0.
    """
    fitnesses = np.asarray(fitnesses, dtype=float)
    weights = fitnesses - fitnesses.min() + 1e-8
    probs = weights / weights.sum()
    indices = np.random.choice(len(population), size=n_parents, p=probs, replace=True)
    return population[indices].copy()
