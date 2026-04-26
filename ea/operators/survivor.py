import numpy as np


def generational_replacement(
    population: np.ndarray,
    offspring: np.ndarray,
    raw_fitnesses: np.ndarray | None = None,
    offspring_fitnesses: np.ndarray | None = None,
) -> np.ndarray:
    """Full generational replacement: discard parents, return offspring.

    The fitness arguments are accepted but ignored so that both survivor
    functions share the same call signature in the GA loop.
    """
    return offspring.copy()


def elitism(
    population: np.ndarray,
    offspring: np.ndarray,
    raw_fitnesses: np.ndarray,
    offspring_fitnesses: np.ndarray,
    elite_size: int = 5,
) -> np.ndarray:
    """Preserve the top elite_size parents by overwriting the weakest offspring.

    Guarantees max(next_population_fitnesses) >= max(parent_fitnesses), so
    best fitness is monotonically non-decreasing across generations.
    """
    raw_fitnesses = np.asarray(raw_fitnesses)
    offspring_fitnesses = np.asarray(offspring_fitnesses)

    elite_indices = np.argsort(raw_fitnesses)[-elite_size:]
    worst_offspring_indices = np.argsort(offspring_fitnesses)[:elite_size]

    next_population = offspring.copy()
    next_population[worst_offspring_indices] = population[elite_indices]
    return next_population
