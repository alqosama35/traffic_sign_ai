import numpy as np


def bit_flip_mutation(chromosome: np.ndarray, rate: float = 0.01) -> np.ndarray:
    """Flip each bit independently with probability rate (FR-EA-09).

    Guarantees the result is never all-zero: if every bit ends up False,
    one random position is forced to True.
    """
    flip_mask = np.random.random(len(chromosome)) < rate
    mutated = chromosome ^ flip_mask
    if not mutated.any():
        mutated[np.random.randint(len(mutated))] = True
    return mutated


def swap_mutation(chromosome: np.ndarray, rate: float = 0.01) -> np.ndarray:
    """For each bit, with probability rate swap it with a randomly chosen other bit.

    On a binary chromosome a swap is only meaningful when the two positions
    differ (one 0 and one 1); otherwise it is a no-op.  This preserves the
    total number of selected features, so it explores *which* features are
    selected rather than *how many* — a qualitatively different search
    neighbourhood compared to bit-flip mutation (FR-EA-09b).

    Guarantees the result is never all-zero: if every bit ends up False,
    one random position is forced to True.
    """
    mutated = chromosome.copy()
    n = len(mutated)
    for i in range(n):
        if np.random.random() < rate:
            j = np.random.randint(n)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    if not mutated.any():
        mutated[np.random.randint(n)] = True
    return mutated


def apply_mutation(population: np.ndarray, mutation_fn, **kwargs) -> np.ndarray:
    """Apply mutation_fn independently to every chromosome in population."""
    return np.array([mutation_fn(chrom, **kwargs) for chrom in population])
