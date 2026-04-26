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


def apply_mutation(population: np.ndarray, mutation_fn, **kwargs) -> np.ndarray:
    """Apply mutation_fn independently to every chromosome in population."""
    return np.array([mutation_fn(chrom, **kwargs) for chrom in population])
