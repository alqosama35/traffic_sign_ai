import numpy as np


def bit_flip_mutation(chromosome, rate=0.01):
    """
    Flip each bit independently with probability rate.
    chromosome : bool array of length n
    rate       : probability of flipping each bit (0.01 = 1%)
    Returns    : mutated chromosome
    """
    mutated = chromosome.copy()

    # For each bit, flip it with probability rate
    flip_mask = np.random.rand(len(mutated)) < rate
    mutated[flip_mask] = ~mutated[flip_mask]

    # Safety check — if all bits are 0, flip one random bit to 1
    if not mutated.any():
        mutated[np.random.randint(len(mutated))] = True

    return mutated


def swap_mutation(chromosome, rate=0.01):
    """
    For each bit, with probability rate, swap it with another random bit.
    Preserves the number of selected features — only changes which ones.
    chromosome : bool array of length n
    rate       : probability of swapping each bit
    Returns    : mutated chromosome
    """
    mutated = chromosome.copy()
    n = len(mutated)

    for i in range(n):
        if np.random.rand() < rate:
            j = np.random.randint(n)
            # Only swap if the two positions are different
            if mutated[i] != mutated[j]:
                mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


def apply_mutation(population, mutation_fn, **kwargs):
    """
    Apply mutation function to every chromosome in the population.
    population : ndarray (pop_size, n)
    mutation_fn: the mutation function to use
    Returns    : mutated population
    """
    return np.array([mutation_fn(chrom, **kwargs) for chrom in population])