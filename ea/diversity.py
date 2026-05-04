import numpy as np


def hamming_distances(population):
    """
    Compute pairwise Hamming distances between all chromosomes.
    population : ndarray (pop_size, n)
    Returns    : ndarray (pop_size, pop_size)
    
    Hamming distance = number of positions where two chromosomes differ.
    Example:
    a = [1, 0, 1, 1]
    b = [1, 1, 1, 0]
    distance = 2  (positions 1 and 3 differ)
    """
    # Vectorized — compares every pair at once without a loop
    dists = np.sum(population[:, None] != population[None, :], axis=2)
    return dists


def sharing_function(d, sigma_share, alpha=1):
    """
    Returns how much two individuals share their fitness.
    d           : Hamming distance between two chromosomes
    sigma_share : niche radius — how far apart before no sharing
    alpha       : shape of the sharing curve (1 = linear)
    
    If d < sigma_share : they are in the same niche → sharing applies
    If d >= sigma_share: they are far apart → no sharing (returns 0)
    """
    if d < sigma_share:
        return 1 - (d / sigma_share) ** alpha
    return 0.0


def apply_fitness_sharing(raw_fitnesses, population, sigma_share, alpha=1):
    """
    Reduce fitness of chromosomes that are too similar to each other.
    This forces the GA to maintain diversity — similar chromosomes
    compete with each other instead of dominating the whole population.
    
    raw_fitnesses : ndarray (pop_size,)
    population    : ndarray (pop_size, n)
    sigma_share   : niche radius (use 0.2 * n as default)
    Returns       : shared_fitnesses ndarray (pop_size,)
    """
    pop_size = len(population)
    dists = hamming_distances(population)
    shared_fitnesses = np.zeros(pop_size)

    for i in range(pop_size):
        # Sum up sharing values between individual i and everyone else
        niche_count = sum(
            sharing_function(dists[i, j], sigma_share, alpha)
            for j in range(pop_size)
        )
        # Divide raw fitness by niche count
        # If many similar chromosomes exist, fitness gets heavily reduced
        shared_fitnesses[i] = raw_fitnesses[i] / niche_count

    return shared_fitnesses