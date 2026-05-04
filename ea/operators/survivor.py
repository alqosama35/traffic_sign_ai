import numpy as np


def generational_replacement(population, offspring, raw_fitnesses=None, offspring_fitnesses=None):
    """
    Replace entire parent population with offspring.
    The simplest survivor selection — parents are completely discarded.
    """
    return offspring


def elitism(population, offspring, raw_fitnesses, offspring_fitnesses, elite_size=5):
    """
    Keep the best elite_size individuals from the parent population.
    Replace the worst elite_size individuals in offspring with them.
    This guarantees the best solution never gets lost.
    
    population         : ndarray (pop_size, n) — current parents
    offspring          : ndarray (pop_size, n) — new children
    raw_fitnesses      : ndarray (pop_size,)   — parent fitnesses
    offspring_fitnesses: ndarray (pop_size,)   — offspring fitnesses
    elite_size         : how many elites to preserve (default 5)
    """
    # Find the best elite_size parents
    elite_indices = np.argsort(raw_fitnesses)[-elite_size:]
    elites = population[elite_indices]

    # Find the worst elite_size offspring to replace
    worst_indices = np.argsort(offspring_fitnesses)[:elite_size]

    # Replace worst offspring with elite parents
    next_population = offspring.copy()
    next_population[worst_indices] = elites

    return next_population