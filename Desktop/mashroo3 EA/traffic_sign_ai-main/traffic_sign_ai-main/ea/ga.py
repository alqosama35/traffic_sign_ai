import numpy as np
from ea.population import init_population
from ea.fitness import evaluate_fitness
from ea.diversity import apply_fitness_sharing
from ea.operators.crossover import apply_crossover
from ea.operators.mutation import apply_mutation


def run_ga(X_train, y_train, X_val, y_val, config):
    """
    Main GA loop.
    Runs the full genetic algorithm and returns results as a dict.
    """
    np.random.seed(config.seed)
    n = X_train.shape[1]

    # Set sigma_share if not already set
    if config.sigma_share is None:
        config.sigma_share = 0.2 * n

    # Initialize population
    population = init_population(config.pop_size, n, config.seed)
    best_history = []

    print(f"\n{'='*60}")
    print(f"Starting run: {config.run_id}")
    print(f"Selection: {config.selection} | Crossover: {config.crossover}")
    print(f"Pop size: {config.pop_size} | Max generations: {config.max_generations}")
    print(f"{'='*60}\n")

    for gen in range(config.max_generations):

        # ── Step 1: Evaluate raw fitness for every chromosome ──
        raw_fitnesses = np.array([
            evaluate_fitness(chrom, X_train, y_train, X_val, y_val,
                           config.subsample_size)
            for chrom in population
        ])

        # ── Step 2: Apply fitness sharing ──
        shared_fitnesses = apply_fitness_sharing(
            raw_fitnesses, population, config.sigma_share, config.alpha
        )

        # ── Step 3: Track best raw fitness ──
        best_idx = np.argmax(raw_fitnesses)
        best_fitness = raw_fitnesses[best_idx]
        best_history.append(float(best_fitness))

        # Count selected features in best chromosome
        n_selected = int(population[best_idx].sum())

        print(f"Gen {gen+1:3d}/{config.max_generations} | "
              f"Best: {best_fitness:.4f} | "
              f"Features: {n_selected}/{n} | "
              f"Mean: {raw_fitnesses.mean():.4f}")

        # ── Step 4: Termination check ──
        if gen >= 20 and best_history[-20] == best_history[-1]:
            print(f"\nEarly stopping — no improvement for 20 generations.")
            break

        # ── Step 5: Selection (uses shared fitness) ──
        parents = config.selection_fn(population, shared_fitnesses, config.pop_size)

        # ── Step 6: Crossover ──
        offspring = apply_crossover(parents, config.crossover_fn, config.crossover_rate)

        # ── Step 7: Mutation ──
        offspring = apply_mutation(offspring, config.mutation_fn, rate=config.mutation_rate)

        # ── Step 8: Evaluate offspring fitness ──
        offspring_fitnesses = np.array([
            evaluate_fitness(chrom, X_train, y_train, X_val, y_val,
                           config.subsample_size)
            for chrom in offspring
        ])

        # ── Step 9: Survivor selection ──
        population = config.survivor_fn(
            population, offspring,
            raw_fitnesses, offspring_fitnesses
        )

    # ── Final result ──
    final_fitnesses = np.array([
        evaluate_fitness(chrom, X_train, y_train, X_val, y_val,
                        config.subsample_size)
        for chrom in population
    ])
    best_idx = np.argmax(final_fitnesses)
    best_chrom = population[best_idx]
    selected_indices = np.where(best_chrom)[0].tolist()

    return {
        "best_chromosome":        best_chrom,
        "fitness_per_generation": best_history,
        "generations_run":        len(best_history),
        "best_fitness":           float(final_fitnesses[best_idx]),
        "selected_feature_indices": selected_indices,
        "total_features":         n,
        "selected_features":      len(selected_indices),
        "reduction_ratio":        round(1.0 - len(selected_indices) / n, 4),
    }