from dataclasses import dataclass, field
from typing import Callable, Optional
from ea.operators.selection import select_parents_tournament, roulette_wheel_selection
from ea.operators.crossover import single_point_crossover, uniform_crossover
from ea.operators.mutation import bit_flip_mutation, swap_mutation
from ea.operators.survivor import generational_replacement, elitism


@dataclass
class ExperimentConfig:
    run_id:        str
    selection:     str
    crossover:     str
    mutation:      str
    survivor:      str
    selection_fn:  Callable
    crossover_fn:  Callable
    mutation_fn:   Callable
    survivor_fn:   Callable
    pop_size:      int            = 50
    max_generations: int          = 100
    mutation_rate: float          = 0.01
    crossover_rate: float         = 0.8
    sigma_share:   Optional[float] = None   # set to 0.2 * n at runtime
    alpha:         float          = 1.0
    seed:          int            = 42
    subsample_size: Optional[int] = None    # set if LinearSVC is too slow


# ── 4 Mandatory Configurations ───────────────────────────────────────────────

MANDATORY_CONFIGS = [
    ExperimentConfig(
        run_id      = "tournament_singlepoint_generational_bitflip_42",
        selection   = "tournament",
        crossover   = "single_point",
        mutation    = "bit_flip",
        survivor    = "generational",
        selection_fn = select_parents_tournament,
        crossover_fn = single_point_crossover,
        mutation_fn  = bit_flip_mutation,
        survivor_fn  = generational_replacement,
        subsample_size = 5000,
        pop_size     = 20,
        max_generations = 30,
    ),
    ExperimentConfig(
        run_id      = "tournament_uniform_generational_bitflip_42",
        selection   = "tournament",
        crossover   = "uniform",
        mutation    = "bit_flip",
        survivor    = "generational",
        selection_fn = select_parents_tournament,
        crossover_fn = uniform_crossover,
        mutation_fn  = bit_flip_mutation,
        survivor_fn  = generational_replacement,
        subsample_size = 5000,
        pop_size     = 20,
        max_generations = 30,
    ),
    ExperimentConfig(
        run_id      = "roulette_singlepoint_generational_bitflip_42",
        selection   = "roulette",
        crossover   = "single_point",
        mutation    = "bit_flip",
        survivor    = "generational",
        selection_fn = roulette_wheel_selection,
        crossover_fn = single_point_crossover,
        mutation_fn  = bit_flip_mutation,
        survivor_fn  = generational_replacement,
        subsample_size = 5000,
        pop_size     = 20,
        max_generations = 30,
    ),
    ExperimentConfig(
        run_id      = "roulette_uniform_generational_bitflip_42",
        selection   = "roulette",
        crossover   = "uniform",
        mutation    = "bit_flip",
        survivor    = "generational",
        selection_fn = roulette_wheel_selection,
        crossover_fn = uniform_crossover,
        mutation_fn  = bit_flip_mutation,
        survivor_fn  = generational_replacement,
        subsample_size = 5000,
        pop_size     = 20,
        max_generations = 30,
    ),
]