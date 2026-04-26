"""Experiment configuration dataclass and the 4 mandatory GA configurations."""

from dataclasses import dataclass, field
from typing import Callable

from ea.operators.crossover import single_point_crossover, uniform_crossover
from ea.operators.mutation import bit_flip_mutation, swap_mutation
from ea.operators.selection import roulette_wheel_selection, select_parents_tournament
from ea.operators.survivor import elitism, generational_replacement


@dataclass
class ExperimentConfig:
    # Identity & operator names (used in JSON logs)
    run_id: str
    selection: str
    crossover: str
    mutation: str
    survivor: str

    # Operator callables (resolved at construction — no hard-coded strings in GA loop)
    selection_fn: Callable
    crossover_fn: Callable
    mutation_fn: Callable
    survivor_fn: Callable

    # GA hyper-parameters
    pop_size: int = 35
    max_generations: int = 100
    mutation_rate: float = 0.01       # FR-EA-09: exactly p=0.01 per bit
    crossover_rate: float = 0.8
    alpha: float = 1.0
    seed: int = 42

    # Set after feature extraction: sigma_share = 0.2 * n
    sigma_share: float | None = None

    # Set if LinearSVC is too slow (Assumption A3); None means use full training set
    subsample_size: int | None = None


def get_mandatory_configs() -> list[ExperimentConfig]:
    """Return the 4 mandatory experiment configurations (FR-EA-12).

    sigma_share is left None here; run_experiments.py sets it to 0.2 * n
    after loading the feature arrays.
    """
    return [
        ExperimentConfig(
            run_id="tournament_singlepoint_generational_bitflip_42",
            selection="tournament",
            crossover="single_point",
            mutation="bit_flip",
            survivor="generational",
            selection_fn=select_parents_tournament,
            crossover_fn=single_point_crossover,
            mutation_fn=bit_flip_mutation,
            survivor_fn=generational_replacement,
        ),
        ExperimentConfig(
            run_id="tournament_uniform_generational_bitflip_42",
            selection="tournament",
            crossover="uniform",
            mutation="bit_flip",
            survivor="generational",
            selection_fn=select_parents_tournament,
            crossover_fn=uniform_crossover,
            mutation_fn=bit_flip_mutation,
            survivor_fn=generational_replacement,
        ),
        ExperimentConfig(
            run_id="roulette_singlepoint_generational_bitflip_42",
            selection="roulette",
            crossover="single_point",
            mutation="bit_flip",
            survivor="generational",
            selection_fn=roulette_wheel_selection,
            crossover_fn=single_point_crossover,
            mutation_fn=bit_flip_mutation,
            survivor_fn=generational_replacement,
        ),
        ExperimentConfig(
            run_id="roulette_uniform_generational_bitflip_42",
            selection="roulette",
            crossover="uniform",
            mutation="bit_flip",
            survivor="generational",
            selection_fn=roulette_wheel_selection,
            crossover_fn=uniform_crossover,
            mutation_fn=bit_flip_mutation,
            survivor_fn=generational_replacement,
        ),
    ]


def get_extension_configs() -> list[ExperimentConfig]:
    """Return 3 Phase-6 extension configs (FR-EA-09b, FR-EA-10b).

    Two configs isolate swap_mutation vs bit_flip (all other operators held
    constant).  One config isolates elitism vs generational replacement.
    sigma_share is left None here; run_experiments.py sets it to 0.2 * n.
    """
    return [
        # ── Mutation comparison: swap_mutation (tournament + single_point baseline)
        ExperimentConfig(
            run_id="tournament_singlepoint_generational_swapmutation_42",
            selection="tournament",
            crossover="single_point",
            mutation="swap",
            survivor="generational",
            selection_fn=select_parents_tournament,
            crossover_fn=single_point_crossover,
            mutation_fn=swap_mutation,
            survivor_fn=generational_replacement,
        ),
        # ── Mutation comparison: swap_mutation (roulette + uniform baseline)
        ExperimentConfig(
            run_id="roulette_uniform_generational_swapmutation_42",
            selection="roulette",
            crossover="uniform",
            mutation="swap",
            survivor="generational",
            selection_fn=roulette_wheel_selection,
            crossover_fn=uniform_crossover,
            mutation_fn=swap_mutation,
            survivor_fn=generational_replacement,
        ),
        # ── Survivor selection comparison: elitism (tournament + single_point baseline)
        ExperimentConfig(
            run_id="tournament_singlepoint_elitism5_bitflip_42",
            selection="tournament",
            crossover="single_point",
            mutation="bit_flip",
            survivor="elitism5",
            selection_fn=select_parents_tournament,
            crossover_fn=single_point_crossover,
            mutation_fn=bit_flip_mutation,
            survivor_fn=lambda pop, off, rf, of: elitism(pop, off, rf, of, elite_size=5),
        ),
    ]
