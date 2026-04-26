"""GA main loop."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ea.diversity import apply_fitness_sharing
from ea.fitness import evaluate_fitness
from ea.operators.crossover import apply_crossover
from ea.operators.mutation import apply_mutation
from ea.population import init_population


def _is_stagnant(best_history: list[float], window: int = 20) -> bool:
    """Return True when best raw fitness has not improved for `window` gens.

    Uses raw (not shared) fitness because shared fitness fluctuates as
    population composition changes even when the best chromosome is stable.
    """
    if len(best_history) <= window:
        return False
    return best_history[-window] == best_history[-1]


def _resolve_workers(n_jobs: int | None) -> int | None:
    """Map n_jobs convention (-1 = all cores) to ThreadPoolExecutor max_workers."""
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs == -1:
        return os.cpu_count()
    return max(1, n_jobs)


def _eval_parallel(
    population: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    desc: str,
    max_workers: int | None,
) -> np.ndarray:
    """Evaluate all chromosomes in parallel with a live tqdm progress bar.

    Uses ThreadPoolExecutor — sklearn's LinearSVC releases the GIL during its
    C-level optimisation, so threads give real parallelism without the pickling
    cost of spawning separate processes.

    X_tr / y_tr are already subsampled by the caller; subsample_size=None is
    passed to evaluate_fitness so no further slicing occurs.
    """
    n = len(population)
    results: list[float | None] = [None] * n

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_fitness, chrom, X_tr, y_tr, X_va, y_va): i
            for i, chrom in enumerate(population)
        }
        with tqdm(
            as_completed(future_to_idx),
            total=n,
            desc=desc,
            unit="chrom",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            for future in pbar:
                idx = future_to_idx[future]
                results[idx] = future.result()
                done = [r for r in results if r is not None]
                pbar.set_postfix({"best": f"{max(done):.4f}", "done": f"{len(done)}/{n}"})

    return np.array(results)


def _save_checkpoint(path: Path, gen: int, population: np.ndarray,
                     best_history: list[float], best_chrom: np.ndarray,
                     global_best_fitness: float) -> None:
    """Save mid-run state so the GA can resume after an interruption."""
    np.savez(
        path,
        population=population,
        best_history=np.array(best_history, dtype=np.float64),
        best_chrom=best_chrom,
        global_best_fitness=np.array([global_best_fitness]),
        next_gen=np.array([gen + 1]),
    )


def _load_checkpoint(path: Path) -> dict | None:
    """Load a checkpoint file. Returns None if the file does not exist."""
    if not path.exists():
        return None
    ckpt = np.load(path, allow_pickle=False)
    return {
        "population":          ckpt["population"],
        "best_history":        ckpt["best_history"].tolist(),
        "best_chrom":          ckpt["best_chrom"],
        "global_best_fitness": float(ckpt["global_best_fitness"][0]),
        "next_gen":            int(ckpt["next_gen"][0]),
    }


def run_ga(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config,
    verbose: bool = True,
    n_jobs: int = -1,
    checkpoint_path: Path | None = None,
) -> dict:
    """Run the genetic algorithm and return a result dict.

    Checkpointing (per-generation):
        After every generation's survivor selection, the full GA state
        (population, best_history, best_chrom, global_best_fitness, next_gen)
        is saved to checkpoint_path.  On startup, if the file exists the run
        resumes from the saved generation rather than restarting from scratch.
        The checkpoint is NOT deleted here — run_experiments.py removes it only
        after the JSON log is successfully written.

    Speed-ups:
        - Subsampling: one random subsample drawn per generation, shared across
          all chromosome evaluations in that generation.
        - Parallel evaluation via ThreadPoolExecutor.

    Returns dict keys:
        best_chromosome, fitness_per_generation, generations_run, best_fitness,
        selected_feature_indices, total_features, selected_features,
        reduction_ratio
    """
    np.random.seed(config.seed)
    max_workers = _resolve_workers(n_jobs)
    n = X_train.shape[1]

    # ── Resume from checkpoint if available ───────────────────────────────────
    start_gen = 0
    population = init_population(config.pop_size, n, config.seed)
    best_history: list[float] = []
    best_chrom: np.ndarray | None = None
    global_best_fitness = -1.0

    if checkpoint_path is not None:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt is not None:
            population          = ckpt["population"]
            best_history        = ckpt["best_history"]
            best_chrom          = ckpt["best_chrom"]
            global_best_fitness = ckpt["global_best_fitness"]
            start_gen           = ckpt["next_gen"]
            print(
                f"\n  [RESUME] Checkpoint found — continuing from generation "
                f"{start_gen + 1}/{config.max_generations} "
                f"(GlobalBest so far: {global_best_fitness:.4f})"
            )

    for gen in range(start_gen, config.max_generations):
        t_gen = time.time()

        # ── Print generation header ───────────────────────────────────────────
        if verbose:
            stagnant_tag = " [stagnant]" if _is_stagnant(best_history) else ""
            print(
                f"\n  -- Gen {gen + 1:>3}/{config.max_generations} "
                f"| GlobalBest so far: {global_best_fitness:.4f}{stagnant_tag}"
            )

        # ── Fix subsample once per generation ─────────────────────────────────
        if config.subsample_size and config.subsample_size < len(X_train):
            idx = np.random.choice(len(X_train), size=config.subsample_size, replace=False)
            X_tr = X_train[idx]
            y_tr = y_train[idx]
        else:
            X_tr, y_tr = X_train, y_train

        # ── Evaluate parents (parallel) ───────────────────────────────────────
        raw_fitnesses = _eval_parallel(
            population, X_tr, y_tr, X_val, y_val,
            desc="    [parents ]",
            max_workers=max_workers,
        )

        # ── Fitness sharing (selection only, not logged) ───────────────────────
        shared_fitnesses = apply_fitness_sharing(
            raw_fitnesses, population, config.sigma_share, config.alpha
        )

        # ── Track global best on raw fitness ──────────────────────────────────
        best_idx = int(np.argmax(raw_fitnesses))
        gen_best = float(raw_fitnesses[best_idx])

        if gen_best > global_best_fitness:
            global_best_fitness = gen_best
            best_chrom = population[best_idx].copy()

        best_history.append(gen_best)

        # ── Termination ───────────────────────────────────────────────────────
        if _is_stagnant(best_history) or gen == config.max_generations - 1:
            if verbose:
                reason = "stagnation" if _is_stagnant(best_history) else "max generations"
                selected = int(best_chrom.sum())
                print(
                    f"    Stopped ({reason}) | Best: {gen_best:.4f} | "
                    f"Selected: {selected}/{n} | Time: {time.time()-t_gen:.1f}s"
                )
            break

        # ── Selection -> crossover -> mutation ────────────────────────────────
        parents  = config.selection_fn(population, shared_fitnesses, config.pop_size)
        offspring = apply_crossover(parents, config.crossover_fn, config.crossover_rate)
        offspring = apply_mutation(offspring, config.mutation_fn, rate=config.mutation_rate)

        # ── Evaluate offspring (parallel, same subsample) ─────────────────────
        offspring_fitnesses = _eval_parallel(
            offspring, X_tr, y_tr, X_val, y_val,
            desc="    [offspring]",
            max_workers=max_workers,
        )

        population = config.survivor_fn(
            population, offspring, raw_fitnesses, offspring_fitnesses
        )

        if verbose:
            selected = int(best_chrom.sum())
            print(
                f"    Best: {gen_best:.4f} | GlobalBest: {global_best_fitness:.4f} | "
                f"Selected: {selected}/{n} | Time: {time.time()-t_gen:.1f}s"
            )

        # ── Save checkpoint after every completed generation ──────────────────
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path, gen, population,
                best_history, best_chrom, global_best_fitness,
            )

    selected_indices = np.where(best_chrom)[0].tolist()

    return {
        "best_chromosome":      best_chrom,
        "fitness_per_generation": best_history,
        "generations_run":      len(best_history),
        "best_fitness":         global_best_fitness,
        "selected_feature_indices": selected_indices,
        "total_features":       n,
        "selected_features":    len(selected_indices),
        "reduction_ratio":      1.0 - len(selected_indices) / n,
    }
