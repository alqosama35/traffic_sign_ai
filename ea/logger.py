"""JSON log builder and writer for GA experiment results."""

import json
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays to plain Python types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_log(
    config,
    result: dict,
    full_feature_val_accuracy: float,
    run_time_seconds: float,
    filter_mask: "np.ndarray | None" = None,
) -> dict:
    """Assemble the SRS §5.3 compliant JSON log dict.

    Required fields (§5.3): run_id, selection, crossover, population_size,
    generations_run, fitness_per_generation, best_fitness, best_chromosome,
    selected_feature_indices, total_features, selected_features, reduction_ratio.

    Additional fields improve dashboard usability and reproducibility.

    Args:
        filter_mask: if a variance pre-filter was applied before the GA,
            pass the bool mask of shape (original_dim,) so that
            selected_feature_indices and total_features are reported in
            the original feature space (not the filtered space).
    """
    accuracy_drop = full_feature_val_accuracy - result["best_fitness"]

    if filter_mask is not None:
        original_positions = np.where(filter_mask)[0]
        selected_indices = [int(original_positions[i]) for i in result["selected_feature_indices"]]
        total_features = int(len(filter_mask))
        selected_features = len(selected_indices)
        reduction_ratio = 1.0 - selected_features / total_features
        best_chrom_orig = np.zeros(len(filter_mask), dtype=bool)
        best_chrom_orig[selected_indices] = True
        best_chromosome = best_chrom_orig.tolist()
        filter_info = {
            "applied": True,
            "original_dim": int(len(filter_mask)),
            "filtered_dim": int(filter_mask.sum()),
        }
    else:
        selected_indices = [int(i) for i in result["selected_feature_indices"]]
        total_features = int(result["total_features"])
        selected_features = int(result["selected_features"])
        reduction_ratio = float(result["reduction_ratio"])
        best_chromosome = result["best_chromosome"].tolist()
        filter_info = {"applied": False}

    return {
        # ── Required SRS §5.3 fields ──────────────────────────────────────
        "run_id": config.run_id,
        "selection": config.selection,
        "crossover": config.crossover,
        "population_size": config.pop_size,
        "generations_run": result["generations_run"],
        "fitness_per_generation": [float(f) for f in result["fitness_per_generation"]],
        "best_fitness": float(result["best_fitness"]),
        "best_chromosome": best_chromosome,
        "selected_feature_indices": selected_indices,
        "total_features": total_features,
        "selected_features": selected_features,
        "reduction_ratio": reduction_ratio,
        # ── Recommended additional fields ─────────────────────────────────
        "mutation": config.mutation,
        "mutation_rate": float(config.mutation_rate),
        "crossover_rate": float(config.crossover_rate),
        "survivor_selection": config.survivor,
        "fitness_sharing": {
            "sigma_share": float(config.sigma_share) if config.sigma_share is not None else None,
            "alpha": float(config.alpha),
        },
        "full_feature_val_accuracy": float(full_feature_val_accuracy),
        "accuracy_drop_pp": float(accuracy_drop),
        "seed": int(config.seed),
        "run_time_seconds": float(run_time_seconds),
        "subsample_size": config.subsample_size,
        "feature_filter": filter_info,
    }


def save_log(log_dict: dict, output_dir: str = "ea/results/") -> Path:
    """Write the log dict to <output_dir>/<run_id>.json.

    Creates output_dir if it does not exist.  All numpy types are converted
    by NumpyEncoder so json.dump never raises TypeError.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{log_dict['run_id']}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(log_dict, fh, indent=2, cls=NumpyEncoder)
    return out_path
