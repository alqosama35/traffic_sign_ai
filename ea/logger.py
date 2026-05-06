import json
import os
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy types.
    json.dump cannot serialize np.int64, np.float32 etc by default.
    This converts them to regular Python int/float automatically.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def build_log(config, result, full_feature_val_accuracy):
    """
    Build the JSON log dictionary for one experiment run.
    All field names match the SRS §5.3 schema exactly.
    """
    accuracy_drop = round(full_feature_val_accuracy - result["best_fitness"], 4)

    log = {
        # ── Required SRS §5.3 fields ──
        "run_id":                   config.run_id,
        "selection":                config.selection,
        "crossover":                config.crossover,
        "population_size":          config.pop_size,
        "generations_run":          result["generations_run"],
        "fitness_per_generation":   result["fitness_per_generation"],
        "best_fitness":             result["best_fitness"],
        "best_chromosome":          result["best_chromosome"],
        "selected_feature_indices": result["selected_feature_indices"],
        "total_features":           result["total_features"],
        "selected_features":        result["selected_features"],
        "reduction_ratio":          result["reduction_ratio"],

        # ── Additional recommended fields ──
        "mutation":                 config.mutation,
        "mutation_rate":            config.mutation_rate,
        "crossover_rate":           config.crossover_rate,
        "survivor_selection":       config.survivor,
        "fitness_sharing": {
            "sigma_share":          config.sigma_share,
            "alpha":                config.alpha,
        },
        "full_feature_val_accuracy": full_feature_val_accuracy,
        "accuracy_drop_pp":         accuracy_drop,
        "seed":                     config.seed,
    }

    return log


def save_log(log_dict, output_dir="ea/results/"):
    """
    Save the log dictionary as a formatted JSON file.
    Filename: <run_id>.json
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{log_dict['run_id']}.json")

    with open(path, "w") as f:
        json.dump(log_dict, f, indent=2, cls=NumpyEncoder)

    print(f"Saved log: {path}")
    return path