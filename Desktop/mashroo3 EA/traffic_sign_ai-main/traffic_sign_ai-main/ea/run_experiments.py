import numpy as np
import os
import time
from ea.experiment import MANDATORY_CONFIGS
from ea.ga import run_ga
from ea.logger import build_log, save_log
from ea.fitness import benchmark_fitness_eval


def main():
    # ── Load feature arrays ──────────────────────────────────────────────────
    print("Loading feature arrays...")
    X_train = np.load("ea/data/X_train.npy")
    y_train = np.load("ea/data/y_train.npy")
    X_val   = np.load("ea/data/X_val.npy")
    y_val   = np.load("ea/data/y_val.npy")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")

    # ── Load baseline accuracy ───────────────────────────────────────────────
    with open("ea/data/baseline_accuracy.txt") as f:
        baseline_acc = float(f.read().strip())
    print(f"Full-feature baseline accuracy: {baseline_acc:.4f}")

    # ── Benchmark one fitness eval ───────────────────────────────────────────
    print("\nBenchmarking fitness evaluation speed...")
    benchmark_fitness_eval(X_train, y_train, X_val, y_val)

    # ── Set sigma_share for all configs ─────────────────────────────────────
    n = X_train.shape[1]
    sigma_share = 0.2 * n
    for config in MANDATORY_CONFIGS:
        config.sigma_share = sigma_share

    # ── Run all 4 experiments ────────────────────────────────────────────────
    summary = []

    for config in MANDATORY_CONFIGS:
        print(f"\n{'#'*60}")
        print(f"# Starting: {config.run_id}")
        print(f"{'#'*60}")

        t0 = time.time()
        result = run_ga(X_train, y_train, X_val, y_val, config)
        elapsed = time.time() - t0

        # Build and save JSON log
        log = build_log(config, result, baseline_acc)
        log["run_time_seconds"] = round(elapsed, 1)
        save_log(log)

        summary.append({
            "run_id":          config.run_id,
            "best_fitness":    result["best_fitness"],
            "reduction_ratio": result["reduction_ratio"],
            "generations_run": result["generations_run"],
            "run_time_seconds": round(elapsed, 1),
        })

        print(f"\nCompleted in {elapsed/3600:.2f} hours")

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run ID':<50} {'Accuracy':>10} {'Reduction':>10} {'Gens':>6}")
    print(f"{'-'*80}")
    for s in summary:
        print(f"{s['run_id']:<50} {s['best_fitness']:>10.4f} "
              f"{s['reduction_ratio']:>10.4f} {s['generations_run']:>6}")
    print(f"{'='*80}")
    print(f"\nBaseline accuracy: {baseline_acc:.4f}")
    print(f"All logs saved to ea/results/")


if __name__ == "__main__":
    main()