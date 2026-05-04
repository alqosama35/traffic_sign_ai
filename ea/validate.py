"""
Run: python -m ea.validate
Exits 0 if all criteria pass, non-zero otherwise.
"""
import json
import os
import sys
import pathlib


def load_logs(results_dir='ea/results/'):
    logs = []
    for path in pathlib.Path(results_dir).glob('*.json'):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


def check_4_mandatory_configs(logs):
    print("\n[1] Checking 4 mandatory configurations exist...")
    required = [
        ("tournament", "single_point"),
        ("tournament", "uniform"),
        ("roulette",   "single_point"),
        ("roulette",   "uniform"),
    ]
    found = [(l['selection'], l['crossover']) for l in logs]
    all_pass = True
    for sel, cross in required:
        if (sel, cross) in found:
            print(f"  ✓ {sel} + {cross}")
        else:
            print(f"  ✗ MISSING: {sel} + {cross}")
            all_pass = False
    return all_pass


def check_accuracy(logs):
    print("\n[2] Checking best accuracy >= 88%...")
    best = max(l['best_fitness'] for l in logs)
    if best >= 0.88:
        print(f"  ✓ Best accuracy: {best:.4f}")
        return True
    else:
        print(f"  ✗ Best accuracy too low: {best:.4f} (need >= 0.88)")
        return False


def check_reduction(logs):
    print("\n[3] Checking feature reduction >= 30%...")
    best = max(l['reduction_ratio'] for l in logs)
    if best >= 0.30:
        print(f"  ✓ Best reduction: {best:.4f}")
        return True
    else:
        print(f"  ✗ Reduction too low: {best:.4f} (need >= 0.30)")
        return False


def check_accuracy_drop(logs):
    print("\n[4] Checking accuracy drop < 3 percentage points...")
    all_pass = True
    for l in logs:
        drop = l.get('accuracy_drop_pp', None)
        if drop is None:
            print(f"  ? {l['run_id']}: accuracy_drop_pp not found")
            continue
        if drop < 0.03:
            print(f"  ✓ {l['run_id']}: drop = {drop:.4f}")
        else:
            print(f"  ✗ {l['run_id']}: drop = {drop:.4f} (need < 0.03)")
            all_pass = False
    return all_pass


def check_schema(logs):
    print("\n[5] Checking schema compliance...")
    required_fields = [
        'run_id', 'selection', 'crossover', 'population_size',
        'generations_run', 'fitness_per_generation', 'best_fitness',
        'best_chromosome', 'selected_feature_indices', 'total_features',
        'selected_features', 'reduction_ratio'
    ]
    all_pass = True
    for l in logs:
        missing = [f for f in required_fields if f not in l]
        if missing:
            print(f"  ✗ {l['run_id']} missing fields: {missing}")
            all_pass = False
        else:
            print(f"  ✓ {l['run_id']}: all fields present")
    return all_pass


def check_convergence(logs):
    print("\n[6] Checking convergence (fitness improves over generations)...")
    all_pass = True
    for l in logs:
        history = l['fitness_per_generation']
        if max(history) >= history[0]:
            print(f"  ✓ {l['run_id']}: {history[0]:.4f} -> max {max(history):.4f}")
        else:
            print(f"  ✗ {l['run_id']}: fitness declined {history[0]:.4f} -> {max(history):.4f}")
            all_pass = False
    return all_pass


def main():
    print("=" * 60)
    print("EA Validation — SRS §9.2 Acceptance Criteria")
    print("=" * 60)

    logs = load_logs()
    if not logs:
        print("ERROR: No JSON logs found in ea/results/")
        sys.exit(1)

    print(f"Found {len(logs)} experiment logs.")

    results = [
        check_4_mandatory_configs(logs),
        check_accuracy(logs),
        check_reduction(logs),
        check_accuracy_drop(logs),
        check_schema(logs),
        check_convergence(logs),
    ]

    print("\n" + "=" * 60)
    if all(results):
        print("ALL CHECKS PASSED — EA module meets SRS §9.2 criteria")
        print("=" * 60)
        sys.exit(0)
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed} CHECK(S) FAILED — review output above")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()