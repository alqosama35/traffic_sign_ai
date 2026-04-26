import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def evaluate_fitness(
    chromosome: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subsample_size: int | None = None,
) -> float:
    """Evaluate a chromosome's fitness as LinearSVC val accuracy.

    Args:
        chromosome: bool array of shape (n,) — True means feature is selected.
        subsample_size: if set, randomly sample this many training rows before
            fitting. Pass pre-subsampled X_train/y_train from the GA loop when
            you want the same subsample across all chromosomes in a generation.

    Returns:
        Validation accuracy in [0, 1], or 0.0 for an all-zero chromosome.
    """
    mask = chromosome.astype(bool)
    if mask.sum() == 0:
        return 0.0

    Xtr = X_train[:, mask]
    ytr = y_train

    if subsample_size is not None and subsample_size < len(Xtr):
        idx = np.random.choice(len(Xtr), size=subsample_size, replace=False)
        Xtr = Xtr[idx]
        ytr = ytr[idx]

    Xva = X_val[:, mask]

    clf = LinearSVC(C=0.1, max_iter=2000)
    clf.fit(Xtr, ytr)
    return float(accuracy_score(y_val, clf.predict(Xva)))


def benchmark_fitness_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Time a single fitness evaluation with a random ~50%-feature chromosome.

    Prints the result and warns if the time exceeds the 1.2s/chromosome budget
    (NFR-PERF-04: <60s per generation ÷ 50 chromosomes).

    Returns:
        Elapsed seconds for the single evaluation.
    """
    rng = np.random.default_rng(42)
    chromosome = rng.integers(0, 2, size=X_train.shape[1]).astype(bool)

    t0 = time.time()
    acc = evaluate_fitness(chromosome, X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0

    selected = int(chromosome.sum())
    total = len(chromosome)
    print(f"Benchmark  |  time: {elapsed:.2f}s  |  accuracy: {acc:.4f}  |  features: {selected}/{total}")

    budget = 60.0 / 50
    if elapsed > budget:
        print(
            f"WARNING: {elapsed:.2f}s > {budget:.2f}s budget. "
            "Consider passing subsample_size (8000-10000) to evaluate_fitness."
        )

    return elapsed
