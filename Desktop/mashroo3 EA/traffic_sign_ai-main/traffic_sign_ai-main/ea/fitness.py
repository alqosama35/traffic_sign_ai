import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time

def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val, subsample_size=None):
    """
    Evaluate a single chromosome's fitness.
    chromosome   : bool array of length n (True = feature selected)
    Returns      : float (validation accuracy on selected features)
    """
    mask = chromosome.astype(bool)

    # Invalid chromosome — no features selected
    if mask.sum() == 0:
        return 0.0

    # Select only the features this chromosome includes
    X_tr = X_train[:, mask]
    X_va = X_val[:, mask]

    # Optionally subsample training rows for speed
    if subsample_size is not None and subsample_size < X_tr.shape[0]:
        idx = np.random.choice(X_tr.shape[0], subsample_size, replace=False)
        X_tr = X_tr[idx]
        y_tr = y_train[idx]
    else:
        y_tr = y_train

    # Train classifier and evaluate
    clf = LinearSVC(C=0.1, max_iter=2000)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_va)
    return accuracy_score(y_val, preds)


def benchmark_fitness_eval(X_train, y_train, X_val, y_val):
    """
    Time a single fitness evaluation with a random chromosome.
    Run this once to decide if subsampling is needed.
    """
    n = X_train.shape[1]
    chromosome = np.random.randint(0, 2, size=n).astype(bool)
    # Ensure at least one feature selected
    if not chromosome.any():
        chromosome[0] = True

    print(f"Benchmarking fitness eval with {chromosome.sum()} features selected...")
    t0 = time.time()
    acc = evaluate_fitness(chromosome, X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0
    print(f"Accuracy: {acc:.4f} | Time: {elapsed:.2f}s")
    print(f"Estimated time per generation (50 evals): {elapsed * 50:.1f}s")
    return elapsed